"""Daily e-bus scheduling and disruption repair: Qollab entry point.

This effort is supported by Qollab & IonQ.

A city runs a 5-bus battery-electric schedule (King County Metro route 150, 200 kWh
batteries, charging during the day). This script first compares MILP, exact QUBO
search, simulated annealing, tabu search and QAOA on the uninterrupted schedule.
Then one bus breaks down. Every affected trip must go to another bus, a reserve
bus or be cancelled, while batteries and chargers stay feasible. The repair part:

  1. builds the repair problem (one qubit per trip option),
  2. solves it classically: exact optimum, greedy dispatcher rule, simulated annealing,
  3. optimizes QAOA angles in an exact simulator, for the textbook penalty QAOA (X mixer)
     and for the XY-mixer QAOA that never leaves the valid "one option per trip" states,
  4. runs ONE XY-QAOA circuit on the selected backend and decodes the measured repairs.

Both comparisons are logged in results/main_bus_<bus>_<time>.json.

On Qollab, press Run: the platform injects `backend` (built-in simulator or the IonQ
simulator; IonQ accepts one circuit per run, which is exactly what we send).
Locally:  python main.py                     (k585 fails at 07:00, 14 qubits)
          python main.py --bus k4 --at 06:00  (20 qubits)
          python main.py --shots 2000 --p 2
          python main.py --schedule-only       (no disruption)
          python main.py --repair-only         (skip daily-schedule comparison)
"""

from __future__ import annotations

import sys
import time
import json
import math
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parent
except NameError:  # some browser runtimes execute main.py without __file__
    ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from evsp.classical_baselines import exact_repair, greedy_repair  # noqa: E402
from evsp.columns import solve_milp  # noqa: E402
from evsp.demo import load_demo_data  # noqa: E402
from evsp.disruption import Disruption, apply_assignment, build_repair_problem  # noqa: E402
from evsp.qaoa import circuit_metrics, run_qaoa  # noqa: E402
from evsp.qubo import build_qubo, variant_weights  # noqa: E402
from evsp.solvers import Landscape, index_to_bits, simulated_annealing, tabu_search  # noqa: E402
from evsp.xy_qaoa import OneHotSpace, best_valid_state, run_xy_qaoa, xy_qaoa_circuit  # noqa: E402

try:
    backend  # noqa: B018  injected by Qollab's Run button
except NameError:
    backend = None


def hhmm(m: float) -> str:
    m = int(round(m))
    return f"{m // 60:02d}:{m % 60:02d}"


def minutes(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def option_arg(name: str, default):
    argv = [a for a in sys.argv[1:] if a]
    if name in argv and argv.index(name) + 1 < len(argv):
        return type(default)(argv[argv.index(name) + 1])
    return default


def get_backend():
    """Qollab's injected backend, else a local Qiskit simulator."""
    if backend is not None:
        return backend, getattr(backend, "name", type(backend).__name__)
    from qiskit.providers.basic_provider import BasicSimulator
    return BasicSimulator(), "local BasicSimulator"


def run_circuit(qc, shots: int, seed: int = 0):
    from qiskit import transpile

    be, name = get_backend()
    tqc = transpile(qc, backend=be, optimization_level=1, seed_transpiler=seed)
    t0 = time.time()
    job = be.run(tqc, shots=shots)
    counts = job.result().get_counts()
    if isinstance(counts, list):              # some providers return one dict per circuit
        counts = counts[0]
    return counts, name, time.time() - t0


def decode(key: str, n: int) -> np.ndarray:
    bits = key.replace(" ", "")[::-1]          # Qiskit keys are big-endian: qubit 0 is the last character
    return np.array([int(b) for b in bits[:n]], dtype=int)


def describe_plan(inst, plan) -> list[str]:
    tb = inst.trip_by_id
    lines = []
    for trip, target in sorted(plan.assignment.items(), key=lambda kv: tb[kv[0]].dep_min):
        t = tb[trip]
        what = "CANCELLED" if target == "cancel" else f"-> {target}" + (" (reserve)" if target.startswith("R") else "")
        lines.append(f"    trip {trip} {hhmm(t.dep_min)}-{hhmm(t.arr_min)} {what}")
    return lines


def draw(inst, sched, event, plan, path: Path) -> None:
    """Before/after Gantt chart with matplotlib (skipped if matplotlib is missing)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    tb = inst.trip_by_id
    from dataclasses import replace

    after = {b: c for b, c in sched.duties.items() if b != event.bus}
    failed = sched.duties[event.bus]
    done = [t for t in failed.trips if tb[t].arr_min <= event.at_min]
    if done:                                   # the failed bus keeps what it drove before the failure
        after[event.bus] = replace(failed, trips=done, charges=[c for c in failed.charges if c.start_min < event.at_min])
    after.update({b: d for b, d in plan.duties.items() if d is not None})
    moved = {t: v for t, v in plan.assignment.items() if v != "cancel"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharex=True)
    for ax, title, duties in ((axes[0], "Before the failure", sched.duties), (axes[1], "After the XY-QAOA repair", after)):
        rows = sorted(duties, key=lambda b: (b.startswith("R"), duties[b].start_min))
        for y, bus in enumerate(rows):
            col = duties[bus]
            for tid in col.trips:
                t = tb[tid]
                if ax is axes[0] and bus == event.bus and t.dep_min >= event.at_min:
                    color, label = "#d03b3b", "lost"
                elif tid in moved and ax is axes[1]:
                    color, label = "#eb6834", "moved"
                else:
                    color, label = "#2a78d6", ""
                ax.barh(y, t.arr_min - t.dep_min, left=t.dep_min, color=color, edgecolor="white", height=0.6)
                if label:
                    ax.text((t.dep_min + t.arr_min) / 2, y, label, ha="center", va="center", color="white", fontsize=8)
            for ch in col.charges:
                ax.barh(y, ch.end_min - ch.start_min, left=ch.start_min, color="#1baf7a", edgecolor="white", height=0.6)
                ax.text((ch.start_min + ch.end_min) / 2, y, ch.station, ha="center", va="center", color="white", fontsize=7)
        if ax is axes[0]:
            ax.axvline(event.at_min, color="#d03b3b", linestyle="--", linewidth=1)
            ax.text(event.at_min, len(rows) - 0.4, f" {event.bus} fails", color="#d03b3b", fontsize=9)
        ax.set_yticks(range(len(rows)), rows)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=12)
        ticks = list(range(5 * 60, 16 * 60, 120))
        ax.set_xticks(ticks, [hhmm(t) for t in ticks])
        ax.grid(axis="x", color="#e9e8e4")
        ax.set_axisbelow(True)
    cancelled = [t for t, v in plan.assignment.items() if v == "cancel"]
    fig.text(0.01, 0.01, "blue: trip   green: charging (station)   orange/'moved': reassigned trip   "
             f"red/'lost': trip of the failed bus.   Cancelled: {', '.join(cancelled) or 'none'}",
             fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=120)
    try:
        plt.show()
    except Exception:
        pass
    print(f"\nFigure: {path.name} (before/after schedule)")


def schedule_comparison(d, p_max: int, shots: int) -> dict:
    """Run all daily-schedule methods on the bundled duty pool (no disruption)."""
    cs = d.cs
    q = build_qubo(cs, variant_weights("E4", cs.assumptions, R=1.0))
    print(f"\nDaily schedule without disruption: {q.n} duties, E4 (R=1)")
    t0 = time.time()
    milp = solve_milp(cs)
    milp_seconds = time.time() - t0
    print(f"MILP: {milp.status}, cost {milp.objective:.2f}, {milp.n_vehicles} buses ({milp_seconds:.1f}s)")
    t0 = time.time()
    land = Landscape.from_qubo(q)
    exact_seconds = time.time() - t0
    exact_x = index_to_bits(int(land.optimum_indices[0]), q.n)
    sa = simulated_annealing(q, reads=200, sweeps=500, seed=0, landscape=land)
    tabu = tabu_search(q, reads=20, steps=500, seed=0)
    print(f"E4 exact energy {land.optimum_energy:.2f}; SA P(optimum) {sa.p_optimal:.3f}; "
          f"tabu best energy {tabu.best_energy:.2f}")
    res = run_qaoa(q, land, p_max=p_max, shots=shots, seed=0)
    best = max(res.layers, key=lambda layer: layer.p_optimal)
    print(f"Schedule QAOA p={best.p}: P(optimum) {best.p_optimal:.4f}, "
          f"P(feasible) {best.p_feasible:.4f}")

    def evaluated(x):
        return q.evaluate(x)

    milp_x = np.zeros(q.n, dtype=int)
    milp_x[milp.columns] = 1
    best_raw = max(res.layers, key=lambda layer: layer.p_optimal).sampled["best_raw_index"]

    return {"columns": d.columns_path.name, "variant": "E4", "R": 1.0,
            "milp": {"objective": "duty cost, without the E4 peak term", "status": milp.status,
                     "cost": milp.objective, "vehicles": milp.n_vehicles,
                     "columns": [cs.columns[i].id for i in milp.columns], "seconds": milp_seconds,
                     "e4_evaluation": evaluated(milp_x)},
            "qubo_exact": {"energy": land.optimum_energy, "seconds": exact_seconds,
                           "schedule": evaluated(exact_x)},
            "simulated_annealing": {"best_energy": sa.best_energy, "p_optimal": sa.p_optimal,
                                    "p_feasible": sa.p_feasible, "seconds": sa.seconds,
                                    "schedule": evaluated(sa.best_x)},
            "tabu_search": {"best_energy": tabu.best_energy, "seconds": tabu.seconds,
                            "schedule": evaluated(tabu.best_x)},
            "qaoa": {**res.to_dict(), "best_raw_schedule": evaluated(index_to_bits(best_raw, q.n))}}


def save_results(out: dict, bus: str, at: int) -> Path:
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if math.isfinite(value) else None  # no feasible QAOA sample
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    directory = ROOT / "results"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"main_bus_{bus}_{hhmm(at).replace(':', '')}.json"
    path.write_text(json.dumps(json_safe(out), indent=2, default=float, allow_nan=False), encoding="utf-8")
    print(f"Results: {path}")
    return path


def main() -> None:
    bus = option_arg("--bus", "k585")
    at = minutes(option_arg("--at", "07:00"))
    shots = option_arg("--shots", 1000)
    p_max = option_arg("--p", 2)
    repair_only = "--repair-only" in sys.argv
    schedule_only = "--schedule-only" in sys.argv
    if repair_only and schedule_only:
        raise ValueError("--repair-only and --schedule-only cannot be combined")

    print("E-bus schedule repair with XY-mixer QAOA")
    print("This effort is supported by Qollab & IonQ.\n")

    d = load_demo_data()
    inst, sched = d.inst, d.schedule
    out = {"settings": {"bus": bus, "at": hhmm(at), "shots": shots, "p_max": p_max},
           "daily_schedule": None, "disruption_repair": None}
    if not repair_only:
        out["daily_schedule"] = schedule_comparison(d, p_max, shots)
        save_results(out, bus, at)
    if schedule_only:
        return
    print(f"Schedule: {len(sched.duties)} electric buses, {len(inst.trips)} trips, "
          f"battery {inst.assumptions['battery_kwh']:.0f} kWh, King County Metro route 150")
    for b, c in sorted(sched.duties.items(), key=lambda kv: kv[1].start_min):
        stops = ", ".join(f"{ch.station} {hhmm(ch.start_min)}" for ch in c.charges) or "-"
        print(f"  {b:<5} {len(c.trips)} trips {hhmm(c.start_min)}-{hhmm(c.end_min)}  charging: {stops}")

    event = Disruption.bus_failure(sched, bus, at)
    rp = build_repair_problem(inst, sched, event, reserves=3)
    space = OneHotSpace.from_problem(rp)
    print(f"\nEvent: bus {bus} fails at {hhmm(at)}. {len(rp.trips)} trips affected -> {rp.n} qubits "
          f"(one per trip option), {space.size} valid one-hot assignments out of 2^{rp.n} = {2 ** rp.n:,}.")

    # ---------------------------------------------------------------- classical
    t0 = time.time()
    exact = exact_repair(inst, rp)              # true cost: replays every valid assignment
    best_plan = exact[1] if exact else best_valid_state(inst, rp, space)[1]
    t_exact = time.time() - t0
    t0 = time.time()
    _, greedy_plan = greedy_repair(inst, rp)
    t_greedy = time.time() - t0
    q = rp.to_qubo()
    sa = simulated_annealing(q, reads=200, sweeps=500, seed=0)
    opt_mask = space.optimum_mask()
    tol = 1e-6 * max(1.0, abs(space.optimum_energy))
    sa_p = float(np.mean(sa.energies <= space.optimum_energy + tol))
    print(f"\nClassical: exact optimum extra cost {best_plan.extra_cost:.0f} ({t_exact:.2f}s), "
          f"greedy dispatcher {greedy_plan.extra_cost:.0f} ({t_greedy * 1000:.0f} ms), "
          f"simulated annealing reaches the QUBO ground state in {sa_p:.0%} of 200 runs")
    print("\n".join(describe_plan(inst, best_plan)))
    repair_log = {"event": {"bus": bus, "at": hhmm(at)}, "qubits": rp.n,
                  "exact_repair": {"extra_cost": best_plan.extra_cost, "seconds": t_exact,
                                   "assignment": best_plan.assignment},
                  "greedy": {"extra_cost": greedy_plan.extra_cost, "seconds": t_greedy},
                  "simulated_annealing": {"p_optimal": sa_p, "best_energy": sa.best_energy,
                                          "reads": 200, "seconds": sa.seconds}}

    # ---------------------------------------------------------------- QAOA angles (exact simulation)
    print(f"\nOptimizing QAOA angles (exact simulation, p = 1..{p_max}) ...")
    xy = run_xy_qaoa(space, p_max=p_max)
    xy_best = max(xy.layers, key=lambda L: L.p_optimal)
    pen = None
    if rp.n <= 18:                              # 2^n statevector; keep the browser run quick
        land = Landscape.from_qubo(q)
        pen = max(run_qaoa(q, land, p_max=p_max, shots=1000, seed=0).layers, key=lambda L: L.p_optimal)
    repair_log["xy_qaoa"] = xy.to_dict()
    repair_log["penalty_qaoa"] = None if pen is None else {"p": pen.p, "p_optimal": pen.p_optimal,
                                                           "p_feasible": pen.p_feasible}

    # ---------------------------------------------------------------- one XY-QAOA circuit on the backend
    qc = xy_qaoa_circuit(rp, xy_best.gammas, xy_best.betas, space)
    m = circuit_metrics(qc, "all")
    print(f"XY-QAOA circuit p={xy_best.p}: {m['qubits']} qubits, depth {m['depth']}, {m['cx']} CX after transpilation")
    counts, be_name, secs = run_circuit(qc, shots)
    total = sum(counts.values())
    valid = optimal = 0
    tally: dict[tuple, int] = {}
    one_hot = {tuple(space.bits(s)): s for s in range(space.size)}
    for key, c in counts.items():
        x = tuple(decode(key, rp.n))
        s = one_hot.get(x)
        if s is None:
            continue
        valid += c
        optimal += c * int(opt_mask[s])
        tally[x] = tally.get(x, 0) + c
    print(f"Ran {total} shots on {be_name} in {secs:.1f}s: {valid / total:.1%} valid repairs, "
          f"{optimal / total:.1%} optimal (exact prediction {xy_best.p_optimal:.1%})")
    repair_log["backend_measurement"] = {"backend": be_name, "shots": total, "seconds": secs,
                                         "p_valid": valid / total, "p_optimal": optimal / total,
                                         "counts": counts}

    # ---------------------------------------------------------------- results
    rows = [(f"random guess (2^{rp.n} bitstrings)", f"{opt_mask.sum() / 2 ** rp.n:.2e}", f"{space.size / 2 ** rp.n:.2e}")]
    if pen is not None:
        rows.append((f"QAOA, X mixer + penalties, p={pen.p}", f"{pen.p_optimal:.4f}", f"{pen.p_feasible:.3f}"))
    rows += [(f"QAOA, XY mixer, p={xy_best.p} (exact)", f"{xy_best.p_optimal:.4f}", f"{xy_best.p_feasible:.3f}"),
             ("QAOA, XY mixer, measured", f"{optimal / total:.4f}", f"{valid / total:.3f}"),
             ("simulated annealing (200 runs)", f"{sa_p:.4f}", "-")]
    print(f"\n{'method':<36} | {'P(optimum)':>10} | {'P(valid repair)':>15}")
    for name, po, pv in rows:
        print(f"{name:<36} | {po:>10} | {pv:>15}")

    if tally:
        # QAOA is used as a sampler: replay the most frequent measured repairs, keep the cheapest
        plans = [(apply_assignment(inst, rp, np.array(x)), x) for x in sorted(tally, key=tally.get, reverse=True)[:200]]
        plan, top = min(((pl, x) for pl, x in plans if pl.feasible), key=lambda px: px[0].extra_cost,
                        default=plans[0])
        gap = plan.extra_cost - best_plan.extra_cost
        print(f"\nBest repair among the {total} shots (seen {tally[top]} times): extra cost {plan.extra_cost:.0f} "
              f"({'= exact optimum' if gap < 1e-6 else f'+{gap:.0f} vs exact optimum'}), battery/charger replay "
              f"{'OK' if plan.feasible else 'FAILED'}")
        print("\n".join(describe_plan(inst, plan)))
        if plan.feasible:
            draw(inst, sched, event, plan, ROOT / "repair.png")
        repair_log["best_measured_repair"] = {"extra_cost": plan.extra_cost, "feasible": plan.feasible,
                                              "assignment": plan.assignment}
    out["disruption_repair"] = repair_log
    save_results(out, bus, at)
    print("\nThis effort is supported by Qollab & IonQ.")


if __name__ == "__main__":
    main()

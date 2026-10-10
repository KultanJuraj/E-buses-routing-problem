"""Repair a schedule after a disruption and compare XY-mixer QAOA with penalty QAOA and SA.

Usage (from the project root):
    python scripts/run_repair.py --bus k4 --at 06:00
    python scripts/run_repair.py --bus k4 --at 06:00 --qiskit
    python scripts/run_repair.py --bus k4 --at 06:00 --qiskit --noise 0.001,0.01
    python scripts/run_repair.py --charger S3 --from 07:00 --to 09:00

Defaults: columns/kcm_small_b20_bat200.json (5-bus optimum with charging) on
instances/kcm_small.json, 3 reserve buses, up to 3 other buses per trip.
Penalty QAOA (X mixer) runs only while 2^n fits in memory (n <= 22); the XY
mixer is simulated in the one-hot subspace and also handles larger cases.
Writes results/repair_<event>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evsp import load_instance  # noqa: E402
from evsp.columns import load_columns  # noqa: E402
from evsp.disruption import Disruption, Schedule, apply_assignment, build_repair_problem  # noqa: E402
from evsp.qaoa import circuit_metrics, qaoa_circuit, run_qaoa, sample_with_qiskit, shots_for_confidence  # noqa: E402
from evsp.solvers import Landscape, simulated_annealing  # noqa: E402
from evsp.xy_qaoa import OneHotSpace, XYQAOA, run_xy_qaoa, xy_qaoa_circuit  # noqa: E402


def minutes(text: str) -> float:
    h, m = text.split(":")
    return int(h) * 60 + int(m)


def hhmm(m: float) -> str:
    m = int(round(m))
    return f"{m // 60:02d}:{m % 60:02d}"


def describe_plan(inst, plan) -> list[str]:
    tb = inst.trip_by_id
    lines = []
    for t, target in sorted(plan.assignment.items(), key=lambda kv: tb[kv[0]].dep_min):
        trip = tb[t]
        lines.append(f"    trip {t} {hhmm(trip.dep_min)}-{hhmm(trip.arr_min)} -> {target}")
    for target, duty in sorted(plan.duties.items()):
        if duty is not None:
            ch = ", ".join(f"{c.station} {hhmm(c.start_min)}-{hhmm(c.end_min)}" for c in duty.charges) or "no charging"
            lines.append(f"    {target}: {len(duty.trips)} trips {hhmm(duty.start_min)}-{hhmm(duty.end_min)}, "
                         f"min SoC {duty.min_soc:.0%} [{ch}]")
    return lines


def _best_of_shots(energies, feasible, probs, shots, seed) -> float:
    """Lowest feasible energy among `shots` samples (what a real run of that many shots returns)."""
    idx = np.random.default_rng(seed).choice(len(probs), size=shots, p=probs / probs.sum())
    ok = idx[feasible[idx]]
    return float(energies[ok].min()) if ok.size else float("inf")


def run_probs(land, layer):
    from evsp.qaoa import FastQAOA
    return FastQAOA(land).probabilities(layer.gammas, layer.betas)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--columns", type=Path, default=ROOT / "columns" / "kcm_small_b20_bat200.json")
    p.add_argument("--instance", type=Path, default=ROOT / "instances" / "kcm_small.json")
    p.add_argument("--bus", help="failed bus id (e.g. k4)")
    p.add_argument("--at", help="failure time HH:MM")
    p.add_argument("--charger", help="station out of service (e.g. S3)")
    p.add_argument("--from", dest="start", help="outage start HH:MM")
    p.add_argument("--to", dest="end", help="outage end HH:MM")
    p.add_argument("--reserves", type=int, default=3)
    p.add_argument("--max-options", type=int, default=3, help="other buses considered per trip")
    p.add_argument("--p-max", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--maxiter", type=int, default=150)
    p.add_argument("--sa-reads", type=int, default=200)
    p.add_argument("--no-penalty-qaoa", action="store_true", help="skip the slow X-mixer comparison")
    p.add_argument("--qiskit", action="store_true", help="sample the XY circuit with Qiskit (n <= 24)")
    p.add_argument("--noise", help="1q,2q depolarizing error for --qiskit (needs qiskit-aer)")
    p.add_argument("--shots", type=int, default=4000)
    p.add_argument("--best-of", type=int, default=1000, help="shots for the best-of-N comparison")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "results")
    args = p.parse_args()

    cs = load_columns(args.columns)
    inst = load_instance(args.instance)
    inst.assumptions.update(cs.assumptions)           # same battery / costs as the columns were built with
    sched = Schedule.from_columns(cs)
    if args.bus:
        if not args.at:
            p.error("--bus needs --at")
        event = Disruption.bus_failure(sched, args.bus, minutes(args.at))
    elif args.charger:
        if not (args.start and args.end):
            p.error("--charger needs --from and --to")
        event = Disruption.charger_outage(args.charger, minutes(args.start), minutes(args.end))
    else:
        p.error("give --bus/--at or --charger/--from/--to")

    print(f"schedule: {len(sched.duties)} buses {sorted(sched.duties)} | event: {event.describe()}")
    rp = build_repair_problem(inst, sched, event, max_bus_options=args.max_options, reserves=args.reserves)
    tb = inst.trip_by_id
    print(f"affected trips: {len(rp.trips)} | qubits: {rp.n} | one-hot states: {rp.subspace_size} of 2^{rp.n} "
          f"| conflicts: {len(rp.conflicts)}")
    for t, g in zip(rp.trips, rp.groups):
        opts = ", ".join(f"{rp.options[k].target or 'cancel'} {rp.options[k].cost:.0f}" for k in g)
        print(f"    {t} {hhmm(tb[t].dep_min)}: {opts}")

    out = {"event": event.__dict__, "trips": rp.trips, "n_qubits": rp.n, "subspace_size": rp.subspace_size,
           "options": [o.__dict__ for o in rp.options], "conflicts": rp.conflicts, "settings": vars(args) | {
               "columns": str(args.columns), "instance": str(args.instance), "out": str(args.out)}}

    # ---- exact optimum in the one-hot subspace
    t = time.time()
    space = OneHotSpace.from_problem(rp)
    best_state = int(np.argmin(space.energies))
    x_opt = space.bits(best_state)
    plan = apply_assignment(inst, rp, x_opt)
    print(f"\nexact optimum ({time.time() - t:.1f}s): QUBO energy {space.optimum_energy:.1f}, true extra cost "
          f"{plan.extra_cost:.1f}, cancelled {len(plan.cancelled)}, jointly feasible {plan.jointly_feasible}, "
          f"charger capacity ok {plan.capacity_ok}")
    print("\n".join(describe_plan(inst, plan)))
    out["optimum"] = {"energy": space.optimum_energy, "extra_cost": plan.extra_cost, "assignment": plan.assignment,
                      "jointly_feasible": plan.jointly_feasible, "capacity_ok": plan.capacity_ok,
                      "problems": plan.problems}

    q = rp.to_qubo()
    rows = []

    # ---- simulated annealing on the penalty QUBO
    sa = simulated_annealing(q, reads=args.sa_reads, sweeps=500, seed=args.seed)
    tol = 1e-6 * max(1.0, abs(space.optimum_energy))
    sa_p = float(np.mean(sa.energies <= space.optimum_energy + tol))
    rows.append(("simulated annealing", sa_p, None, None, sa.best_energy))
    out["simulated_annealing"] = {"p_optimal": sa_p, "best_energy": sa.best_energy, "reads": args.sa_reads}

    # ---- penalty QAOA with the standard X mixer (full 2^n space)
    if rp.n <= 22 and not args.no_penalty_qaoa:
        land = Landscape.from_qubo(q)
        print(f"\npenalty QAOA (X mixer, 2^{rp.n} = {2 ** rp.n} states, {int(land.feasible.sum())} feasible):")
        pq = run_qaoa(q, land, p_max=args.p_max, alpha=args.alpha, maxiter=args.maxiter, shots=args.shots,
                      seed=args.seed, verbose=True)
        best = max(pq.layers, key=lambda L: L.p_optimal)
        rows.append(("QAOA, X mixer + penalties", best.p_optimal, best.p_feasible, best.p,
                     _best_of_shots(land.energies, land.feasible, run_probs(land, best), args.best_of, args.seed)))
        out["penalty_qaoa"] = pq.to_dict()
        pen_qc = qaoa_circuit(q, best.gammas, best.betas, land)
    else:
        print("\npenalty QAOA skipped" + ("" if args.no_penalty_qaoa else f": 2^{rp.n} states do not fit a statevector"))
        rows.append(("QAOA, X mixer + penalties", None, None, None, None))
        pen_qc = qaoa_circuit(q, [0.5], [0.5], std=1.0)

    # ---- XY-mixer QAOA in the one-hot subspace
    print(f"\nXY-mixer QAOA ({space.size} one-hot states):")
    xy = run_xy_qaoa(space, p_max=args.p_max, alpha=args.alpha, maxiter=args.maxiter, verbose=True)
    bxy = max(xy.layers, key=lambda L: L.p_optimal)
    xy_probs = XYQAOA(space).probabilities(bxy.gammas, bxy.betas)
    rows.append(("QAOA, XY mixer", bxy.p_optimal, bxy.p_feasible, bxy.p,
                 _best_of_shots(space.energies, space.feasible, xy_probs, args.best_of, args.seed)))
    out["xy_qaoa"] = xy.to_dict()
    xy_qc = xy_qaoa_circuit(rp, bxy.gammas, bxy.betas, space)

    # ---- circuits
    out["circuits"] = {"xy": [circuit_metrics(xy_qc, c) for c in ("all", "heavy-hex")],
                       "penalty": [circuit_metrics(pen_qc, c) for c in ("all", "heavy-hex")]}
    for name, (c_all, c_hh) in out["circuits"].items():
        label = "penalty (1 layer, angles not optimized)" if name == "penalty" and "penalty_qaoa" not in out else name
        print(f"  {label} circuit: {c_all['cx']} CX, depth {c_all['depth']} (all-to-all) | "
              f"{c_hh['cx']} CX on heavy-hex")

    # ---- optional Qiskit sampling of the XY circuit
    if args.qiskit:
        if rp.n > 24:
            print("Qiskit sampling skipped: more than 24 qubits")
        else:
            n1, n2 = (float(v) for v in args.noise.split(",")) if args.noise else (0.0, 0.0)
            counts = sample_with_qiskit(xy_qc, shots=args.shots, seed=args.seed, noise_1q=n1, noise_2q=n2)
            onehot_idx = {space.full_index(s): s for s in range(space.size)}
            opt_mask = space.optimum_mask()
            shots = sum(counts.values())
            in_space = sum(c for i, c in counts.items() if i in onehot_idx)
            optimal = sum(c for i, c in counts.items() if i in onehot_idx and opt_mask[onehot_idx[i]])
            post = optimal / in_space if in_space else 0.0
            out["qiskit_xy"] = {"noise_1q": n1, "noise_2q": n2, "shots": shots, "frac_one_hot": in_space / shots,
                                "frac_optimal": optimal / shots, "frac_optimal_postselected": post}
            print(f"  Qiskit XY sampling ({shots} shots{', noise ' + args.noise if args.noise else ''}): "
                  f"one-hot {in_space / shots:.3f}, optimal {optimal / shots:.3f}, "
                  f"optimal among one-hot shots (post-selection) {post:.3f}")

    # ---- summary
    print(f"\n  method                     | P(optimum) | shots for 95 % | P(feasible) | best of {args.best_of} "
          f"(gap to optimum)")
    opt_e = space.optimum_energy
    for name, p_opt, p_feas, depth, best_e in rows:
        if p_opt is None:
            print(f"  {name:<26} | {'skipped':>10} |                |             |")
            continue
        gap = (best_e - opt_e) / abs(opt_e) if np.isfinite(best_e) and opt_e else float("inf")
        print(f"  {name:<26} | {p_opt:>10.4f} | {shots_for_confidence(p_opt):>14.0f} | "
              f"{'' if p_feas is None else f'{p_feas:.3f}':>11} | {best_e:.1f} ({gap:+.1%})"
              + (f"  p={depth}" if depth else ""))
    out["summary"] = [{"method": r[0], "p_optimal": r[1], "p_feasible": r[2], "p": r[3], "best_energy": r[4]}
                      for r in rows]
    top = int(np.argmax(xy_probs))
    top_plan = apply_assignment(inst, rp, space.bits(top))
    print(f"\nmost likely XY-QAOA answer: extra cost {top_plan.extra_cost:.1f}, optimal "
          f"{bool(space.optimum_mask()[top])}, jointly feasible {top_plan.jointly_feasible}")
    out["xy_most_likely"] = {"assignment": top_plan.assignment, "extra_cost": top_plan.extra_cost,
                             "optimal": bool(space.optimum_mask()[top])}

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"bus_{event.bus}_{hhmm(event.at_min)}" if event.kind == "bus" else \
        f"charger_{event.station}_{hhmm(event.at_min)}-{hhmm(event.until_min)}"
    path = args.out / f"repair_{tag.replace(':', '')}.json"
    path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"-> {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


if __name__ == "__main__":
    main()

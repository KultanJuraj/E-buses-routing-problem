"""Classical baselines (package D): diesel flow models, electric greedy vs MILP, repair greedy vs exact.

Usage (from the project root):
    python scripts/run_classical.py                          # everything, default instances
    python scripts/run_classical.py --part diesel --instances tiny small medium large
    python scripts/run_classical.py --part electric --battery 200
    python scripts/run_classical.py --part repair

Parts:
  diesel    scenarios S1-S3 of the assignment (no battery) as minimum-cost flow, S1 also by
            bipartite matching, S3 as a trade-off curve over v_max, real GTFS blocks for comparison
  electric  battery check of the diesel optimum, greedy dispatcher, exact MILP over all
            enumerated duties (tiny/small) or the diesel fleet as a lower bound (larger instances)
  repair    every bus failure of the 5-bus demo schedule (05:00-14:00, hourly): greedy repair,
            QUBO optimum replayed (what QAOA targets) and true-cost optimum by full replay
Writes results/baselines.json (a single --part updates only its section).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evsp import load_instance  # noqa: E402
from evsp.classical_baselines import (baseline_blocks, battery_check, exact_repair, greedy_electric, greedy_repair,  # noqa: E402
                            is_valid_cover, scenario1, scenario2, scenario3)


def hhmm(m: float) -> str:
    m = int(round(m))
    return f"{m // 60:02d}:{m % 60:02d}"


def diesel(names: list[str], steps: int) -> list[dict]:
    print("\n== Diesel scenarios (no battery): minimum-cost flow ==")
    print(f"{'instance':<8} {'trips':>5} {'arcs':>7} | {'S1 buses':>8} {'(match)':>7} | {'S2 buses':>8} {'deadhead km':>11} "
          f"{'cost':>8} {'time':>6} | {'real blocks':>11} {'deadhead km':>11} {'bad pairs':>9}")
    out = []
    for name in names:
        inst = load_instance(ROOT / "instances" / f"kcm_{name}.json")
        s1, s2, real = scenario1(inst), scenario2(inst), baseline_blocks(inst)
        assert is_valid_cover(inst, s2.blocks)
        curve = []
        for v in range(s1.vehicles, s1.vehicles + steps + 1):
            s3 = scenario3(inst, v)
            curve.append({"vmax": v, "vehicles": s3.vehicles, "deadhead_km": round(s3.deadhead_km, 2)})
        print(f"{name:<8} {len(inst.trips):>5} {s1.extra['arcs']:>7} | {s1.vehicles:>8} {s1.extra['matching_fleet']:>7} | "
              f"{s2.vehicles:>8} {s2.deadhead_km:>11.1f} {s2.cost:>8.0f} {s2.seconds:>5.2f}s | {real.vehicles:>11} "
              f"{real.deadhead_km:>11.1f} {real.extra['infeasible_pairs']:>9}")
        print("         S3 trade-off (v_max -> deadhead km): "
              + ", ".join(f"{c['vmax']} -> {c['deadhead_km']:.0f}" for c in curve))
        out.append({"instance": inst.name, "trips": len(inst.trips), "S1": s1.summary(), "S2": s2.summary(),
                    "S3": curve, "real_blocks": real.summary()})
    return out


def electric(names: list[str], batteries: list[str]) -> list[dict]:
    from evsp.columns import generate_columns, solve_milp

    print("\n== Electric (battery + charging) ==")
    print(f"{'instance':<8} {'battery':>7} | {'diesel S2 battery-ok':>20} | {'greedy buses':>12} {'cost':>8} {'stops':>5} "
          f"{'time':>6} | {'exact buses':>11} {'cost':>8} {'time':>6} {'gap':>6}")
    out = []
    for name, battery in [(n, b) for n in names for b in batteries]:
        inst = load_instance(ROOT / "instances" / f"kcm_{name}.json")
        if battery != "instance":
            inst.assumptions["battery_kwh"] = float(battery)
        s1, s2 = scenario1(inst), scenario2(inst)
        bc = battery_check(inst, s2.blocks)
        g = greedy_electric(inst)
        row = {"instance": inst.name, "battery_kwh": inst.assumptions["battery_kwh"],
               "diesel_S2_battery_check": bc, "greedy": g.summary(), "lower_bound_buses": s1.vehicles}
        exact = "-"
        if len(inst.trips) <= 40:                    # full enumeration of duties is cheap
            t0 = time.time()
            cs = generate_columns(inst)
            sol = solve_milp(cs)
            secs = time.time() - t0
            row["exact"] = {"method": "MILP over all enumerated duties", "vehicles": sol.n_vehicles,
                            "cost": round(sol.objective, 2), "columns": len(cs.columns), "seconds": round(secs, 3)}
            exact = (f"{sol.n_vehicles:>11} {sol.objective:>8.0f} {secs:>5.2f}s "
                     f"{(g.cost - sol.objective) / sol.objective:>+6.0%}")
        else:
            exact = f"{'>= ' + str(s1.vehicles):>11} (diesel fleet is a lower bound)"
        print(f"{name:<8} {inst.assumptions['battery_kwh']:>7.0f} | {bc['battery_feasible']:>9} of {bc['blocks']:<8} | "
              f"{g.vehicles:>12} {g.cost:>8.0f} {sum(len(c.charges) for c in g.duties):>5} {g.seconds:>5.2f}s | {exact}")
        out.append(row)
    return out


def repair() -> dict:
    from evsp.demo import load_demo_data
    from evsp.disruption import Disruption, build_repair_problem
    from evsp.xy_qaoa import OneHotSpace, best_valid_state

    print("\n== Repair after a bus failure (5-bus demo schedule, 3 reserves) ==")
    print(f"{'event':<16} {'qubits':>6} {'states':>6} | {'true optimum':>12} {'QUBO optimum':>12} {'greedy':>8} | "
          f"{'exact time':>10} {'greedy time':>11}")
    d = load_demo_data()
    inst, sched = d.inst, d.schedule
    rows = []
    for bus in sched.duties:
        for h in range(5, 15):
            try:
                rp = build_repair_problem(inst, sched, Disruption.bus_failure(sched, bus, h * 60), reserves=3)
            except ValueError:
                continue                              # no trips left after the failure
            t0 = time.time()
            ex = exact_repair(inst, rp)
            t_ex = time.time() - t0
            t0 = time.time()
            _, gp = greedy_repair(inst, rp)
            t_gr = time.time() - t0
            _, qp, _ = best_valid_state(inst, rp, OneHotSpace.from_problem(rp))
            r = {"bus": bus, "at": hhmm(h * 60), "qubits": rp.n, "states": rp.subspace_size,
                 "true_optimum": round(ex[1].extra_cost, 2) if ex else None, "qubo_optimum": round(qp.extra_cost, 2),
                 "greedy": round(gp.extra_cost, 2), "exact_seconds": round(t_ex, 3), "greedy_seconds": round(t_gr, 4)}
            rows.append(r)
            print(f"{bus + ' ' + r['at']:<16} {rp.n:>6} {rp.subspace_size:>6} | {r['true_optimum']:>12.0f} "
                  f"{r['qubo_optimum']:>12.0f} {r['greedy']:>8.0f} | {t_ex:>9.2f}s {t_gr:>10.4f}s")
    valid = [r for r in rows if r["true_optimum"] is not None]
    summary = {
        "events": len(rows),
        "greedy_equals_qubo_optimum": sum(abs(r["greedy"] - r["qubo_optimum"]) < 1e-6 for r in rows),
        "greedy_equals_true_optimum": sum(abs(r["greedy"] - r["true_optimum"]) < 1e-6 for r in valid),
        "qubo_equals_true_optimum": sum(abs(r["qubo_optimum"] - r["true_optimum"]) < 1e-6 for r in valid),
        "mean_qubo_gap_to_true": round(float(sum(r["qubo_optimum"] - r["true_optimum"] for r in valid) / len(valid)), 2),
        "max_qubo_gap_to_true": round(float(max(r["qubo_optimum"] - r["true_optimum"] for r in valid)), 2),
    }
    print(f"\n{summary['events']} events: greedy = QUBO optimum in {summary['greedy_equals_qubo_optimum']}, "
          f"greedy = true optimum in {summary['greedy_equals_true_optimum']}, QUBO optimum = true optimum in "
          f"{summary['qubo_equals_true_optimum']} (QUBO gap to true optimum: mean {summary['mean_qubo_gap_to_true']:.0f}, "
          f"max {summary['max_qubo_gap_to_true']:.0f} cost units)")
    return {"events": rows, "summary": summary}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", choices=["all", "diesel", "electric", "repair"], default="all")
    p.add_argument("--instances", nargs="+", default=["tiny", "small", "medium", "large"],
                   choices=["tiny", "small", "medium", "large"])
    p.add_argument("--electric-instances", nargs="+", default=["tiny", "small", "medium"],
                   choices=["tiny", "small", "medium", "large"])
    p.add_argument("--battery", nargs="+", default=["instance", "200"],
                   help="usable battery kWh for the electric part ('instance' = value in the instance file)")
    p.add_argument("--s3-steps", type=int, default=6, help="v_max values above the minimum fleet for S3")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "baselines.json")
    args = p.parse_args()

    out = {}
    if args.out.exists() and args.part != "all":     # keep the other parts of an earlier run
        out = json.loads(args.out.read_text(encoding="utf-8"))
    if args.part in ("all", "diesel"):
        out["diesel"] = diesel(args.instances, args.s3_steps)
    if args.part in ("all", "electric"):
        out["electric"] = electric(args.electric_instances, args.battery)
    if args.part in ("all", "repair"):
        out["repair"] = repair()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n-> {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")


if __name__ == "__main__":
    main()

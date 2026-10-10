"""Generate battery-feasible duties (columns) and select a qubit budget.

Usage (from the project root):
    python scripts/build_columns.py --instance instances/kcm_tiny.json --budget 20
    python scripts/build_columns.py --instance instances/kcm_small.json --budget 20 --battery 200
    python scripts/build_columns.py --instance instances/kcm_medium.json --budget 120 --cg

Writes columns/<instance>_b<budget>.json with the selected columns, their
charger conflicts C and tight pairs r_kl, and prints the optimal schedule
found on the selection.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evsp import load_instance  # noqa: E402
from evsp.columns import (GenConfig, check_schedule, column_generation, generate_columns,  # noqa: E402
                          select_columns, solve_milp, validate_column)


def hhmm(m: float) -> str:
    m = int(round(m))
    return f"{m // 60:02d}:{m % 60:02d}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance", type=Path, required=True)
    p.add_argument("--budget", type=int, default=20, help="number of columns kept (= qubits)")
    p.add_argument("--battery", type=float, default=None, help="override usable battery kWh (e.g. 200 to force charging)")
    p.add_argument("--max-trips", type=int, default=8)
    p.add_argument("--max-successors", type=int, default=4)
    p.add_argument("--cg", action="store_true", help="column generation instead of full enumeration (big instances)")
    p.add_argument("--milp-time", type=float, default=60.0)
    p.add_argument("--out", type=Path, default=ROOT / "columns")
    p.add_argument("--no-depot-charging", action="store_true",
                   help="exclude depot charging when pairwise QUBO cannot represent its capacity")
    args = p.parse_args()

    inst = load_instance(args.instance)
    if args.battery is not None:
        if not math.isfinite(args.battery) or args.battery <= 0:
            p.error("--battery must be finite and positive")
        inst.assumptions["battery_kwh"] = args.battery
    cfg = GenConfig(max_trips=args.max_trips, max_successors=args.max_successors,
                    allow_depot_charging=not args.no_depot_charging)
    print(inst.summary())

    t = time.time()
    cs = column_generation(inst, cfg) if args.cg else generate_columns(inst, cfg)
    invalid = [c.id for c in cs.columns if validate_column(inst, c)]
    print(f"columns: {len(cs.columns)} ({cs.stats['n_with_charging']} with charging) in {time.time() - t:.1f}s, "
          f"invalid on replay: {len(invalid)}")
    if invalid:
        p.error(f"Invalid columns; export stopped: {invalid[:10]}")

    sel = select_columns(cs, budget=args.budget, milp_time_limit=args.milp_time)
    sol = solve_milp(cs, sel.indices, time_limit=args.milp_time)
    chk = check_schedule(cs, sol.columns)
    if not chk["ok"] or not math.isfinite(sol.objective):
        p.error(f"No valid schedule to export: {sol.status}; {chk}")
    conflicts, tight = cs.pairs(sel.indices)
    print(f"LP bound {sel.lp_objective:.1f} | selected {len(sel.indices)} columns, C = {len(conflicts)} pairs, "
          f"r_kl = {len(tight)} pairs")
    print(f"best schedule on the selection: {sol.n_vehicles} buses, cost {sol.objective:.1f}, "
          f"valid = {chk['ok']} (real blocks: {len(inst.baseline_blocks)})")
    for k in sol.columns:
        c = cs.columns[k]
        ch = ", ".join(f"{s.station} {hhmm(s.start_min)}-{hhmm(s.end_min)} {s.soc_before:.0%}->{s.soc_after:.0%}"
                       for s in c.charges) or "no charging"
        print(f"  {c.id:>6}: {hhmm(c.start_min)}-{hhmm(c.end_min)}  {c.n_trips} trips  min SoC {c.min_soc:.0%}  "
              f"{c.deadhead_km:.0f} km deadhead  [{ch}]")

    pos = {k: i for i, k in enumerate(sel.indices)}
    name = f"{inst.name}_b{args.budget}" + (f"_bat{int(args.battery)}" if args.battery else "")
    path = cs.to_json(args.out / f"{name}.json", indices=sel.indices, extra={
        "selection": {"budget": args.budget, "lp_objective": sel.lp_objective,
                      "anchor": [pos[k] for k in sel.anchor if k in pos]},
        "best_on_selection": {"columns": [pos[k] for k in sol.columns], "objective": sol.objective,
                              "n_vehicles": sol.n_vehicles},
        "assumptions": inst.assumptions,
    })
    print(f"-> {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


if __name__ == "__main__":
    main()

"""Build the QUBO (E1-E4) from a columns file and check its ground state.

Usage (from the project root):
    python scripts/build_qubo.py --columns columns/kcm_tiny_b20.json --variant E1
    python scripts/build_qubo.py --columns columns/kcm_small_b20_bat200.json --variant E4 --R 1 --w-rob 1
    python scripts/build_qubo.py --columns columns/kcm_small_b20_bat200.json --variant E3 --V 5

For n <= 26 variables the exact ground state is found by brute force and
compared with the MILP optimum on the same columns (E1/E2 must match exactly;
E3/E4 add terms the MILP does not have). Writes qubo/<columns>_<variant>.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evsp.columns import load_columns, solve_milp  # noqa: E402
from evsp.qubo import brute_force, build_qubo, tune_weights, variant_weights  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--columns", type=Path, required=True)
    p.add_argument("--variant", default="E4", choices=["E1", "E2", "E3", "E4"])
    p.add_argument("--V", type=int, default=None, help="fleet size for E3")
    p.add_argument("--R", type=float, default=1.0, help="peak weight (E3, E4)")
    p.add_argument("--w-rob", type=float, default=1.0, help="charger robustness weight (E4)")
    p.add_argument("--peak-scope", choices=["station", "total"], default="total",
                   help="total matches the grid-wide peak in the project specification")
    p.add_argument("--energy-basis", choices=["tariff", "kwh"], default="tariff",
                   help="tariff preserves MILP parity; kwh uses daytime plus refill grid energy")
    p.add_argument("--normalize-costs", action="store_true", help="normalize duty costs to at most 1")
    p.add_argument("--P", type=float, default=None, help="explicit coverage penalty (default: automatic)")
    p.add_argument("--Q", type=float, default=None, help="explicit charger-conflict penalty (default: automatic)")
    p.add_argument("--penalty-factor", type=float, default=2.0)
    p.add_argument("--tune", action="store_true", help="grid-search R and penalty factor by exact ground state (max 20 variables)")
    p.add_argument("--R-grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    p.add_argument("--penalty-grid", type=float, nargs="+", default=[1.5, 2.0, 3.0])
    p.add_argument("--out", type=Path, default=ROOT / "qubo")
    args = p.parse_args()

    cs = load_columns(args.columns)
    w = variant_weights(args.variant, cs.assumptions, V=args.V, R=args.R, w_rob=args.w_rob)
    w.peak_scope, w.penalty_factor = args.peak_scope, args.penalty_factor
    w.energy_basis, w.normalize_costs = args.energy_basis, args.normalize_costs
    w.P, w.Q = args.P, args.Q
    if args.tune:
        if args.P is not None or args.Q is not None:
            p.error("--tune cannot be combined with explicit --P or --Q")
        if len(cs.columns) > 20:
            p.error("--tune requires at most 20 selected columns")
        w, trials = tune_weights(cs, w, list(range(len(cs.columns))), args.penalty_grid, args.R_grid)
        print(f"weight grid: {len(trials)} settings, selected R={w.R:g}, penalty factor={w.penalty_factor:g}")
        for row in trials:
            print(f"  R={row['R']:g} factor={row['penalty_factor']:g} P=Q={row['P']:.4g} "
                  f"feasible={row['feasible']} peak={row['peak_kw_total']:.1f} kW"
                  + ("  <- selected" if row["selected"] else ""))
    q = build_qubo(cs, w, name=args.columns.stem)
    s = q.stats()
    print(f"{args.columns.name} {args.variant}: {s['n_variables']} variables, {s['n_quadratic']} quadratic terms "
          f"(density {s['density']:.2f}), |coef| {s['coef_min']:.3g}..{s['coef_max']:.3g}, "
          f"P={s['P']:.1f} Q={s['Q']:.1f}" + (f" M={s['M']:.1f}" if s["M"] else ""))

    if q.n > 26:
        print("too many variables for brute force; exporting without an exact ground-state check")
    else:
        t = time.time()
        x, e = brute_force(q)
        rep = q.evaluate(x)
        print(f"ground state (brute force, {time.time() - t:.1f}s): E = {e:.2f}, feasible = {rep['feasible']}, "
              f"{rep['n_vehicles']} buses, cost {rep['cost']:.2f}, peak {rep['peak_kw_total']:.0f} kW total, "
              f"tight charging {rep['tight_minutes']:.1f} min")
        if not rep["feasible"]:
            p.error("ground state is infeasible; increase penalties or tune weights; export stopped")
        print(f"  columns: {rep['columns']}")
        if args.variant == "E1":
            milp = solve_milp(cs)
            if rep["n_vehicles"] != milp.n_vehicles:
                p.error("E1 QUBO ground state does not match the same-pool MILP fleet count; export stopped")
            print(f"E1 fleet count matches same-pool MILP ({milp.n_vehicles} buses)")
        elif args.variant == "E2" and args.energy_basis == "tariff":
            milp = solve_milp(cs)
            if abs(rep["cost"] - milp.objective) >= 1e-6:
                p.error("E2 QUBO ground-state cost does not match the same-pool MILP; export stopped")
            print(f"E2 cost matches same-pool MILP ({milp.objective:.2f})")

    path = q.to_json(args.out / f"{args.columns.stem}_{args.variant}.json")
    print(f"-> {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


if __name__ == "__main__":
    main()

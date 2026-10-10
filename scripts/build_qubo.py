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
from evsp.qubo import brute_force, build_qubo, variant_weights  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--columns", type=Path, required=True)
    p.add_argument("--variant", default="E4", choices=["E1", "E2", "E3", "E4"])
    p.add_argument("--V", type=int, default=None, help="fleet size for E3")
    p.add_argument("--R", type=float, default=1.0, help="peak weight (E3, E4)")
    p.add_argument("--w-rob", type=float, default=1.0, help="charger robustness weight (E4)")
    p.add_argument("--peak-scope", choices=["station", "total"], default="station")
    p.add_argument("--penalty-factor", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=ROOT / "qubo")
    args = p.parse_args()

    cs = load_columns(args.columns)
    w = variant_weights(args.variant, cs.assumptions, V=args.V, R=args.R, w_rob=args.w_rob)
    w.peak_scope, w.penalty_factor = args.peak_scope, args.penalty_factor
    q = build_qubo(cs, w, name=args.columns.stem)
    s = q.stats()
    print(f"{args.columns.name} {args.variant}: {s['n_variables']} variables, {s['n_quadratic']} quadratic terms "
          f"(density {s['density']:.2f}), |coef| {s['coef_min']:.3g}..{s['coef_max']:.3g}, "
          f"P={s['P']:.1f} Q={s['Q']:.1f}" + (f" M={s['M']:.1f}" if s["M"] else ""))

    path = q.to_json(args.out / f"{args.columns.stem}_{args.variant}.json")
    print(f"-> {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")

    if q.n > 26:
        print("too many variables for brute force; skipping the ground-state check")
        return
    t = time.time()
    x, e = brute_force(q)
    rep = q.evaluate(x)
    print(f"ground state (brute force, {time.time() - t:.1f}s): E = {e:.2f}, feasible = {rep['feasible']}, "
          f"{rep['n_vehicles']} buses, cost {rep['cost']:.2f}, peak {rep['peak_kw_total']:.0f} kW total, "
          f"tight charging {rep['tight_minutes']:.1f} min")
    print(f"  columns: {rep['columns']}")
    milp = solve_milp(cs)
    print(f"MILP on the same columns (cost only, exact charger capacity): {milp.n_vehicles} buses, "
          f"cost {milp.objective:.2f}")
    if args.variant in ("E1", "E2"):
        same = abs(rep["cost"] - (milp.objective if args.variant == "E2" else rep["cost"])) < 1e-6
        print(f"  E2 cost matches MILP: {same}" if args.variant == "E2" else
              f"  E1 vehicles match MILP: {rep['n_vehicles'] == milp.n_vehicles}")


if __name__ == "__main__":
    main()

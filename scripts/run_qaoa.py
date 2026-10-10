"""Benchmark exact search, simulated annealing, tabu and optional QAOA on one E-VSP QUBO.

Usage (from the project root):
    python scripts/run_qaoa.py --columns columns/kcm_tiny_b20.json --variant E2
    python scripts/run_qaoa.py --columns columns/kcm_small_b20_bat200.json --variant E4 --p-max 2
    python scripts/run_qaoa.py --columns columns/kcm_tiny_b20.json --variant E2 --qiskit --shots 2000
    python scripts/run_qaoa.py --columns columns/kcm_tiny_b20.json --variant E2 --qiskit --noise 0.001,0.01
    python scripts/run_qaoa.py --columns columns/kcm_tiny_b20.json --variant E2 --optimizer spsa
    python scripts/run_qaoa.py --columns columns/kcm_tiny_b20.json --variant E4 --compare-peak
    python scripts/run_qaoa.py --columns columns/kcm_small_b20_bat200.json --variant E4 --classical-only --compare-peak

Angles are optimized with the exact numpy simulator (same circuit, see evsp/qaoa.py).
--qiskit additionally samples the optimized circuit with Qiskit (optionally with
depolarizing noise, needs qiskit-aer). --scaling repeats the run on the best
k columns for each k to show how QAOA scales with the number of qubits.
Writes JSON results and a matching CSV benchmark table. --classical-only skips
QAOA and circuit construction; exact enumeration supports at most 24 variables.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evsp.columns import load_columns, select_columns  # noqa: E402
from evsp.qaoa import circuit_metrics, counts_metrics, qaoa_circuit, run_qaoa, sample_with_qiskit  # noqa: E402
from evsp.qubo import build_qubo, variant_weights  # noqa: E402
from evsp.solvers import Landscape, index_to_bits, simulated_annealing, tabu_search  # noqa: E402


def build(cs, variant, V, indices=None, R=None):
    w = variant_weights(variant, cs.assumptions, V=V, R=1.0 if R is None else R)
    return build_qubo(cs, w, indices=indices)


def benchmark_row(q, cs, method, x, seconds) -> dict:
    """Evaluate raw solver output; infeasible solutions remain explicitly labelled."""
    report = q.evaluate(x)
    by_id = {c.id: c for c in cs.columns}
    chosen = [by_id[name] for name in report["columns"]]
    return {"instance": cs.instance_name, "variant": q.weights.variant, "R": q.weights.R,
            "qubits": q.n, "method": method, "energy": report["energy"],
            "feasible": report["feasible"], "cost": report["cost"], "vehicles": report["n_vehicles"],
            "deadhead_km": sum(c.deadhead_km for c in chosen),
            "grid_kwh": sum(c.grid_kwh_day + c.refill_kwh for c in chosen),
            "peak_kw": report["peak_kw_total"], "seconds": seconds}


def run_one(q, args, cs, label="") -> dict:
    t = time.time()
    land = Landscape.from_qubo(q)
    t_land = time.time() - t
    sa = simulated_annealing(q, reads=args.sa_reads, sweeps=args.sa_sweeps, seed=args.seed, landscape=land)
    tabu = tabu_search(q, reads=args.tabu_reads, steps=args.tabu_steps, seed=args.seed)
    exact_x = index_to_bits(land.optimum_indices[0], q.n)
    benchmark = [benchmark_row(q, cs, "exact", exact_x, t_land),
                 benchmark_row(q, cs, "simulated_annealing", sa.best_x, sa.seconds),
                 benchmark_row(q, cs, "tabu", tabu.best_x, tabu.seconds)]
    print(f"{label}{q.n} qubits | optimum E = {land.optimum_energy:.2f} | feasible states "
          f"{int(land.feasible.sum())} of {len(land.energies)} | random guess P(opt) = {land.uniform_p_optimal():.2e}")
    print(f"  brute force: {t_land:.1f}s | simulated annealing ({args.sa_reads} reads): best E = "
          f"{sa.best_energy:.2f}, P(opt) = {sa.p_optimal:.2f}, {sa.seconds:.1f}s")
    print(f"  tabu ({args.tabu_reads} starts): best E = {tabu.best_energy:.2f}, {tabu.seconds:.1f}s")
    out = {"landscape": {"n": q.n, "optimum_energy": land.optimum_energy, "seconds": t_land,
                          "n_feasible": int(land.feasible.sum()), "uniform_p_optimal": land.uniform_p_optimal()},
            "simulated_annealing": {"best_energy": sa.best_energy, "p_optimal": sa.p_optimal,
                                    "p_feasible": sa.p_feasible, "seconds": sa.seconds,
                                    "best_x": sa.best_x.tolist(), **sa.params},
            "tabu": {"best_energy": tabu.best_energy, "seconds": tabu.seconds,
                     "best_x": tabu.best_x.tolist(), **tabu.params},
            "exact": {"best_x": exact_x.tolist()},
            "benchmark": benchmark, "_land": land, "_res": None}
    if not args.classical_only:
        res = run_qaoa(q, land, p_max=args.p_max, alpha=args.alpha, maxiter=args.maxiter, shots=args.shots,
                       seed=args.seed, verbose=True, optimizer=args.optimizer)
        elapsed = 0.0
        for layer in res.layers:
            elapsed += layer.seconds  # INTERP also needs all preceding depths
            x = index_to_bits(layer.sampled["best_raw_index"], q.n)
            benchmark.append(benchmark_row(q, cs, f"qaoa_p{layer.p}_raw", x, elapsed))
        out.update(qaoa=res.to_dict(), _res=res)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--columns", type=Path, required=True)
    p.add_argument("--variant", default="E2", choices=["E1", "E2", "E3", "E4"])
    p.add_argument("--V", type=int, default=None, help="fleet size for E3")
    p.add_argument("--p-max", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.2, help="CVaR fraction (1 = plain expectation)")
    p.add_argument("--maxiter", type=int, default=150, help="COBYLA iterations per depth and start")
    p.add_argument("--optimizer", choices=["cobyla", "spsa"], default="cobyla")
    p.add_argument("--shots", type=int, default=10_000)
    p.add_argument("--sa-reads", type=int, default=100)
    p.add_argument("--sa-sweeps", type=int, default=500)
    p.add_argument("--classical-only", action="store_true", help="benchmark exact search, SA and tabu without QAOA")
    p.add_argument("--tabu-reads", type=int, default=20, help="independent tabu starts")
    p.add_argument("--tabu-steps", type=int, default=500, help="tabu moves per start")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--qiskit", action="store_true", help="also sample the optimized circuit with Qiskit")
    p.add_argument("--noise", type=str, default=None, help="1q,2q depolarizing error for --qiskit (needs qiskit-aer)")
    p.add_argument("--scaling", type=str, default=None, help="comma-separated qubit counts, e.g. 8,10,12,16,20")
    p.add_argument("--compare-peak", action="store_true",
                   help="run the same solver settings with R=0 and --peak-r and save both results")
    p.add_argument("--peak-r", type=float, default=1.0, help="positive peak weight for --compare-peak")
    p.add_argument("--out", type=Path, default=ROOT / "results")
    args = p.parse_args()
    if args.classical_only and (args.qiskit or args.noise):
        p.error("--classical-only cannot be combined with --qiskit or --noise")
    if args.compare_peak and args.scaling:
        p.error("--compare-peak and --scaling are separate experiments")
    if min(args.sa_reads, args.sa_sweeps, args.tabu_reads, args.tabu_steps) < 1:
        p.error("SA and tabu reads/iterations must be positive")
    if not args.classical_only and (args.p_max < 1 or args.shots < 1):
        p.error("--p-max and --shots must be positive")

    cs = load_columns(args.columns)
    out = {"columns": args.columns.name, "variant": args.variant, "settings": {
        k: v for k, v in vars(args).items() if k not in ("columns", "out")}}
    benchmark = []

    if args.compare_peak:
        if args.variant not in ("E3", "E4"):
            p.error("--compare-peak requires variant E3 or E4")
        if args.peak_r <= 0:
            p.error("--peak-r must be positive")
        comparisons = {}
        for label, peak_r in (("R=0", 0.0), ("R>0", args.peak_r)):
            q = build(cs, args.variant, args.V, R=peak_r)
            r = run_one(q, args, cs, label=f"[{label}] ")
            land, res = r.pop("_land"), r.pop("_res")
            comparisons[label] = {
                "R": peak_r,
                **r,
                "peak_kw": q.evaluate(index_to_bits(land.optimum_indices[0], q.n))["peak_kw_total"],
            }
            benchmark.extend(r["benchmark"])
        out["peak_comparison"] = comparisons
        name = f"{args.columns.stem}_{args.variant}_peak_comparison.json"
    elif args.scaling:
        rows = []
        for k in [int(v) for v in args.scaling.split(",")]:
            idx = select_columns(cs, budget=k).indices
            r = run_one(build(cs, args.variant, args.V, idx), args, cs, label=f"[{k} columns] ")
            r.pop("_land")
            res = r.pop("_res")
            benchmark.extend(r["benchmark"])
            row = {"qubits": r["landscape"]["n"], **r["landscape"],
                   "sa_p_optimal": r["simulated_annealing"]["p_optimal"],
                   "simulated_annealing": r["simulated_annealing"], "tabu": r["tabu"],
                   "exact": r["exact"], "benchmark": r["benchmark"]}
            if res is not None:
                row.update(qaoa=[{"p": L.p, "p_optimal": L.p_optimal, "p_feasible": L.p_feasible,
                                   "amplification": L.amplification} for L in res.layers],
                           best_p_optimal=max(L.p_optimal for L in res.layers))
            rows.append(row)
            print()
        if not args.classical_only:
            print("qubits | random P(opt) | QAOA best P(opt) | SA P(opt)")
            for r in rows:
                print(f"{r['qubits']:>6} | {r['uniform_p_optimal']:>13.2e} | {r['best_p_optimal']:>16.4f} | "
                      f"{r['sa_p_optimal']:>9.2f}")
        out["scaling"] = rows
        name = f"{args.columns.stem}_{args.variant}_scaling.json"
    else:
        q = build(cs, args.variant, args.V)
        r = run_one(q, args, cs)
        land, res = r.pop("_land"), r.pop("_res")
        out.update(r)
        benchmark.extend(r["benchmark"])
        name = f"{args.columns.stem}_{args.variant}_qaoa.json"
        if res is not None:
            add_circuit_results(q, land, res, args, out)

    if args.classical_only:
        name = name.replace("_qaoa.json", ".json").replace(".json", "_classical.json")
    out["benchmark"] = benchmark
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / name
    path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["columns", *benchmark[0]])
        writer.writeheader()
        writer.writerows({"columns": args.columns.name, **row} for row in benchmark)
    print("\nmethod | feasible | cost | vehicles | deadhead km | peak kW | seconds")
    for row in benchmark:
        print(f"{row['method']} (n={row['qubits']}, R={row['R']:g}) | {row['feasible']} | {row['cost']:.2f} | "
              f"{row['vehicles']} | {row['deadhead_km']:.2f} | {row['peak_kw']:.2f} | {row['seconds']:.3f}")
    print(f"-> {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} (+ CSV)")


def add_circuit_results(q, land, res, args, out):
    last = res.layers[-1]
    qc = qaoa_circuit(q, last.gammas, last.betas, land)
    out["circuit"] = [circuit_metrics(qc, "all"), circuit_metrics(qc, "heavy-hex")]
    c_all, c_hh = out["circuit"]
    print(f"  circuit p={last.p}: {c_all['cx']} CX, depth {c_all['depth']} (all-to-all) | "
          f"{c_hh['cx']} CX, depth {c_hh['depth']} on heavy-hex ({c_hh['physical_qubits']} physical qubits)")
    if args.qiskit:
        n1, n2 = (float(v) for v in args.noise.split(",")) if args.noise else (0.0, 0.0)
        t = time.time()
        counts = sample_with_qiskit(qc, shots=args.shots, seed=args.seed, noise_1q=n1, noise_2q=n2)
        m = counts_metrics(q, land, counts)
        out["qiskit_sampling"] = {"noise_1q": n1, "noise_2q": n2, "seconds": time.time() - t, **m}
        print(f"  Qiskit sampling ({args.shots} shots{', noise ' + args.noise if args.noise else ''}): "
              f"feasible {m['frac_feasible']:.3f}, optimal {m['frac_optimal']:.3f}, "
              f"found optimum {m['found_optimum']} (after repair {m['optimum_after_repair']}), "
              f"{time.time() - t:.0f}s")
    print("\n  p | P(opt)   | x random | shots for 95 % | P(feasible) | approx ratio | optimum in shots (after repair)")
    for L in res.layers:
        s = L.sampled
        print(f"  {L.p} | {L.p_optimal:.5f}  | {L.amplification:>8.0f} | {L.shots_for_optimum_95:>14.0f} | "
              f"{L.p_feasible:>11.4f} | "
              f"{L.approx_ratio:>12.4f} | {s['found_optimum']} ({s['optimum_after_repair']})")


if __name__ == "__main__":
    main()

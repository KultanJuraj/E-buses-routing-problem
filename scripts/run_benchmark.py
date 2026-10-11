"""Solution-quality benchmark: best of N samples, QAOA vs simulated annealing (repair problems).

Usage (from the project root):
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --events k585@07:00 k4@06:00 S3@07:00-09:00 --budgets 1 10 100
    python scripts/run_benchmark.py --p-max 3 --sa-reads 5000

For every event: XY-mixer QAOA and penalty QAOA (exact statevector, p = 1..p_max),
simulated annealing (--sa-reads independent runs) and random baselines. Each sample is
replayed to its true extra cost (invalid samples count as the cancel-all fallback); the
table gives, for a budget of N samples, the probability that the best sample is the
true optimum and the expected cost of the best sample.
Writes results/quality_benchmark.json and .csv, and the charts of scripts/plot_benchmark.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evsp.benchmark import run_quality_benchmark  # noqa: E402
from evsp.demo import load_demo_data  # noqa: E402
from evsp.disruption import Disruption, build_repair_problem  # noqa: E402

DEFAULT_EVENTS = ["k585@07:00", "k190@05:00", "k4@06:00", "S3@07:00-09:00", "k585@05:00"]
def minutes(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def parse_event(spec: str, sched) -> Disruption:
    who, when = spec.split("@")
    if "-" in when:
        a, b = when.split("-")
        return Disruption.charger_outage(who, minutes(a), minutes(b))
    return Disruption.bus_failure(sched, who, minutes(when))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", nargs="+", default=DEFAULT_EVENTS, help="bus@HH:MM or station@HH:MM-HH:MM")
    p.add_argument("--budgets", nargs="+", type=int, default=[1, 3, 10, 30, 100, 300, 1000])
    p.add_argument("--p-max", type=int, default=2)
    p.add_argument("--sa-reads", type=int, default=2000)
    p.add_argument("--sa-sweeps", type=int, default=500)
    p.add_argument("--reserves", type=int, default=3)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "quality_benchmark.json")
    p.add_argument("--no-plot", action="store_true", help="skip the charts (scripts/plot_benchmark.py draws them later)")
    args = p.parse_args()

    d = load_demo_data()
    results = []
    for spec in args.events:
        ev = parse_event(spec, d.schedule)
        rp = build_repair_problem(d.inst, d.schedule, ev, reserves=args.reserves)
        r = run_quality_benchmark(d.inst, rp, budgets=args.budgets, p_max=args.p_max, sa_reads=args.sa_reads,
                                  sa_sweeps=args.sa_sweeps)
        r["event"] = spec
        results.append(r)
        print(f"\n== {spec}: {r['qubits']} qubits, {r['valid_states']} valid assignments | true optimum "
              f"{r['true_optimum']:.0f}, greedy {r['greedy']:.0f}, cancel-all {r['fallback']:.0f} ({r['seconds']:.1f}s)")
        print("   one sample: " + ", ".join(f"{s['method']} P(valid) {s['p_valid']:.3f} P(opt) {s['p_optimal']:.3f}"
                                             for s in r["per_sample"][:3]))
        methods = list(dict.fromkeys(row["method"] for row in r["best_of_n"]))
        print(f"   {'N':>5} | " + " | ".join(f"{m[:24]:>24}" for m in methods))
        for n in args.budgets:
            cells = []
            for m in methods:
                row = next(x for x in r["best_of_n"] if x["method"] == m and x["n"] == n)
                cells.append(f"P={row['p_found_optimum']:.2f} gap {row['expected_gap']:>5.0f}")
            print(f"   {n:>5} | " + " | ".join(f"{c:>24}" for c in cells))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                    "events": results}, indent=1), encoding="utf-8")
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event", "qubits", "valid_states", "true_optimum", "greedy", "method", "n",
                    "p_found_optimum", "expected_best_cost", "expected_gap", "p_any_valid"])
        for r in results:
            for row in r["best_of_n"]:
                w.writerow([r["event"], r["qubits"], r["valid_states"], round(r["true_optimum"], 2), round(r["greedy"], 2),
                            row["method"], row["n"], round(row["p_found_optimum"], 5), round(row["expected_best_cost"], 2),
                            round(row["expected_gap"], 2), round(row["p_any_valid"], 5)])
    print(f"\n-> {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out} (+ CSV)")
    if not args.no_plot:
        from plot_benchmark import plot_all     # scripts/plot_benchmark.py, next to this file

        plot_all(results, args.out.parent, args.out.stem)


if __name__ == "__main__":
    main()
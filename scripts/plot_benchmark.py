"""Charts for the solution-quality benchmark (results/quality_benchmark.json).

Usage (from the project root):
    python scripts/run_benchmark.py            # computes the results (and calls this script)
    python scripts/plot_benchmark.py           # redraws the charts from the saved JSON
    python scripts/plot_benchmark.py --input results/quality_benchmark.json --out results

Writes three PNG charts:
  quality_benchmark.png          P(best of N samples = true optimum) vs N, one panel per event
  quality_benchmark_gap.png      expected extra cost of the best of N samples above the optimum
  quality_benchmark_samples.png  samples needed to see the true optimum with 95 % confidence
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# same colors as the demo app (evsp/viz.py); every series also has a legend label
COLORS = [("XY", "#eb6834"), ("penalty", "#2a78d6"), ("simulated", "#7d7c76"),
          ("random valid", "#1baf7a"), ("random bitstring", "#b5b4ae")]
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e9e8e4"


def color(method: str) -> str:
    return next((c for key, c in COLORS if key in method), TEXT_2)


def short(method: str) -> str:
    return (method.replace("simulated annealing", "SA").replace("penalty QAOA, X mixer", "penalty QAOA")
            .replace("XY-mixer QAOA", "XY-QAOA").replace("random valid assignment", "random valid"))


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is needed for the charts: pip install matplotlib")
    plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2,
                         "ytick.color": TEXT_2, "axes.titlecolor": TEXT, "font.size": 10})
    return plt


def _methods(event: dict, skip_bitstring: bool = False) -> list[str]:
    out = list(dict.fromkeys(row["method"] for row in event["best_of_n"]))
    return [m for m in out if not (skip_bitstring and m.startswith("random bitstring"))]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor="white")
    print(f"-> {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


def _panels(plt, events: list[dict], ylabel: str, value, log_y: bool = False, skip_bitstring: bool = False):
    fig, axes = plt.subplots(1, len(events), figsize=(3.4 * len(events), 3.7), sharey=True, squeeze=False)
    for ax, ev in zip(axes[0], events):
        for m in _methods(ev, skip_bitstring):
            rows = [r for r in ev["best_of_n"] if r["method"] == m]
            ax.plot([r["n"] for r in rows], [value(r) for r in rows], ":" if m.startswith("random") else "-",
                    marker="o", markersize=4, linewidth=2, color=color(m), label=short(m))
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("symlog", linthresh=10)
        ax.set_title(f"{ev['event']}\n{ev['qubits']} qubits, {ev['valid_states']} valid states", fontsize=10, loc="left")
        ax.set_xlabel("samples N")
        ax.grid(color=GRID)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel(ylabel)
    handles = {}
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig, axes


def plot_probability(events: list[dict], path: Path) -> None:
    plt = _plt()
    fig, axes = _panels(plt, events, "P(best of N = true optimum)", lambda r: r["p_found_optimum"])
    for ax in axes[0]:
        ax.set_ylim(0, 1.05)
    _save(fig, path)
    plt.close(fig)


def plot_gap(events: list[dict], path: Path) -> None:
    plt = _plt()
    fig, _ = _panels(plt, events, "expected extra cost above the optimum", lambda r: max(r["expected_gap"], 0.0),
                     log_y=True, skip_bitstring=True)
    _save(fig, path)
    plt.close(fig)


def samples_for_95(p: float) -> float:
    if p <= 0:
        return math.inf
    if p >= 1:
        return 1.0
    return math.ceil(math.log(0.05) / math.log(1 - p))


def plot_samples(events: list[dict], path: Path) -> None:
    """Grouped bars: samples needed to see the true optimum with 95 % confidence (log scale)."""
    plt = _plt()
    methods = [m for m in dict.fromkeys(s["method"] for ev in events for s in ev["per_sample"])
               if not m.startswith("random bitstring")]
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(1.9 * len(events) + 2.5, 4.2))
    for k, m in enumerate(methods):
        xs, ys = [], []
        for i, ev in enumerate(events):
            s = next((s for s in ev["per_sample"] if s["method"] == m), None)
            if s is None:
                continue
            xs.append(i - 0.4 + width * (k + 0.5))
            ys.append(samples_for_95(s["p_optimal"]))
        bars = ax.bar(xs, ys, width=width * 0.92, color=color(m), label=short(m), edgecolor="white")
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y * 1.08, f"{y:.0f}", ha="center", va="bottom", fontsize=8,
                    color=TEXT_2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(events)), [f"{ev['event']}\n{ev['qubits']} qubits" for ev in events])
    ax.set_ylabel("samples for 95 % confidence (fewer is better)")
    ax.set_title("Samples needed to find the true optimum", loc="left")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=len(methods), loc="upper left")
    ax.set_ylim(1, ax.get_ylim()[1] * 3)
    missing = [short(m) for m in methods if any(all(s["method"] != m for s in ev["per_sample"]) for ev in events)]
    if missing:
        fig.text(0.01, 0.01, f"No bar: {', '.join(missing)} not simulated for that event (full 2^n statevector, n > 18).",
                 fontsize=8, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.04 if missing else 0, 1, 1))
    _save(fig, path)
    plt.close(fig)


def plot_all(events: list[dict], out: Path, stem: str = "quality_benchmark") -> None:
    plot_probability(events, out / f"{stem}.png")
    plot_gap(events, out / f"{stem}_gap.png")
    plot_samples(events, out / f"{stem}_samples.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=ROOT / "results" / "quality_benchmark.json")
    p.add_argument("--out", type=Path, default=ROOT / "results")
    args = p.parse_args()
    if not args.input.exists():
        sys.exit(f"{args.input} not found: run scripts/run_benchmark.py first")
    events = json.loads(args.input.read_text(encoding="utf-8"))["events"]
    plot_all(events, args.out, args.input.stem)


if __name__ == "__main__":
    main()
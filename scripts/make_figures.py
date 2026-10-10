"""Export the pitch figures (package G) as interactive HTML (and PNG if kaleido + Chrome work).

    python scripts/make_figures.py                         # bus k4 fails at 06:00
    python scripts/make_figures.py --bus k585 --at 07:00   # 14 qubits: penalty QAOA computed live
    python scripts/make_figures.py --png --offline

Writes figures/schedule_gantt.html, schedule_soc.html, schedule_power.html,
repair_<event>_before.html, repair_<event>_after.html, repair_<event>_methods.html
and scaling_<file>.html for every results/*_scaling.json. Penalty QAOA above 16
qubits is taken from results/repair_<event>.json (scripts/run_repair.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evsp.demo import DEFAULT_COLUMNS, DEFAULT_INSTANCE, load_demo_data, solve_disruption  # noqa: E402
from evsp.disruption import Disruption  # noqa: E402
from evsp.viz import describe_sk, gantt_figure, method_figure, power_figure, scaling_figure, soc_figure  # noqa: E402


def minutes(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--columns", type=Path, default=DEFAULT_COLUMNS)
    p.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    p.add_argument("--bus", default="k4", help="failed bus")
    p.add_argument("--at", default="06:00", help="failure time HH:MM")
    p.add_argument("--charger", help="station out of service instead of a bus failure (e.g. S3)")
    p.add_argument("--from", dest="start", default="07:00")
    p.add_argument("--to", dest="end", default="09:00")
    p.add_argument("--reserves", type=int, default=3)
    p.add_argument("--p-max", type=int, default=3)
    p.add_argument("--out", type=Path, default=ROOT / "figures")
    p.add_argument("--png", action="store_true", help="also write PNG (needs kaleido and Chrome)")
    p.add_argument("--offline", action="store_true", help="embed plotly.js (~4 MB per file) instead of CDN")
    args = p.parse_args()

    d = load_demo_data(args.columns, args.instance)
    args.out.mkdir(parents=True, exist_ok=True)
    figs = {"schedule_gantt": gantt_figure(d.inst, d.schedule.duties, title="Optimálny rozvrh"),
            "schedule_soc": soc_figure(d.inst, d.schedule.duties),
            "schedule_power": power_figure(d.schedule.duties)}

    if args.charger:
        ev = Disruption.charger_outage(args.charger, minutes(args.start), minutes(args.end))
        tag = f"charger_{args.charger}_{args.start}-{args.end}".replace(":", "")
    else:
        ev = Disruption.bus_failure(d.schedule, args.bus, minutes(args.at))
        tag = f"bus_{args.bus}_{args.at}".replace(":", "")
    rep = solve_disruption(d, ev, reserves=args.reserves, p_max=args.p_max)
    failed = {ev.bus: ev.at_min} if ev.kind == "bus" else None
    outage = (ev.station, ev.at_min, ev.until_min) if ev.kind == "charger" else None
    figs[f"repair_{tag}_before"] = gantt_figure(d.inst, rep.before, title="Pred udalosťou", failed=failed,
                                                outage=outage)
    figs[f"repair_{tag}_after"] = gantt_figure(d.inst, rep.after, title=f"Po oprave: {describe_sk(ev)}",
                                               failed=failed, moved=rep.moved, cancelled=rep.cancelled,
                                               outage=outage)
    figs[f"repair_{tag}_methods"] = method_figure(
        rep.rows, title=f"Oprava rozvrhu, {rep.problem.n} qubitov: ako často metóda nájde optimum")
    print(f"{describe_sk(ev)}: {rep.problem.n} qubits, penalty QAOA {rep.penalty_source}")
    for r in rep.rows:
        po = "-" if r["p_optimal"] is None else f"{r['p_optimal']:.4f}"
        print(f"  {r['method']:<24} P(opt) {po}")

    for f in sorted((ROOT / "results").glob("*_scaling.json")):
        rows = json.loads(f.read_text(encoding="utf-8"))["scaling"]
        figs[f"scaling_{f.stem.removesuffix('_scaling')}"] = scaling_figure(rows)

    for name, fig in figs.items():
        path = args.out / f"{name}.html"
        fig.write_html(path, include_plotlyjs=True if args.offline else "cdn", full_html=True)
        print(f"-> {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
        if args.png:
            try:
                fig.write_image(args.out / f"{name}.png", scale=2, width=1200)
            except Exception as e:                       # kaleido needs a local Chrome
                print(f"   PNG skipped: {type(e).__name__}: {str(e).splitlines()[0]}")
                args.png = False


if __name__ == "__main__":
    main()

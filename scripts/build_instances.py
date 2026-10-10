"""Build tiny / small / medium / large instances from the King County Metro GTFS.

Usage (from the project root):
    git clone --depth 1 https://github.com/dan-mccabe/beb-chargers-pactrans.git external/beb-chargers-pactrans
    python scripts/build_instances.py
    python scripts/build_instances.py --preset tiny small --depot 47.49,-122.27
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evsp.data.gtfs_kcm import (CASE_STUDY_ROUTES, Assumptions, build_instance, load_sites,  # noqa: E402
                                load_trip_table, save_instance, select_trips)
from evsp.instance import load_instance  # noqa: E402

DEFAULT_REPO = ROOT / "external" / "beb-chargers-pactrans" / "beb_chargers" / "data"

# name -> selection arguments for select_trips (times in minutes after midnight)
PRESETS = {
    "tiny":   dict(routes=["150"], max_blocks=2, max_trips=8),     # brute force / QAOA
    "small":  dict(routes=["150"], max_blocks=5, max_trips=20),    # QAOA in simulation
    "medium": dict(routes=["150"]),                                                     # one route, full day
    "large":  dict(routes=CASE_STUDY_ROUTES, whole_blocks=True),                        # McCabe study area
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_REPO,
                   help="folder with gtfs/ and so_king_cty_sites.csv")
    p.add_argument("--out", type=Path, default=ROOT / "instances")
    p.add_argument("--preset", nargs="+", default=list(PRESETS), choices=list(PRESETS))
    p.add_argument("--depot", type=str, default=None, help="lat,lon of the depot (default: synthetic centroid)")
    p.add_argument("--max-site-km", type=float, default=15.0,
                   help="drop charging sites farther than this from every terminal of the instance")
    args = p.parse_args()

    gtfs_dir = args.data_dir / "gtfs"
    if not gtfs_dir.exists():
        sys.exit(f"GTFS not found in {gtfs_dir}. Clone the repo first (see the docstring).")
    depot = tuple(float(x) for x in args.depot.split(",")) if args.depot else None

    print("Loading GTFS (stop_times has ~1.2M rows, takes a few seconds)...")
    trips = load_trip_table(gtfs_dir)
    sites = load_sites(args.data_dir / "so_king_cty_sites.csv")
    assumptions = Assumptions()

    for name in args.preset:
        sel = select_trips(trips, **PRESETS[name])
        inst = build_instance(sel, sites, assumptions, name=f"kcm_{name}", depot=depot,
                              max_site_km=args.max_site_km)
        path = save_instance(inst, args.out / f"kcm_{name}.json")
        loaded = load_instance(path)
        check = loaded.check_baseline()
        print(f"  {loaded.summary()}")
        print(f"    arcs |A| = {len(loaded.arcs(max_gap=180))} (max_gap 180 min); "
              f"real-block pairs feasible: {check['pairs_ok']}/{check['pairs_ok'] + check['pairs_fail']}"
              f"  -> {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


if __name__ == "__main__":
    main()

"""Parser for the King County Metro GTFS shipped in dan-mccabe/beb-chargers-pactrans.

Data facts this module relies on (checked on the Feb 2020 feed in that repo):
  * shape_dist_traveled is in FEET (about 3281 per km).
  * GTFS times can exceed 24:00 (up to 28:51); they are kept as minutes after
    the service day's midnight, so 25:10 -> 1510.
  * stops.txt is NOT in the repo, so terminal coordinates are taken from the
    first and last point of each trip's shape.
  * block_id in trips.txt is the operator's real vehicle schedule, used as a
    baseline. Weekday service is service_id 86021.

Everything that is not in the data (battery, chargers' power, depot, tariff,
costs) lives in `Assumptions` and is written into every instance file, so it
can be shown openly and replaced with real values later.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from evsp.validation import SCHEMA_VERSION, validate_instance

FT_PER_KM = 3280.84
KM_PER_MILE = 1.609344
WEEKDAY_SERVICE_ID = "86021"

# The 18 routes used in McCabe's South King County case study.
CASE_STUDY_ROUTES = ["101", "102", "111", "116", "143", "150", "157", "158", "159",
                     "177", "178", "179", "180", "190", "192", "193", "197", "952"]


@dataclass
class Assumptions:
    """Values not contained in the GTFS. Source noted where one exists."""

    # Battery and consumption: McCabe case study (466 kWh * 0.9, 3 kWh/mile).
    battery_kwh: float = 466 * 0.9
    kwh_per_km: float = 3.0 / KM_PER_MILE
    soc_min: float = 0.10          # fraction of usable capacity (study charges below 10 %)
    soc_max: float = 1.00
    soc_start: float = 1.00        # buses leave the depot full
    # Chargers: 300 kW at each candidate site (study); depot chargers are assumed.
    site_charger_kw: float = 300.0
    chargers_per_site: int = 1
    depot_charger_kw: float = 150.0
    depot_chargers: int = 10
    charge_efficiency: float = 0.90
    min_charge_min: float = 5.0
    # Deadhead estimate: straight line * detour factor at a fixed speed.
    detour_factor: float = 1.3
    deadhead_speed_kmh: float = 30.0
    min_layover_min: float = 5.0
    # Time slots and a simple two-rate tariff (assumed, currency-neutral units).
    slot_min: int = 15
    tariff_day: float = 0.20
    tariff_night: float = 0.10
    night_start_h: float = 22.0
    night_end_h: float = 6.0
    # Cost weights used by the objective (assumed; tune later).
    vehicle_cost: float = 500.0
    deadhead_cost_per_km: float = 1.5
    peak_scope: str = "global"   # document's sum_t (sum_k p_kt y_k)^2

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- utils

def hms_to_min(value: str) -> float:
    """'25:10:30' -> 1510.5 (minutes after service-day midnight)."""
    h, m, s = (int(x) for x in value.strip().split(":"))
    if h < 0 or not 0 <= m < 60 or not 0 <= s < 60:
        raise ValueError(f"Invalid GTFS time: {value!r}")
    return h * 60 + m + s / 60.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; works on scalars and numpy arrays."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- loading

def load_trip_table(gtfs_dir: str | Path, service_id: str = WEEKDAY_SERVICE_ID) -> pd.DataFrame:
    """One row per trip of `service_id` with times, terminals, distance and block.

    Columns: trip_id, route, block_id, direction_id, start_stop, end_stop,
    dep_min, arr_min, dist_km, start_lat, start_lon, end_lat, end_lon.
    """
    g = Path(gtfs_dir)
    routes = pd.read_csv(g / "routes.txt", dtype=str, usecols=["route_id", "route_short_name"])
    trips = pd.read_csv(g / "trips.txt", dtype=str,
                        usecols=["route_id", "service_id", "trip_id", "direction_id", "block_id", "shape_id"])
    trips = trips[trips["service_id"] == service_id]
    if trips.empty:
        raise ValueError(f"No trips for service_id={service_id}")
    trips = trips.merge(routes, on="route_id", how="left").rename(columns={"route_short_name": "route"})

    st = pd.read_csv(g / "stop_times.txt", dtype=str,
                     usecols=["trip_id", "arrival_time", "departure_time", "stop_id",
                              "stop_sequence", "shape_dist_traveled"])
    st = st[st["trip_id"].isin(set(trips["trip_id"]))].copy()
    st["stop_sequence"] = st["stop_sequence"].astype(int)
    st["shape_dist_traveled"] = pd.to_numeric(st["shape_dist_traveled"], errors="coerce")
    st = st.sort_values(["trip_id", "stop_sequence"])
    # Keep actual endpoint rows: groupby.first/last skip nulls per column and
    # can silently substitute an intermediate stop's cumulative distance.
    first = st.drop_duplicates("trip_id", keep="first").set_index("trip_id")
    last = st.drop_duplicates("trip_id", keep="last").set_index("trip_id")

    ends = pd.DataFrame({
        "start_stop": first["stop_id"],
        "dep_min": first["departure_time"].map(hms_to_min),
        "end_stop": last["stop_id"],
        "arr_min": last["arrival_time"].map(hms_to_min),
        "dist_ft": last["shape_dist_traveled"] - first["shape_dist_traveled"],
    })

    shapes = pd.read_csv(g / "shapes.txt",
                         dtype={"shape_id": str},
                         usecols=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"])
    shapes = shapes[shapes["shape_id"].isin(set(trips["shape_id"]))].sort_values(["shape_id", "shape_pt_sequence"])
    s_first = shapes.drop_duplicates("shape_id", keep="first").set_index("shape_id")
    s_last = shapes.drop_duplicates("shape_id", keep="last").set_index("shape_id")
    shape_ends = pd.DataFrame({
        "start_lat": s_first["shape_pt_lat"], "start_lon": s_first["shape_pt_lon"],
        "end_lat": s_last["shape_pt_lat"], "end_lon": s_last["shape_pt_lon"],
    })

    df = trips.join(ends, on="trip_id").join(shape_ends, on="shape_id")
    df["dist_km"] = df["dist_ft"] / FT_PER_KM
    # Fallback when a trip lacks shape_dist_traveled: straight line * 1.3.
    missing = df["dist_km"].isna() | (df["dist_km"] <= 0)
    df.loc[missing, "dist_km"] = 1.3 * haversine_km(df.loc[missing, "start_lat"], df.loc[missing, "start_lon"],
                                                    df.loc[missing, "end_lat"], df.loc[missing, "end_lon"])
    required = ["dep_min", "arr_min", "start_lat", "start_lon", "end_lat", "end_lon", "dist_km"]
    bad = ~np.isfinite(df[required]).all(axis=1) | (df["arr_min"] <= df["dep_min"]) | (df["dist_km"] <= 0)
    if bad.any():
        raise ValueError(f"Invalid GTFS endpoints, times or distance for trips: {df.loc[bad, 'trip_id'].tolist()}")
    cols = ["trip_id", "route", "block_id", "direction_id", "start_stop", "end_stop", "dep_min", "arr_min",
            "dist_km", "start_lat", "start_lon", "end_lat", "end_lon"]
    return df[cols].sort_values(["dep_min", "trip_id"]).reset_index(drop=True)


def load_sites(sites_csv: str | Path) -> list[dict]:
    """Candidate charging sites (Name, y=lat, x=lon)."""
    with open(sites_csv, newline="", encoding="utf-8") as f:
        return [{"name": r["Name"].strip(), "lat": float(r["y"]), "lon": float(r["x"])} for r in csv.DictReader(f)]


# --------------------------------------------------------------------------- selection

def select_trips(trips: pd.DataFrame,
                 routes: Optional[Iterable[str]] = None,
                 start_min: Optional[float] = None,
                 end_min: Optional[float] = None,
                 max_trips: Optional[int] = None,
                 whole_blocks: bool = False,
                 max_blocks: Optional[int] = None) -> pd.DataFrame:
    """Cut a sub-instance out of the day.

    whole_blocks=True keeps every trip of every block that touches the chosen
    routes (blocks interline across routes), so the real block schedule stays
    a complete, fair baseline.
    max_blocks keeps only the first N real blocks (by first departure), which
    gives small instances that still contain chains of trips one bus can do.
    Time window and max_trips are applied last.
    """
    df = trips
    if routes is not None:
        routes = {str(r) for r in routes}
        if whole_blocks:
            blocks = set(df.loc[df["route"].isin(routes), "block_id"])
            df = df[df["block_id"].isin(blocks)]
        else:
            df = df[df["route"].isin(routes)]
    if max_blocks is not None:
        order = df.groupby("block_id")["dep_min"].min().sort_values()
        df = df[df["block_id"].isin(set(order.index[:max_blocks]))]
    if start_min is not None:
        df = df[df["dep_min"] >= start_min]
    if end_min is not None:
        df = df[df["arr_min"] <= end_min]
    df = df.sort_values(["dep_min", "trip_id"])
    if max_trips is not None:
        df = df.head(max_trips)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- instance

def _tariff_slots(a: Assumptions, day_end_min: float) -> list[dict]:
    slots = []
    n = int(np.ceil(day_end_min / a.slot_min))
    for t in range(n):
        start = t * a.slot_min
        hour = (start / 60.0) % 24
        night = hour >= a.night_start_h or hour < a.night_end_h
        slots.append({"t": t, "start_min": start, "price": a.tariff_night if night else a.tariff_day})
    return slots


def build_instance(trips: pd.DataFrame,
                   sites: list[dict],
                   assumptions: Optional[Assumptions] = None,
                   name: str = "instance",
                   depot: Optional[tuple[float, float]] = None,
                   max_site_km: Optional[float] = None,
                   service_id: str = WEEKDAY_SERVICE_ID) -> dict:
    """Turn selected trips into the JSON instance used by the rest of the pipeline."""
    a = assumptions or Assumptions()
    if trips.empty:
        raise ValueError("No trips selected")

    # Terminals: one location per stop_id, coordinates averaged over shape ends.
    pts = pd.concat([
        trips[["start_stop", "start_lat", "start_lon"]].set_axis(["stop", "lat", "lon"], axis=1),
        trips[["end_stop", "end_lat", "end_lon"]].set_axis(["stop", "lat", "lon"], axis=1),
    ]).groupby("stop")[["lat", "lon"]].mean()
    locations = [{"id": f"T{stop}", "type": "terminal", "stop_id": stop, "lat": float(r.lat), "lon": float(r.lon)}
                 for stop, r in pts.iterrows()]

    if depot is None:
        depot_lat, depot_lon, synthetic = float(pts["lat"].mean()), float(pts["lon"].mean()), True
    else:
        depot_lat, depot_lon, synthetic = float(depot[0]), float(depot[1]), False
    locations.append({"id": "DEPOT", "type": "depot", "lat": depot_lat, "lon": depot_lon, "synthetic": synthetic})

    chargers = [{"id": f"DEPOT-C{n + 1}", "location": "DEPOT", "station": "DEPOT", "power_kw": a.depot_charger_kw,
                 "availability": "assumed"}
                for n in range(a.depot_chargers)]
    for i, s in enumerate(sites, start=1):
        if max_site_km is not None:
            nearest = float(np.min(haversine_km(s["lat"], s["lon"], pts["lat"].values, pts["lon"].values)))
            if nearest > max_site_km:
                continue
        loc_id = f"S{i}"
        locations.append({"id": loc_id, "type": "charger_site", "name": s["name"], "lat": s["lat"], "lon": s["lon"]})
        chargers += [{"id": f"{loc_id}-C{n + 1}", "location": loc_id, "station": loc_id, "power_kw": a.site_charger_kw,
                      "availability": "assumed"}
                     for n in range(a.chargers_per_site)]

    # Deadhead matrix between every pair of locations (straight line * detour).
    lat = np.array([l["lat"] for l in locations])
    lon = np.array([l["lon"] for l in locations])
    km = a.detour_factor * haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    minutes = km / a.deadhead_speed_kmh * 60.0
    ids = [l["id"] for l in locations]
    deadhead = {ids[i]: {ids[j]: {"km": round(float(km[i, j]), 3), "min": round(float(minutes[i, j]), 2)}
                         for j in range(len(ids))} for i in range(len(ids))}

    trip_rows = [{
        "id": r.trip_id, "route": r.route, "block_id": r.block_id, "direction_id": r.direction_id,
        "start": f"T{r.start_stop}", "end": f"T{r.end_stop}",
        "dep_min": round(float(r.dep_min), 2), "arr_min": round(float(r.arr_min), 2),
        "dist_km": round(float(r.dist_km), 3), "energy_kwh": round(float(r.dist_km) * a.kwh_per_km, 3),
    } for r in trips.itertuples(index=False)]

    baseline = [{"block_id": b, "trips": list(g.sort_values("dep_min")["trip_id"])}
                for b, g in trips.groupby("block_id")]

    instance = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "meta": {
            "source": "King County Metro GTFS (Feb 2020) via github.com/dan-mccabe/beb-chargers-pactrans",
            "service_id": service_id,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "units": {"time": "minutes after service-day midnight", "distance": "km", "energy": "kWh",
                      "power": "kW"},
            "n_trips": len(trip_rows), "n_chargers": len(chargers),
            "notes": [
                "Terminal coordinates come from shape endpoints (stops.txt not in repo).",
                "Deadheads are straight-line distance * detour_factor at deadhead_speed_kmh.",
                "baseline_blocks lists the operator's real blocks restricted to the selected trips; "
                "a block is complete only if the instance was built with whole_blocks=True.",
            ] + (["Depot location is synthetic (centroid of terminals)."] if synthetic else []),
        },
        "assumptions": a.to_dict(),
        "locations": locations,
        "chargers": chargers,
        "trips": trip_rows,
        "deadhead": deadhead,
        "tariff": _tariff_slots(a, max(t["arr_min"] for t in trip_rows) + 120),
        "baseline_blocks": baseline,
    }
    validate_instance(instance)
    return instance


def save_instance(instance: dict, path: str | Path) -> Path:
    validate_instance(instance)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(instance, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path

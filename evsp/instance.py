"""Instance model shared by the column generator, QUBO builder and solvers.

    inst = load_instance("instances/kcm_small.json")
    arcs = inst.arcs()                 # admissible (i, j) trip connections = set A
    inst.check_baseline()              # real blocks vs. our deadhead estimate
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .validation import SCHEMA_VERSION, validate_instance


@dataclass(frozen=True)
class Trip:
    id: str
    route: str
    block_id: str
    start: str          # location id
    end: str            # location id
    dep_min: float
    arr_min: float
    dist_km: float
    energy_kwh: float


@dataclass(frozen=True)
class Location:
    id: str
    type: str           # terminal | depot | charger_site
    lat: float
    lon: float
    name: str = ""


@dataclass(frozen=True)
class Charger:
    id: str
    location: str
    station: str        # grid connection identifier; v1.0 peak objective is global
    power_kw: float
    availability: str   # assumed infrastructure or confirmed installation


@dataclass
class Instance:
    name: str
    trips: list[Trip]
    locations: dict[str, Location]
    chargers: list[Charger]
    deadhead_km: dict[str, dict[str, float]]
    deadhead_min: dict[str, dict[str, float]]
    tariff: list[dict]
    assumptions: dict
    baseline_blocks: list[dict]
    meta: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    # ------------------------------------------------------------------ helpers
    @property
    def trip_by_id(self) -> dict[str, Trip]:
        return {t.id: t for t in self.trips}

    @property
    def depot(self) -> str:
        return next(l.id for l in self.locations.values() if l.type == "depot")

    def dh_km(self, a: str, b: str) -> float:
        return self.deadhead_km[a][b]

    def dh_min(self, a: str, b: str) -> float:
        return self.deadhead_min[a][b]

    def dh_energy_kwh(self, a: str, b: str) -> float:
        """Energy for every empty leg, including depot and charger detours."""
        return self.dh_km(a, b) * self.assumptions["kwh_per_km"]

    def can_follow(self, i: Trip, j: Trip, min_layover: Optional[float] = None,
                   max_gap: Optional[float] = None) -> bool:
        """True if one bus can do trip j right after trip i (no charging in between)."""
        layover = self.assumptions["min_layover_min"] if min_layover is None else min_layover
        ready = i.arr_min + self.dh_min(i.end, j.start) + layover
        if ready > j.dep_min:
            return False
        return max_gap is None or j.dep_min - i.arr_min <= max_gap

    def arcs(self, min_layover: Optional[float] = None, max_gap: Optional[float] = None) -> list[tuple[str, str]]:
        """Admissible connections A (graph is acyclic: time strictly increases).

        max_gap (minutes) prunes very long idle connections to keep the graph small.
        """
        trips = sorted(self.trips, key=lambda t: t.dep_min)
        out = []
        for a_idx, i in enumerate(trips):
            for j in trips[a_idx + 1:]:
                if self.can_follow(i, j, min_layover, max_gap):
                    out.append((i.id, j.id))
        return out

    def check_baseline(self, min_layover: Optional[float] = None) -> dict:
        """How many consecutive pairs in the real blocks pass our time check.

        Failures usually mean the deadhead estimate is too slow for that pair,
        a useful sanity check for detour_factor and deadhead_speed_kmh.
        """
        tb = self.trip_by_id
        layover = self.assumptions["min_layover_min"] if min_layover is None else min_layover
        ok = bad = 0
        failures = []
        for blk in self.baseline_blocks:
            seq = [tb[t] for t in blk["trips"]]
            for i, j in zip(seq, seq[1:]):
                if self.can_follow(i, j, min_layover=min_layover):
                    ok += 1
                else:
                    bad += 1
                    available = j.dep_min - i.arr_min
                    travel = self.dh_min(i.end, j.start)
                    failures.append({
                        "block_id": blk["block_id"], "from_trip": i.id, "to_trip": j.id,
                        "from_location": i.end, "to_location": j.start,
                        "available_min": available, "deadhead_km": self.dh_km(i.end, j.start),
                        "deadhead_min": travel, "min_layover_min": layover,
                        "required_min": travel + layover,
                        "shortfall_min": travel + layover - available,
                    })
        return {"instance": self.name, "min_layover_min": layover,
                "pairs_ok": ok, "pairs_fail": bad, "failures": failures}

    def summary(self) -> str:
        n_term = sum(1 for l in self.locations.values() if l.type == "terminal")
        km = sum(t.dist_km for t in self.trips)
        kwh = sum(t.energy_kwh for t in self.trips)
        span = (min(t.dep_min for t in self.trips), max(t.arr_min for t in self.trips))
        return (f"{self.name}: {len(self.trips)} trips, {len(self.baseline_blocks)} real blocks, "
                f"{n_term} terminals, {len(self.chargers)} chargers, {km:.0f} km, {kwh:.0f} kWh, "
                f"{_hhmm(span[0])}-{_hhmm(span[1])}")


def _hhmm(minutes: float) -> str:
    m = int(round(minutes))
    return f"{m // 60:02d}:{m % 60:02d}"


def load_instance(path: str | Path) -> Instance:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    try:
        validate_instance(raw)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    trips = [Trip(id=t["id"], route=t["route"], block_id=t["block_id"], start=t["start"], end=t["end"],
                  dep_min=t["dep_min"], arr_min=t["arr_min"], dist_km=t["dist_km"], energy_kwh=t["energy_kwh"])
             for t in raw["trips"]]
    locations = {l["id"]: Location(id=l["id"], type=l["type"], lat=l["lat"], lon=l["lon"], name=l.get("name", ""))
                 for l in raw["locations"]}
    chargers = [Charger(**c) for c in raw["chargers"]]
    dh = raw["deadhead"]
    return Instance(
        name=raw["name"], trips=trips, locations=locations, chargers=chargers,
        deadhead_km={a: {b: v["km"] for b, v in row.items()} for a, row in dh.items()},
        deadhead_min={a: {b: v["min"] for b, v in row.items()} for a, row in dh.items()},
        tariff=raw["tariff"], assumptions=raw["assumptions"], baseline_blocks=raw["baseline_blocks"],
        meta=raw.get("meta", {}),
    )

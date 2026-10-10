"""Column (duty) generator for the E-VSP set-partitioning QUBO.

A column k is one bus for one day:

    depot -> trip -> [deadhead | charge at a station] -> trip -> ... -> depot

It is feasible when every trip is reached in time and the battery never drops
below soc_min. For every column we precompute everything the QUBO needs:

    c_k     cost (vehicle + deadhead km + electricity)       -> linear terms
    a_ik    which trips it covers (Column.trips)             -> coverage penalty
    D_k     deadhead km                                       -> f_dh
    e_kt    grid energy per tariff slot [kWh]                 -> f_en
    p_kst   average charging power per station and slot [kW] -> f_peak
    C       pairs on the same 1-charger station at overlapping times -> f_conf
    r_kl    pairs charging back-to-back on the same station   -> f_rob,ch

Charger capacity is exact in the LP / MILP (clique constraints at every
charging start time). The pairwise C and r_kl for the QUBO are computed only on
the selected columns, because on the full set they grow quadratically.

Usage:
    inst = load_instance("instances/kcm_tiny.json")
    cs = generate_columns(inst)
    sel = select_columns(cs, budget=20)        # qubit budget, always contains a full schedule
    sol = solve_milp(cs, sel.indices)          # exact check on the selection
    conflicts, tight = cs.pairs(sel.indices)   # C and r_kl for the QUBO
"""

from __future__ import annotations

import heapq
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .instance import Instance, Trip

EPS = 1e-9


# --------------------------------------------------------------------------- config

@dataclass
class GenConfig:
    """Limits that keep enumeration tractable. Defaults enumerate tiny/small fully."""

    max_trips: int = 8                 # trips per duty
    max_successors: int = 4            # nearest next trips considered after each trip
    max_gap_min: float = 180.0         # longest idle gap for a direct connection
    max_charge_gap_min: float = 300.0  # longest gap that may contain a charging stop
    charge_below: float = 0.6          # offer a charging stop only if SoC fraction < this
    max_charge_stations: int = 2       # nearest stations tried per gap
    allow_depot_charging: bool = True
    max_columns_per_start: int = 2_000 # spread the column budget over all first trips
    max_nodes_per_start: int = 50_000  # search effort per first trip (matters when pricing)
    max_columns: int = 100_000

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- data

@dataclass
class ChargeStop:
    station: str
    location: str
    start_min: float
    end_min: float
    power_kw: float                 # grid power while charging
    grid_kwh: float                 # energy drawn from the grid
    soc_before: float               # fraction of battery
    soc_after: float


@dataclass
class Column:
    id: str
    trips: list[str]
    cost: float
    vehicle_cost: float
    deadhead_km: float
    deadhead_cost: float
    energy_cost: float              # daytime charging at tariff + overnight refill at night rate
    grid_kwh_day: float
    refill_kwh: float
    min_soc: float                  # fraction of battery
    final_soc: float
    start_min: float                # pull-out departure from the depot
    end_min: float                  # pull-in arrival at the depot
    charges: list[ChargeStop] = field(default_factory=list)
    energy_by_slot: dict[int, float] = field(default_factory=dict)               # e_kt [kWh]
    power_by_station: dict[str, dict[int, float]] = field(default_factory=dict)  # p_kst [kW]

    @property
    def n_trips(self) -> int:
        return len(self.trips)


@dataclass
class ColumnSet:
    instance_name: str
    trip_ids: list[str]
    columns: list[Column]
    station_capacity: dict[str, int]
    slot_min: int
    buffer_min: float
    config: dict
    stats: dict = field(default_factory=dict)
    assumptions: dict = field(default_factory=dict)   # instance assumptions (costs, battery, ...)
    extra: dict = field(default_factory=dict)         # other top-level keys of a loaded file

    # ---------------------------------------------------------------- matrices
    def coverage_matrix(self, indices: Optional[list[int]] = None) -> np.ndarray:
        """a_ik as a dense |I| x |K| 0/1 array (columns restricted to `indices`)."""
        idx = range(len(self.columns)) if indices is None else indices
        row = {t: r for r, t in enumerate(self.trip_ids)}
        a = np.zeros((len(self.trip_ids), len(idx)), dtype=np.int8)
        for c, k in enumerate(idx):
            for t in self.columns[k].trips:
                a[row[t], c] = 1
        return a

    def costs(self, indices: Optional[list[int]] = None) -> np.ndarray:
        idx = range(len(self.columns)) if indices is None else indices
        return np.array([self.columns[k].cost for k in idx], dtype=float)

    def power_tensor(self, indices: Optional[list[int]] = None) -> tuple[np.ndarray, list[str], int]:
        """p_kst as an array [K, S, T] plus the station order and number of slots."""
        idx = list(range(len(self.columns))) if indices is None else list(indices)
        stations = sorted(self.station_capacity)
        n_slots = 1 + max((t for k in idx for prof in self.columns[k].power_by_station.values() for t in prof),
                          default=0)
        p = np.zeros((len(idx), len(stations), n_slots))
        s_pos = {s: i for i, s in enumerate(stations)}
        for c, k in enumerate(idx):
            for s, prof in self.columns[k].power_by_station.items():
                for t, kw in prof.items():
                    p[c, s_pos[s], t] = kw
        return p, stations, n_slots

    def pairs(self, indices: list[int]) -> tuple[list[tuple[int, int]], list[tuple[int, int, float]]]:
        """C and r_kl among `indices`, returned as positions 0..len(indices)-1.

        C: both columns charge at the same 1-charger station at overlapping times.
        r_kl = buffer - gap when l starts charging less than `buffer_min` after k ends
        on the same 1-charger station. Multi-charger stations are exportable only
        if even the whole candidate set stays within capacity. Otherwise this
        pairwise formulation cannot enforce capacity and must fail explicitly.
        """
        per_station: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
        for pos, k in enumerate(indices):
            for ch in self.columns[k].charges:
                per_station[ch.station].append((ch.start_min, ch.end_min, pos))
        conflicts: set[tuple[int, int]] = set()
        tight: dict[tuple[int, int], float] = {}
        for station, iv in per_station.items():
            capacity = self.station_capacity[station]
            if capacity > 1:
                events = sorted((time, delta) for start, end, _ in iv
                                for time, delta in ((start, 1), (end, -1)))
                level = 0
                for _, delta in events:
                    level += delta
                    if level > capacity:
                        raise ValueError(
                            f"Station {station}: candidate charging overlaps exceed capacity {capacity}; "
                            "pairwise QUBO conflicts cannot encode this. Reduce the selection or "
                            "disable depot charging; LP/MILP capacity constraints remain supported.")
                continue
            iv.sort()
            for x, (_, e1, k) in enumerate(iv):
                for s2, _, l in iv[x + 1:]:
                    if s2 > e1 + self.buffer_min:
                        break
                    if k == l:
                        continue
                    if s2 < e1 - EPS:
                        conflicts.add((min(k, l), max(k, l)))
                    else:
                        r = self.buffer_min - (s2 - e1)
                        tight[(k, l)] = max(tight.get((k, l), 0.0), r)
        return sorted(conflicts), sorted((k, l, r) for (k, l), r in tight.items())

    # ---------------------------------------------------------------- io
    def to_json(self, path: str | Path, indices: Optional[list[int]] = None, extra: Optional[dict] = None) -> Path:
        """Write columns (all, or only `indices`) with their C and r_kl.

        Pairs are always computed on what is written, so pass `indices` for big sets.
        """
        idx = list(range(len(self.columns))) if indices is None else list(indices)
        conf, tight = self.pairs(idx)
        out = {
            "instance": self.instance_name,
            "trip_ids": self.trip_ids,
            "slot_min": self.slot_min,
            "buffer_min": self.buffer_min,
            "station_capacity": self.station_capacity,
            "config": self.config,
            "stats": self.stats,
            "assumptions": self.assumptions,
            "columns": [_column_to_dict(self.columns[k]) for k in idx],
            "conflicts": [list(p) for p in conf],
            "tight_pairs": [[a, b, round(r, 3)] for a, b, r in tight],
        }
        if extra:
            out.update(extra)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1, allow_nan=False), encoding="utf-8")
        return path


def _column_to_dict(c: Column) -> dict:
    d = asdict(c)
    d["energy_by_slot"] = {str(t): round(v, 4) for t, v in c.energy_by_slot.items()}
    d["power_by_station"] = {s: {str(t): round(v, 3) for t, v in prof.items()}
                             for s, prof in c.power_by_station.items()}
    for key in ("cost", "deadhead_km", "deadhead_cost", "energy_cost", "grid_kwh_day", "refill_kwh",
                "min_soc", "final_soc", "start_min", "end_min"):
        d[key] = round(d[key], 4)
    return d


def load_columns(path: str | Path) -> ColumnSet:
    """Read a file written by ColumnSet.to_json (conflicts are recomputed by .pairs())."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cols = []
    for d in raw["columns"]:
        d = dict(d)
        charges = [ChargeStop(**c) for c in d.pop("charges")]
        e = {int(t): v for t, v in d.pop("energy_by_slot").items()}
        p = {s: {int(t): v for t, v in prof.items()} for s, prof in d.pop("power_by_station").items()}
        cols.append(Column(**d, charges=charges, energy_by_slot=e, power_by_station=p))
    known = {"instance", "trip_ids", "slot_min", "buffer_min", "station_capacity", "config", "stats",
             "assumptions", "columns", "conflicts", "tight_pairs"}
    return ColumnSet(instance_name=raw["instance"], trip_ids=raw["trip_ids"], columns=cols,
                     station_capacity=raw["station_capacity"], slot_min=raw["slot_min"],
                     buffer_min=raw["buffer_min"], config=raw["config"], stats=raw.get("stats", {}),
                     assumptions=raw.get("assumptions", {}),
                     extra={k: v for k, v in raw.items() if k not in known})


# --------------------------------------------------------------------------- generator

@dataclass
class _Station:
    id: str
    location: str
    power_kw: float
    capacity: int


def _stations(inst: Instance) -> list[_Station]:
    by_station: dict[str, list] = defaultdict(list)
    for c in inst.chargers:
        by_station[c.station].append(c)
    for station, chargers in by_station.items():
        if len({(c.location, c.power_kw) for c in chargers}) != 1:
            raise ValueError(f"Station {station}: aggregated charging requires identical locations and powers")
    return [_Station(id=s, location=chs[0].location, power_kw=max(c.power_kw for c in chs), capacity=len(chs))
            for s, chs in sorted(by_station.items())]


def signature(trips: list[str], charges: list[ChargeStop]) -> tuple:
    """Identity of a column: its trips and where/when it charges."""
    return tuple(trips), tuple((c.station, round(c.start_min, 2), round(c.end_min, 2)) for c in charges)


def _slot_overlaps(start: float, end: float, slot: int):
    """Yield (slot index, minutes of [start, end) inside that slot)."""
    t = int(math.floor(start / slot))
    while t * slot < end - EPS:
        lo, hi = max(start, t * slot), min(end, (t + 1) * slot)
        if hi > lo + EPS:
            yield t, hi - lo
        t += 1


def generate_columns(inst: Instance, cfg: Optional[GenConfig] = None,
                     duals: Optional[dict[str, float]] = None, rc_tol: float = 1e-6,
                     skip: Optional[set] = None) -> ColumnSet:
    """Enumerate battery-feasible duties by depth-first search over the trip graph.

    With `duals` (LP duals per trip) this becomes the pricing step of column
    generation: successors are explored in order of decreasing dual value and
    only columns with reduced cost  c_k - sum_i pi_i  < -rc_tol are kept.
    `skip` holds signatures of columns already in the pool.
    """
    cfg = cfg or GenConfig()
    skip = skip or set()
    a = inst.assumptions
    cap = a["battery_kwh"]
    soc_min_kwh = a["soc_min"] * cap
    soc_max_kwh = a["soc_max"] * cap
    soc_start_kwh = a["soc_start"] * cap
    kwh_km = a["kwh_per_km"]
    eta = a["charge_efficiency"]
    layover = a["min_layover_min"]
    min_charge = a["min_charge_min"]
    slot = int(a["slot_min"])
    price = {s["t"]: s["price"] for s in inst.tariff}
    night_price = a["tariff_night"]
    depot = inst.depot
    all_stations = _stations(inst)
    stations = [s for s in all_stations if cfg.allow_depot_charging or s.id != "DEPOT"]

    trips = sorted(inst.trips, key=lambda t: (t.dep_min, t.id))
    by_id = {t.id: t for t in trips}

    # ---- successor options per trip (SoC-independent; SoC is checked during the search)
    direct: dict[str, list[Trip]] = {}
    via_charge: dict[str, list[tuple[Trip, _Station, float, float]]] = {}
    for i in trips:
        later = [j for j in trips if j.dep_min > i.arr_min]
        direct[i.id] = [j for j in later if inst.can_follow(i, j, max_gap=cfg.max_gap_min)][:cfg.max_successors]
        opts: list[tuple[Trip, _Station, float, float]] = []
        for j in later:
            if j.dep_min - i.arr_min > cfg.max_charge_gap_min:
                break
            cand = []
            for st in stations:
                t_arr = i.arr_min + inst.dh_min(i.end, st.location)
                t_leave = j.dep_min - layover - inst.dh_min(st.location, j.start)
                if t_leave - t_arr >= min_charge:
                    detour = inst.dh_km(i.end, st.location) + inst.dh_km(st.location, j.start)
                    cand.append((detour, st.id, st, t_arr, t_leave))
            cand.sort(key=lambda x: (x[0], x[1]))
            opts += [(j, st, t_arr, t_leave) for _, _, st, t_arr, t_leave in cand[:cfg.max_charge_stations]]
            if len({o[0].id for o in opts}) >= cfg.max_successors:
                break
        via_charge[i.id] = opts

    columns: list[Column] = []
    per_start = 0
    truncated_starts = 0
    hit_global = False

    def emit(path: list[str], soc: float, dh_km: float, charges: list[ChargeStop], min_soc: float, t0: float):
        nonlocal per_start
        last = by_id[path[-1]]
        pull_in = inst.dh_km(last.end, depot)
        soc_end = soc - pull_in * kwh_km
        if soc_end < soc_min_kwh - EPS:
            return
        total_dh = dh_km + pull_in
        energy_by_slot: dict[int, float] = defaultdict(float)
        power_by_station: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        day_cost = 0.0
        for ch in charges:
            for t, minutes in _slot_overlaps(ch.start_min, ch.end_min, slot):
                kwh = ch.power_kw * minutes / 60.0
                energy_by_slot[t] += kwh
                power_by_station[ch.station][t] += ch.power_kw * minutes / slot
                day_cost += price.get(t, a["tariff_day"]) * kwh
        refill = max(0.0, soc_start_kwh - soc_end) / eta
        energy_cost = day_cost + night_price * refill
        dh_cost = a["deadhead_cost_per_km"] * total_dh
        cost = a["vehicle_cost"] + dh_cost + energy_cost
        if duals is not None and cost - sum(duals.get(t, 0.0) for t in path) >= -rc_tol:
            return
        sig = signature(path, charges)
        if sig in skip:
            return
        skip.add(sig)
        columns.append(Column(
            id=f"k{len(columns)}", trips=list(path),
            cost=cost,
            vehicle_cost=a["vehicle_cost"], deadhead_km=total_dh, deadhead_cost=dh_cost,
            energy_cost=energy_cost, grid_kwh_day=sum(ch.grid_kwh for ch in charges), refill_kwh=refill,
            min_soc=min(min_soc, soc_end) / cap, final_soc=soc_end / cap,
            start_min=t0, end_min=last.arr_min + inst.dh_min(last.end, depot),
            charges=list(charges),
            energy_by_slot=dict(energy_by_slot),
            power_by_station={s: dict(v) for s, v in power_by_station.items()},
        ))
        per_start += 1

    def order(js: list[Trip]) -> list[Trip]:
        return js if duals is None else sorted(js, key=lambda j: -duals.get(j.id, 0.0))

    def order_opts(opts):
        return opts if duals is None else sorted(opts, key=lambda o: -duals.get(o[0].id, 0.0))

    # every first trip gets an equal share, so late trips are never starved by the global cap
    per_start_cap = max(1, min(cfg.max_columns_per_start, cfg.max_columns // max(1, len(trips))))

    nodes = 0

    def full() -> bool:
        return (per_start >= per_start_cap or nodes >= cfg.max_nodes_per_start
                or len(columns) >= cfg.max_columns)

    def dfs(path: list[str], soc: float, dh_km: float, charges: list[ChargeStop], min_soc: float, t0: float):
        nonlocal nodes
        if full():
            return
        nodes += 1
        emit(path, soc, dh_km, charges, min_soc, t0)
        if len(path) >= cfg.max_trips:
            return
        i = by_id[path[-1]]
        for j in order(direct[i.id]):                            # direct connection
            km = inst.dh_km(i.end, j.start)
            s1 = soc - km * kwh_km - j.energy_kwh
            if s1 >= soc_min_kwh - EPS:
                dfs(path + [j.id], s1, dh_km + km, charges, min(min_soc, s1), t0)
        if soc / cap >= cfg.charge_below:
            return
        for j, st, t_arr, t_leave in order_opts(via_charge[i.id]):  # connection with a charging stop
            km_in = inst.dh_km(i.end, st.location)
            s_arr = soc - km_in * kwh_km
            if s_arr < soc_min_kwh - EPS:
                continue
            need_min = (soc_max_kwh - s_arr) / (eta * st.power_kw / 60.0)
            dur = min(t_leave - t_arr, need_min)
            if dur < min_charge - EPS:
                continue                                         # battery (nearly) full: no useful charge
            s_chg = s_arr + eta * st.power_kw * dur / 60.0
            km_out = inst.dh_km(st.location, j.start)
            s1 = s_chg - km_out * kwh_km - j.energy_kwh
            if s1 < soc_min_kwh - EPS:
                continue
            stop = ChargeStop(station=st.id, location=st.location, start_min=t_arr, end_min=t_arr + dur,
                              power_kw=st.power_kw, grid_kwh=st.power_kw * dur / 60.0,
                              soc_before=s_arr / cap, soc_after=s_chg / cap)
            dfs(path + [j.id], s1, dh_km + km_in + km_out, charges + [stop], min(min_soc, s_arr, s1), t0)

    starts = trips if duals is None else sorted(trips, key=lambda t: -duals.get(t.id, 0.0))
    for first in starts:
        km = inst.dh_km(depot, first.start)
        s0 = soc_start_kwh - km * kwh_km - first.energy_kwh
        if s0 < soc_min_kwh - EPS:
            continue
        per_start = 0
        nodes = 0
        dfs([first.id], s0, km, [], s0, first.dep_min - inst.dh_min(depot, first.start))
        if per_start >= per_start_cap or nodes >= cfg.max_nodes_per_start:
            truncated_starts += 1
        if len(columns) >= cfg.max_columns:
            hit_global = True
            break

    covered = {t for c in columns for t in c.trips}
    stats = {
        "n_columns": len(columns),
        "n_with_charging": sum(1 for c in columns if c.charges),
        "n_charge_events": sum(len(c.charges) for c in columns),
        "max_trips_in_column": max((c.n_trips for c in columns), default=0),
        "trips_uncovered": sorted(set(by_id) - covered),
        "per_start_cap": per_start_cap,
        "starts_truncated": truncated_starts,
        "hit_max_columns": hit_global,
    }
    return ColumnSet(instance_name=inst.name, trip_ids=[t.id for t in trips], columns=columns,
                     station_capacity={s.id: s.capacity for s in all_stations}, slot_min=slot,
                     buffer_min=a.get("charge_buffer_min", 10.0), config=cfg.to_dict(), stats=stats,
                     assumptions=dict(a))


# --------------------------------------------------------------------------- validation

def validate_column(inst: Instance, col: Column, tol: float = 1e-6) -> list[str]:
    """Replay a column independently of the generator; returns a list of problems."""
    a = inst.assumptions
    cap, kwh_km, eta = a["battery_kwh"], a["kwh_per_km"], a["charge_efficiency"]
    by_id = inst.trip_by_id
    errors = []
    if not col.trips or len(set(col.trips)) != len(col.trips):
        return ["trips must be nonempty and unique"]
    if any(t not in by_id for t in col.trips):
        return ["unknown trip ID"]
    values = [col.start_min, col.end_min, col.min_soc, col.final_soc, col.grid_kwh_day]
    values += [v for ch in col.charges for v in
               (ch.start_min, ch.end_min, ch.power_kw, ch.grid_kwh, ch.soc_before, ch.soc_after)]
    if not all(math.isfinite(v) for v in values):
        return ["column times, energy and SoC must be finite"]
    stations = {s.id: s for s in _stations(inst)}
    for ch in col.charges:
        station = stations.get(ch.station)
        if station is None or ch.location != station.location:
            return [f"unknown station or wrong charging location: {ch.station}"]
        if ch.power_kw <= 0 or ch.power_kw > station.power_kw + tol:
            errors.append(f"{ch.station}: invalid charging power")
        if ch.end_min - ch.start_min < a["min_charge_min"] - tol:
            errors.append(f"{ch.station}: charging duration below minimum")
        if abs(ch.grid_kwh - ch.power_kw * (ch.end_min - ch.start_min) / 60) > 1e-3:
            errors.append(f"{ch.station}: grid energy mismatch")
    if errors:
        return errors
    soc = a["soc_start"] * cap
    minimum = soc
    loc, now = inst.depot, col.start_min
    pending = sorted(col.charges, key=lambda c: c.start_min)
    for n, tid in enumerate(col.trips):
        t = by_id[tid]
        if n > 0 and pending and pending[0].start_min < t.dep_min:
            ch = pending.pop(0)
            soc -= inst.dh_km(loc, ch.location) * kwh_km
            minimum = min(minimum, soc)
            if now + inst.dh_min(loc, ch.location) > ch.start_min + tol:
                errors.append(f"{tid}: cannot reach charger {ch.station} in time")
            if soc < a["soc_min"] * cap - tol:
                errors.append(f"{tid}: SoC below minimum on the way to charger {ch.station}")
            if abs(soc / cap - ch.soc_before) > 1e-3:
                errors.append(f"{ch.station}: SoC before charging mismatch")
            soc += eta * ch.power_kw * (ch.end_min - ch.start_min) / 60.0
            if soc > a["soc_max"] * cap + tol:
                errors.append(f"{ch.station}: charging exceeds battery capacity")
            if abs(soc / cap - ch.soc_after) > 1e-3:
                errors.append(f"{ch.station}: SoC after charging mismatch")
            loc, now = ch.location, ch.end_min
        soc -= inst.dh_km(loc, t.start) * kwh_km
        pause = a["min_layover_min"] if n > 0 else 0
        if now + inst.dh_min(loc, t.start) + pause > t.dep_min + 1e-3:
            errors.append(f"{tid}: not reachable in time")
        if soc < a["soc_min"] * cap - tol:
            errors.append(f"{tid}: SoC below minimum before the trip")
        soc -= t.energy_kwh
        minimum = min(minimum, soc)
        if soc < a["soc_min"] * cap - tol:
            errors.append(f"{tid}: SoC below minimum after the trip")
        loc, now = t.end, t.arr_min
    soc -= inst.dh_km(loc, inst.depot) * kwh_km
    minimum = min(minimum, soc)
    if abs(col.end_min - (now + inst.dh_min(loc, inst.depot))) > 1e-3:
        errors.append("pull-in: arrival time mismatch")
    if abs(minimum / cap - col.min_soc) > 1e-3:
        errors.append("minimum SoC mismatch")
    if abs(sum(ch.grid_kwh for ch in col.charges) - col.grid_kwh_day) > 1e-3:
        errors.append("total grid energy mismatch")
    if soc < a["soc_min"] * cap - tol:
        errors.append("pull-in: SoC below minimum")
    if abs(soc / cap - col.final_soc) > 1e-3:
        errors.append(f"final SoC mismatch: replay {soc / cap:.4f} vs column {col.final_soc:.4f}")
    if pending:
        errors.append("charging stop not matched to a gap")
    return errors


def check_schedule(cs: ColumnSet, indices: list[int]) -> dict:
    """Is `indices` a valid schedule: every trip exactly once, charger capacity respected?"""
    count: dict[str, int] = defaultdict(int)
    for k in indices:
        for t in cs.columns[k].trips:
            count[t] += 1
    missing = [t for t in cs.trip_ids if count[t] == 0]
    double = [t for t, n in count.items() if n > 1]
    over = []
    events: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for k in indices:
        for ch in cs.columns[k].charges:
            events[ch.station] += [(ch.start_min, 1), (ch.end_min, -1)]
    for s, ev in events.items():
        level = 0
        for _, d in sorted(ev, key=lambda e: (e[0], e[1])):   # ends before starts at equal times
            level += d
            if level > cs.station_capacity.get(s, 1):
                over.append(s)
                break
    return {"ok": not (missing or double or over), "missing": missing, "double": double, "over_capacity": over}


# --------------------------------------------------------------------------- LP / MILP

def _matrices(cs: ColumnSet, indices: list[int]):
    """Coverage A_eq (= 1) and charger capacity A_ub (<= capacity).

    Capacity rows: for every station and every charging start time tau, the
    columns charging at tau form a clique; sum <= capacity. This is exact for
    interval overlaps and has at most one row per charging event.
    """
    from scipy.sparse import coo_matrix

    row = {t: r for r, t in enumerate(cs.trip_ids)}
    r_idx, c_idx = [], []
    for c, k in enumerate(indices):
        for t in cs.columns[k].trips:
            r_idx.append(row[t])
            c_idx.append(c)
    a_eq = coo_matrix((np.ones(len(r_idx)), (r_idx, c_idx)), shape=(len(cs.trip_ids), len(indices))).tocsr()

    per_station: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    for c, k in enumerate(indices):
        for ch in cs.columns[k].charges:
            per_station[ch.station].append((ch.start_min, ch.end_min, c))
    rows_r, rows_c, b_ub = [], [], []
    n_rows = 0
    for s, iv in per_station.items():
        cap = cs.station_capacity.get(s, 1)
        iv.sort()
        active: list[tuple[float, int]] = []          # heap of (end, column) charging right now
        seen: set[tuple[int, ...]] = set()
        x = 0
        while x < len(iv):
            tau = iv[x][0]
            while x < len(iv) and iv[x][0] <= tau + EPS:  # all starts at this instant
                heapq.heappush(active, (iv[x][1], iv[x][2]))
                x += 1
            while active and active[0][0] <= tau + EPS:  # ended before tau
                heapq.heappop(active)
            clique = tuple(sorted({c for _, c in active}))
            if len(clique) <= cap or clique in seen:
                continue
            seen.add(clique)
            rows_r += [n_rows] * len(clique)
            rows_c += list(clique)
            b_ub.append(cap)
            n_rows += 1
    a_ub = (coo_matrix((np.ones(len(rows_r)), (rows_r, rows_c)), shape=(n_rows, len(indices))).tocsr()
            if n_rows else None)
    return a_eq, a_ub, np.array(b_ub, dtype=float)


@dataclass
class Solution:
    columns: list[int]
    objective: float
    n_vehicles: int
    status: str


def solve_milp(cs: ColumnSet, indices: Optional[list[int]] = None, time_limit: float = 60.0) -> Solution:
    """Exact set partitioning with charger capacity (no peak term) on `indices`."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    idx = list(range(len(cs.columns))) if indices is None else list(indices)
    c = cs.costs(idx)
    a_eq, a_ub, b_ub = _matrices(cs, idx)
    cons = [LinearConstraint(a_eq, np.ones(a_eq.shape[0]), np.ones(a_eq.shape[0]))]
    if a_ub is not None:
        cons.append(LinearConstraint(a_ub, -np.inf, b_ub))
    res = milp(c, constraints=cons, integrality=np.ones(len(idx)), bounds=Bounds(0, 1),
               options={"time_limit": time_limit})
    if res.x is None:
        return Solution(columns=[], objective=math.inf, n_vehicles=0, status=res.message)
    picked = [idx[i] for i in np.flatnonzero(res.x > 0.5)]
    return Solution(columns=picked, objective=float(res.fun), n_vehicles=len(picked), status=res.message)


def greedy_cover(cs: ColumnSet) -> list[int]:
    """Feasible partition: take columns by cost per trip if they overlap nothing taken
    and keep every station within capacity. Single-trip columns close the gaps."""
    covered: set[str] = set()
    taken: list[int] = []
    order = sorted(range(len(cs.columns)), key=lambda k: cs.columns[k].cost / cs.columns[k].n_trips)
    for k in order:
        col = cs.columns[k]
        if covered.isdisjoint(col.trips) and check_schedule(cs, taken + [k])["over_capacity"] == []:
            taken.append(k)
            covered.update(col.trips)
    if covered != set(cs.trip_ids):
        raise RuntimeError("greedy cover failed: some trips have no admissible column")
    return taken


@dataclass
class Selection:
    indices: list[int]
    lp_objective: float
    lp_support: list[int]
    anchor: list[int]
    reduced_costs: np.ndarray
    lp_values: np.ndarray


def select_columns(cs: ColumnSet, budget: int, keep_singletons: bool = True,
                   anchor: str = "milp", milp_time_limit: float = 30.0) -> Selection:
    """Pick at most `budget` columns (= qubits) using the LP relaxation.

    Priority, until the budget is full:
      1. an integer feasible schedule ("anchor": exact MILP, or greedy if the
         MILP finds nothing in time), so the selection always contains a full
         cover and the QUBO has a feasible ground state,
      2. columns used by the LP optimum,
      3. one single-trip column per trip,
      4. the rest by LP reduced cost.
    """
    lp = solve_lp(cs)
    y, rc = lp.values, lp.reduced_costs

    anchor_cols: list[int] = []
    if anchor == "milp":
        anchor_cols = solve_milp(cs, time_limit=milp_time_limit).columns
    if not anchor_cols and anchor in ("milp", "greedy"):
        anchor_cols = greedy_cover(cs)
    if len(anchor_cols) > budget:
        raise ValueError(f"budget {budget} is smaller than the anchor schedule ({len(anchor_cols)} columns)")

    chosen: list[int] = []
    seen: set[int] = set()

    def take(k: int):
        if k not in seen and len(chosen) < budget:
            chosen.append(int(k))
            seen.add(int(k))

    for k in anchor_cols:
        take(k)
    support = [int(k) for k in np.argsort(-y) if y[k] > 1e-7]
    for k in support:
        take(k)
    if keep_singletons:
        best_single: dict[str, int] = {}
        for k, col in enumerate(cs.columns):
            if col.n_trips == 1:
                t = col.trips[0]
                if t not in best_single or col.cost < cs.columns[best_single[t]].cost:
                    best_single[t] = k
        for t in cs.trip_ids:
            if t in best_single:
                take(best_single[t])
    for k in np.argsort(rc):
        take(int(k))
    return Selection(indices=sorted(chosen), lp_objective=lp.objective, lp_support=sorted(support),
                     anchor=sorted(anchor_cols), reduced_costs=rc, lp_values=y)


@dataclass
class LPResult:
    objective: float
    values: np.ndarray
    reduced_costs: np.ndarray
    trip_duals: dict[str, float]


def solve_lp(cs: ColumnSet) -> LPResult:
    """LP relaxation of set partitioning with exact charger capacity."""
    from scipy.optimize import linprog

    covered = {t for col in cs.columns for t in col.trips}
    missing = [t for t in cs.trip_ids if t not in covered]
    if missing:
        raise ValueError(f"{len(missing)} trips have no column (e.g. {missing[:3]}); "
                         "they cannot be served from the depot on one battery")
    idx = list(range(len(cs.columns)))
    c = cs.costs()
    a_eq, a_ub, b_ub = _matrices(cs, idx)
    res = linprog(c, A_eq=a_eq, b_eq=np.ones(a_eq.shape[0]), A_ub=a_ub, b_ub=b_ub if a_ub is not None else None,
                  bounds=(0, 1), method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP relaxation failed: {res.message}")
    pi = res.eqlin.marginals
    rc = c - a_eq.T @ pi
    if a_ub is not None:
        rc = rc - a_ub.T @ res.ineqlin.marginals
    return LPResult(objective=float(res.fun), values=res.x, reduced_costs=rc,
                    trip_duals={t: float(pi[r]) for r, t in enumerate(cs.trip_ids)})


def solve_restricted_master(cs: ColumnSet, n: int = 1500, time_limit: float = 60.0) -> Solution:
    """MILP on the n columns with lowest LP reduced cost (+ LP support + single-trip columns).

    The usual price-and-branch heuristic for pools too big for an exact MILP.
    """
    lp = solve_lp(cs)
    keep = set(np.argsort(lp.reduced_costs)[:n].tolist())
    keep |= {k for k in range(len(cs.columns)) if lp.values[k] > 1e-7}
    keep |= {k for k, c in enumerate(cs.columns) if c.n_trips == 1}
    return solve_milp(cs, sorted(keep), time_limit=time_limit)


def column_generation(inst: Instance, cfg: Optional[GenConfig] = None, init_per_start: int = 30,
                      price_per_start: int = 20, max_iters: int = 30, verbose: bool = False) -> ColumnSet:
    """Heuristic column generation for instances too big to enumerate.

    1. small initial pool (DFS with `init_per_start` columns per first trip),
    2. repeat: LP -> trip duals -> dual-guided DFS keeps only negative reduced cost
       columns (`price_per_start` per first trip) -> add to the pool,
    3. stop when pricing finds nothing or after `max_iters`.
    Charger-capacity duals are not used in pricing (capacity is still exact in the LP).
    """
    from dataclasses import replace

    cfg = cfg or GenConfig()
    pool = generate_columns(inst, replace(cfg, max_columns_per_start=init_per_start))
    seen = {signature(c.trips, c.charges) for c in pool.columns}
    history = []
    for it in range(max_iters):
        lp = solve_lp(pool)
        new = generate_columns(inst, replace(cfg, max_columns_per_start=price_per_start),
                               duals=lp.trip_duals, skip=seen)
        history.append({"iter": it, "lp": round(lp.objective, 3), "pool": len(pool.columns),
                        "added": len(new.columns)})
        if verbose:
            print(f"  CG iter {it}: LP {lp.objective:.1f}, pool {len(pool.columns)}, +{len(new.columns)}")
        if not new.columns:
            break
        pool.columns.extend(new.columns)
    for n, col in enumerate(pool.columns):
        col.id = f"k{n}"
    pool.stats = {**pool.stats, "n_columns": len(pool.columns),
                  "n_with_charging": sum(1 for c in pool.columns if c.charges),
                  "column_generation": history}
    return pool

"""Disruption repair: re-assign the trips of a failed bus or charger (package F).

After a disruption, every affected trip t gets exactly one option o:
another bus of the schedule, a reserve bus from the depot, or "cancel".

    z[t, o] = 1   trip t is taken by option o              (one qubit per pair)

    H(z) = sum c[t,o] z[t,o]                                 extra cost of each option
         + P sum_t (sum_o z[t,o] - 1)^2                      exactly one option per trip
         + Q sum_{conflicts} z[t,o] z[t',o]                  one bus cannot do both trips

Feasibility of every option (and every pair on the same bus) is checked by
replaying the bus's whole new duty with `plan_duty`: times, deadheads, battery
and charging stops, with stations blocked by the outage and by other buses'
charging. The pairwise model is exact for up to two inserted trips per bus;
`apply_assignment` replays the final plan and reports joint feasibility.

The one-hot groups are what the XY mixer (evsp/xy_qaoa.py) preserves.

    inst, cs = ...; sched = Schedule.from_columns(cs)
    ev = Disruption.bus_failure(sched, "k4", at_min=9 * 60)
    rp = build_repair_problem(inst, sched, ev)
    q = rp.to_qubo()                          # QUBO with one-hot penalties (for X-mixer QAOA, SA)
    plan = apply_assignment(inst, rp, x)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Optional

import numpy as np

from .columns import ChargeStop, Column, ColumnSet, _slot_overlaps, _stations
from .instance import Instance
from .qubo import QUBO, Weights

EPS = 1e-9


# --------------------------------------------------------------------------- duty replay

def plan_duty(inst: Instance, trip_ids: list[str], earliest_start: Optional[float] = None,
              blocked: Optional[dict[str, list[tuple[float, float]]]] = None,
              charge_below: float = 0.6, max_stations: int = 3, vehicle_cost: Optional[float] = None,
              col_id: str = "plan", history: Optional[list[ChargeStop]] = None,
              not_before: Optional[float] = None) -> Optional[Column]:
    """Greedy, deterministic duty for a fixed trip sequence, or None if infeasible.

    depot -> trips in departure order -> depot. Between two trips the bus charges
    (as long as the gap allows, up to soc_max) at the nearest unblocked station if
    its SoC is below `charge_below`, or if the next trip is otherwise out of
    battery reach. `blocked[station]` lists unavailable time windows.

    Re-planning after an event at time tau: `history` are the charging stops that
    already started before tau (kept as they are, in the gap they belong to) and
    `not_before=tau` forbids any other charging stop from starting earlier, so the
    plan only changes what happens after the event.
    """
    a = inst.assumptions
    tb = inst.trip_by_id
    trips = sorted((tb[t] for t in trip_ids), key=lambda t: (t.dep_min, t.id))
    if not trips or len({t.id for t in trips}) != len(trips):
        return None
    cap = a["battery_kwh"]
    soc_min, soc_max = a["soc_min"] * cap, a["soc_max"] * cap
    kwh_km, eta, layover = a["kwh_per_km"], a["charge_efficiency"], a["min_layover_min"]
    min_charge = a["min_charge_min"]
    depot = inst.depot
    stations = _stations(inst)
    blocked = blocked or {}

    def is_blocked(station: str, start: float, end: float) -> bool:
        return any(s < end - EPS and start < e - EPS for s, e in blocked.get(station, []))

    first = trips[0]
    t0 = first.dep_min - inst.dh_min(depot, first.start)
    if earliest_start is not None and t0 < earliest_start - EPS:
        return None
    km = inst.dh_km(depot, first.start)
    soc = a["soc_start"] * cap - km * kwh_km - first.energy_kwh
    dh_km, charges, min_soc = km, [], soc
    if soc < soc_min - EPS:
        return None

    history = sorted(history or [], key=lambda c: c.start_min)

    def fixed_charge(i, j, soc_now, ch):
        km_in, km_out = inst.dh_km(i.end, ch.location), inst.dh_km(ch.location, j.start)
        s_arr = soc_now - km_in * kwh_km
        if s_arr < soc_min - EPS:
            return None
        s_chg = min(soc_max, s_arr + eta * ch.power_kw * (ch.end_min - ch.start_min) / 60.0)
        s1 = s_chg - km_out * kwh_km - j.energy_kwh
        if s1 < soc_min - EPS:
            return None
        return replace(ch, soc_before=s_arr / cap, soc_after=s_chg / cap), s1, km_in + km_out, min(s_arr, s1)

    def try_charge(i, j, soc_now):
        opts = []
        ready = i.arr_min if not_before is None else max(i.arr_min, not_before)
        for st in stations:
            t_arr = ready + inst.dh_min(i.end, st.location)
            t_leave = j.dep_min - layover - inst.dh_min(st.location, j.start)
            if t_leave - t_arr < min_charge:
                continue
            km_in, km_out = inst.dh_km(i.end, st.location), inst.dh_km(st.location, j.start)
            opts.append((km_in + km_out, st.id, st, t_arr, t_leave, km_in, km_out))
        for _, _, st, t_arr, t_leave, km_in, km_out in sorted(opts)[:max_stations]:
            s_arr = soc_now - km_in * kwh_km
            if s_arr < soc_min - EPS:
                continue
            dur = min(t_leave - t_arr, (soc_max - s_arr) / (eta * st.power_kw / 60.0))
            if dur < min_charge - EPS or is_blocked(st.id, t_arr, t_arr + dur):
                continue
            s_chg = s_arr + eta * st.power_kw * dur / 60.0
            s1 = s_chg - km_out * kwh_km - j.energy_kwh
            if s1 < soc_min - EPS:
                continue
            stop = ChargeStop(station=st.id, location=st.location, start_min=t_arr, end_min=t_arr + dur,
                              power_kw=st.power_kw, grid_kwh=st.power_kw * dur / 60.0,
                              soc_before=s_arr / cap, soc_after=s_chg / cap)
            return stop, s1, km_in + km_out, min(s_arr, s1)
        return None

    def charge_home(i, soc_now):
        """Shortest charge after the last trip that is enough to get back to the depot."""
        ready = i.arr_min if not_before is None else max(i.arr_min, not_before)
        opts = sorted((inst.dh_km(i.end, st.location) + inst.dh_km(st.location, depot), st.id, st)
                      for st in stations if st.location != depot)
        for _, _, st in opts[:max_stations]:
            km_in, km_out = inst.dh_km(i.end, st.location), inst.dh_km(st.location, depot)
            s_arr = soc_now - km_in * kwh_km
            if s_arr < soc_min - EPS:
                continue
            rate = eta * st.power_kw / 60.0
            dur = max(min_charge, (soc_min + km_out * kwh_km - s_arr) / rate)
            start = ready + inst.dh_min(i.end, st.location)
            for _ in range(20):                       # wait until the station is free
                clash = [e for s_, e in blocked.get(st.id, []) if s_ < start + dur - EPS and start < e - EPS]
                if not clash:
                    break
                start = max(clash)
            else:
                continue
            s_chg = min(soc_max, s_arr + rate * dur)
            s_end = s_chg - km_out * kwh_km
            if s_end < soc_min - EPS:
                continue
            stop = ChargeStop(station=st.id, location=st.location, start_min=start, end_min=start + dur,
                              power_kw=st.power_kw, grid_kwh=st.power_kw * dur / 60.0,
                              soc_before=s_arr / cap, soc_after=s_chg / cap)
            return stop, s_end, km_in + km_out, min(s_arr, s_end), start + dur + inst.dh_min(st.location, depot)
        return None

    for i, j in zip(trips, trips[1:]):
        if not inst.can_follow(i, j):
            return None
        fixed = [ch for ch in history if i.arr_min - EPS <= ch.start_min < j.dep_min]
        if fixed:                                     # charging that already happened before the event
            charged = fixed_charge(i, j, soc, fixed[0])
            if charged is None:
                return None
            stop, s1, km, low = charged
            charges.append(stop)
            soc, dh_km, min_soc = s1, dh_km + km, min(min_soc, low)
            continue
        charged = try_charge(i, j, soc) if soc / cap < charge_below else None
        if charged is None:
            km = inst.dh_km(i.end, j.start)
            s1 = soc - km * kwh_km - j.energy_kwh
            if s1 >= soc_min - EPS:
                soc, dh_km, min_soc = s1, dh_km + km, min(min_soc, s1)
                continue
            charged = try_charge(i, j, soc)          # forced charge: next trip out of reach
            if charged is None:
                return None
        stop, s1, km, low = charged
        charges.append(stop)
        soc, dh_km, min_soc = s1, dh_km + km, min(min_soc, low)
    last = trips[-1]
    km = inst.dh_km(last.end, depot)
    soc_end = soc - km * kwh_km
    t_end = last.arr_min + inst.dh_min(last.end, depot)
    if soc_end < soc_min - EPS:                       # cannot reach the depot: top up on the way home
        home = charge_home(last, soc)
        if home is None:
            return None
        stop, soc_end, km, low, t_end = home
        charges.append(stop)
        min_soc = min(min_soc, low)
    return _make_column(inst, col_id, [t.id for t in trips], charges, dh_km + km, min(min_soc, soc_end),
                        soc_end, t0, t_end, vehicle_cost)


def _make_column(inst: Instance, col_id: str, trips: list[str], charges: list[ChargeStop], dh_km: float,
                 min_soc: float, soc_end: float, t0: float, t1: float, vehicle_cost: Optional[float]) -> Column:
    """Same cost model as the column generator (evsp/columns.py)."""
    a = inst.assumptions
    cap, eta, slot = a["battery_kwh"], a["charge_efficiency"], int(a["slot_min"])
    price = {s["t"]: s["price"] for s in inst.tariff}
    energy_by_slot: dict[int, float] = defaultdict(float)
    power_by_station: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    day_cost = 0.0
    for ch in charges:
        for t, minutes in _slot_overlaps(ch.start_min, ch.end_min, slot):
            kwh = ch.power_kw * minutes / 60.0
            energy_by_slot[t] += kwh
            power_by_station[ch.station][t] += ch.power_kw * minutes / slot
            day_cost += price.get(t, a["tariff_day"]) * kwh
    refill = max(0.0, a["soc_start"] * cap - soc_end) / eta
    energy_cost = day_cost + a["tariff_night"] * refill
    veh = a["vehicle_cost"] if vehicle_cost is None else vehicle_cost
    dh_cost = a["deadhead_cost_per_km"] * dh_km
    return Column(id=col_id, trips=trips, cost=veh + dh_cost + energy_cost, vehicle_cost=veh, deadhead_km=dh_km,
                  deadhead_cost=dh_cost, energy_cost=energy_cost, grid_kwh_day=sum(c.grid_kwh for c in charges),
                  refill_kwh=refill, min_soc=min_soc / cap, final_soc=soc_end / cap, start_min=t0, end_min=t1,
                  charges=list(charges), energy_by_slot=dict(energy_by_slot),
                  power_by_station={s: dict(v) for s, v in power_by_station.items()})


# --------------------------------------------------------------------------- schedule and events

@dataclass
class Schedule:
    duties: dict[str, Column]               # bus id -> duty

    @classmethod
    def from_columns(cls, cs: ColumnSet, indices: Optional[list[int]] = None) -> "Schedule":
        """Default: the optimal schedule stored by build_columns.py (best_on_selection)."""
        if indices is None:
            best = cs.extra.get("best_on_selection")
            if best:
                indices = best["columns"]
            else:
                from .columns import solve_milp
                indices = solve_milp(cs).columns
        return cls({cs.columns[k].id: cs.columns[k] for k in indices})

    def occupied(self, exclude: tuple[str, ...] = ()) -> dict[str, list[tuple[float, float]]]:
        """Charging intervals of the other buses, per station."""
        out: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for bus, col in self.duties.items():
            if bus not in exclude:
                for ch in col.charges:
                    out[ch.station].append((ch.start_min, ch.end_min))
        return out


@dataclass
class Disruption:
    kind: str                                 # "bus" or "charger"
    at_min: float                             # time the disruption starts
    bus: Optional[str] = None
    station: Optional[str] = None
    until_min: Optional[float] = None         # charger outage end

    @classmethod
    def bus_failure(cls, schedule: Schedule, bus: str, at_min: float) -> "Disruption":
        if bus not in schedule.duties:
            raise ValueError(f"unknown bus {bus!r}; buses: {sorted(schedule.duties)}")
        return cls(kind="bus", at_min=at_min, bus=bus)

    @classmethod
    def charger_outage(cls, station: str, start_min: float, end_min: float) -> "Disruption":
        if end_min <= start_min:
            raise ValueError("outage must end after it starts")
        return cls(kind="charger", at_min=start_min, station=station, until_min=end_min)

    def describe(self) -> str:
        if self.kind == "bus":
            return f"bus {self.bus} fails at {_hhmm(self.at_min)}"
        return f"charger {self.station} out {_hhmm(self.at_min)}-{_hhmm(self.until_min)}"


def _hhmm(m: float) -> str:
    m = int(round(m))
    return f"{m // 60:02d}:{m % 60:02d}"


def _history_matches(before: Column, after: Column, inst: Instance, at_min: float) -> bool:
    """Return whether a replanned duty preserves all work and charging before an event."""
    tb = inst.trip_by_id
    before_trips = [t for t in before.trips if tb[t].dep_min < at_min]
    after_trips = [t for t in after.trips if tb[t].dep_min < at_min]
    if before_trips != after_trips:
        return False

    def past_charges(col: Column):
        return sorted((ch.station, round(ch.start_min, 6), round(min(ch.end_min, at_min), 6),
                       round(ch.power_kw, 6))
                      for ch in col.charges if ch.start_min < at_min)

    return past_charges(before) == past_charges(after)


def _past_charges(col: Column, event: "Disruption") -> list[ChargeStop]:
    """Charging stops of a duty that started before the event (cut at the outage start)."""
    out = []
    for ch in col.charges:
        if ch.start_min >= event.at_min:
            continue
        if event.kind == "charger" and ch.station == event.station and ch.end_min > event.at_min:
            ch = replace(ch, end_min=event.at_min, grid_kwh=ch.power_kw * (event.at_min - ch.start_min) / 60.0)
        out.append(ch)
    return out


def _replan(inst: Instance, col: Column, trips: list[str], event: "Disruption",
            blocked: dict[str, list[tuple[float, float]]], col_id: str) -> Optional[Column]:
    """New duty of an existing bus for `trips`; everything before the event stays as it was."""
    plan = plan_duty(inst, trips, blocked=blocked, vehicle_cost=col.vehicle_cost, col_id=col_id,
                     history=_past_charges(col, event), not_before=event.at_min)
    return plan if plan is not None and _history_matches(col, plan, inst, event.at_min) else None


# --------------------------------------------------------------------------- repair problem

@dataclass
class Option:
    trip: str
    kind: str                                 # "bus", "reserve" or "cancel"
    target: Optional[str]                     # bus id or reserve id
    cost: float                               # extra cost against the disrupted schedule

    @property
    def name(self) -> str:
        return f"{self.trip}->{self.target or 'cancel'}"


@dataclass
class RepairProblem:
    event: Disruption
    trips: list[str]                          # affected trips (groups), in departure order
    options: list[Option]                     # one qubit each
    groups: list[list[int]]                   # option indices per trip (one-hot)
    conflicts: list[tuple[int, int]]          # same bus cannot take both
    base_plans: dict[str, Column]             # unaffected / shortened duties after the event
    blocked: dict[str, list[tuple[float, float]]]
    settings: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.options)

    @property
    def subspace_size(self) -> int:
        return int(np.prod([len(g) for g in self.groups]))

    def to_qubo(self, P: Optional[float] = None, Q: Optional[float] = None, penalty_factor: float = 2.0) -> QUBO:
        """QUBO with one-hot penalties P (needed by X-mixer QAOA and SA; zero inside the XY subspace)."""
        costs = np.array([o.cost for o in self.options], dtype=float)
        pen = penalty_factor * max(float(costs.max(initial=0.0)), 1.0)
        P = pen if P is None else P
        Q = pen if Q is None else Q
        lin = costs.copy()
        quad: dict[tuple[int, int], float] = defaultdict(float)
        offset = 0.0
        for members in self.groups:
            for a_ in members:
                lin[a_] -= P
            for x, a_ in enumerate(members):
                for b_ in members[x + 1:]:
                    quad[(a_, b_)] += 2 * P
            offset += P
        for k, l in self.conflicts:
            quad[(min(k, l), max(k, l))] += Q
        n = self.n
        return QUBO(linear=lin, quadratic={kl: v for kl, v in quad.items() if abs(v) > 1e-12}, offset=offset,
                    names=[o.name for o in self.options], weights=Weights(variant="repair"),
                    meta={"name": "repair", "variant": "repair", "P": P, "Q": Q, "M": 0.0,
                          "event": self.event.describe(), "scale": 1.0},
                    _cover=[list(g) for g in self.groups], _conflicts=list(self.conflicts), _tight=[],
                    _costs=costs, _power=np.zeros((n, 1, 1)), _stations=["TOTAL"], _trip_ids=list(self.trips))


def build_repair_problem(inst: Instance, schedule: Schedule, event: Disruption, max_bus_options: int = 3,
                         reserves: int = 1, reserve_cost: Optional[float] = None,
                         cancel_cost: Optional[float] = None, one_trip_per_bus: bool = False) -> RepairProblem:
    """Affected trips, their options and pairwise conflicts.

    Costs are extra cost against the disrupted schedule: inserting trip t into bus v
    costs plan(v + t) - plan(v); a reserve bus costs `reserve_cost` per trip plus
    its deadhead and energy (the activation is counted per trip, an upper bound
    for a reserve doing several trips); cancelling costs `cancel_cost` per trip.

    Conflicts: two options on the same bus conflict when that bus cannot drive both
    trips (checked by replay; `one_trip_per_bus=True` forbids every pair instead),
    and options of different buses conflict when their charging overlaps at one
    station. Combinations of three or more trips are only checked when a solution is
    replayed (apply_assignment, best_valid_state).
    """
    a = inst.assumptions
    reserve_cost = a["vehicle_cost"] if reserve_cost is None else reserve_cost
    cancel_cost = 3 * a["vehicle_cost"] if cancel_cost is None else cancel_cost
    tb = inst.trip_by_id

    blocked: dict[str, list[tuple[float, float]]] = defaultdict(list)
    if event.kind == "charger":
        blocked[event.station].append((event.at_min, event.until_min))

    # ---- what the event takes away
    base: dict[str, Column] = {}
    affected: list[str] = []
    for bus, col in schedule.duties.items():
        if event.kind == "bus" and bus == event.bus:
            # A trip already in progress when the bus fails is disrupted too.
            affected += [t for t in col.trips if tb[t].dep_min >= event.at_min or tb[t].arr_min > event.at_min]
            # The failed bus is gone. Keep only charger occupancy up to the failure time.
            past_charges = [replace(ch, end_min=min(ch.end_min, event.at_min),
                                    grid_kwh=ch.power_kw * (min(ch.end_min, event.at_min) - ch.start_min) / 60.0)
                            for ch in col.charges if ch.start_min < event.at_min]
            base[f"{bus}*"] = replace(col, charges=past_charges)
            continue
        hit = event.kind == "charger" and any(
            ch.station == event.station and ch.start_min < event.until_min and event.at_min < ch.end_min
            for ch in col.charges)
        if not hit:
            base[bus] = col
            continue
        others = _merge(blocked, schedule.occupied(exclude=(bus,)))
        plan = _replan(inst, col, col.trips, event, others, bus)
        if plan is not None:
            base[bus] = plan                                          # re-routed to another charger
            continue
        cut = min(ch.start_min for ch in col.charges if ch.station == event.station
                  and ch.start_min < event.until_min and event.at_min < ch.end_min)
        keep = [t for t in col.trips if tb[t].dep_min < cut]
        plan = None
        while keep:              # without that charge the bus may not even get back: hand over more trips
            plan = _replan(inst, col, keep, event, others, bus)
            if plan is not None:
                break
            keep = keep[:-1]
        if plan is not None:
            base[bus] = plan
            affected += [t for t in col.trips if t not in keep]
            continue
        # trips that started before the event were driven: only the rest is handed over
        affected += [t for t in col.trips if tb[t].dep_min >= event.at_min]
        if any(tb[t].dep_min < event.at_min for t in col.trips):
            # The whole duty may be impossible to replay even though its earlier
            # trips are already committed. Keep its past charger occupancy and
            # exclude it from future assignments, as for a failed bus.
            past_charges = [replace(ch, end_min=min(ch.end_min, event.at_min),
                                    grid_kwh=ch.power_kw * (min(ch.end_min, event.at_min) - ch.start_min) / 60.0)
                            for ch in col.charges if ch.start_min < event.at_min]
            base[f"{bus}*"] = replace(col, charges=past_charges)
    affected = sorted(set(affected), key=lambda t: (tb[t].dep_min, t))
    if not affected:
        raise ValueError(f"{event.describe()}: no trips are affected")
    buses = [b for b in base if not b.endswith("*")]

    def occupied_except(bus: Optional[str]) -> dict[str, list[tuple[float, float]]]:
        occ = defaultdict(list, {s: list(v) for s, v in blocked.items()})
        for b, col in base.items():
            if b != bus:
                for ch in col.charges:
                    occ[ch.station].append((ch.start_min, ch.end_min))
        return occ

    def plan_with(target: str, extra: list[str]) -> Optional[Column]:
        if target.startswith("R"):
            return plan_duty(inst, extra, earliest_start=event.at_min, blocked=occupied_except(None),
                             vehicle_cost=0.0, col_id=target)
        col = base[target]
        return _replan(inst, col, col.trips + extra, event, occupied_except(target), target)

    base_cost = {b: (_replan(inst, base[b], base[b].trips, event, occupied_except(b), b) or base[b]).cost
                 for b in buses}

    # ---- options per trip
    options: list[Option] = []
    groups: list[list[int]] = []
    option_plans: dict[int, Column] = {}
    for t in affected:
        cand = []
        for b in buses:
            plan = plan_with(b, [t])
            if plan is not None:
                cand.append((Option(t, "bus", b, plan.cost - base_cost[b]), plan))
        cand = sorted(cand, key=lambda item: item[0].cost)[:max_bus_options]
        for r in range(reserves):
            plan = plan_with(f"R{r + 1}", [t])
            if plan is not None:
                cand.append((Option(t, "reserve", f"R{r + 1}", reserve_cost + plan.cost), plan))
        cand.append((Option(t, "cancel", None, cancel_cost), None))
        groups.append(list(range(len(options), len(options) + len(cand))))
        for option, plan in cand:
            pos = len(options)
            options.append(option)
            if plan is not None:
                option_plans[pos] = plan

    # ---- pairwise conflicts
    conflicts = set()
    for k, l in combinations(range(len(options)), 2):
        ok, ol = options[k], options[l]
        if ok.trip == ol.trip or ok.target is None or ol.target is None:
            continue
        if ok.target == ol.target:
            # A target may take multiple affected trips when their combined duty
            # is feasible; only incompatible pairs belong in O_v.
            conflict = one_trip_per_bus or plan_with(ok.target, [ok.trip, ol.trip]) is None
        elif k in option_plans and l in option_plans:
            # Replace the two independently planned duties in the base schedule,
            # then check the actual connector capacity for their combined charges.
            replaced = {ok.target, ol.target}
            duties = [col for bus, col in base.items() if bus not in replaced]
            duties.extend((option_plans[k], option_plans[l]))
            conflict = not _capacity_ok(inst, duties)
        else:
            conflict = False
        if conflict:
            conflicts.add((k, l))
    conflicts = sorted(conflicts)

    return RepairProblem(event=event, trips=affected, options=options, groups=groups, conflicts=conflicts,
                         base_plans=base, blocked=dict(blocked),
                         settings={"max_bus_options": max_bus_options, "reserves": reserves,
                                   "reserve_cost": reserve_cost, "cancel_cost": cancel_cost,
                                   "one_trip_per_bus": one_trip_per_bus})


def _merge(*maps: dict[str, list[tuple[float, float]]]) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for m in maps:
        for s, v in m.items():
            out[s] += list(v)
    return out


# --------------------------------------------------------------------------- applying a solution

@dataclass
class RepairPlan:
    feasible: bool                            # one option per trip and every changed duty replays
    jointly_feasible: bool                    # every bus's full new duty is feasible together
    capacity_ok: bool                         # no 1-charger station is used twice at once
    assignment: dict[str, str]                # trip -> bus / reserve / "cancel"
    duties: dict[str, Optional[Column]]       # changed duties (None = infeasible)
    cancelled: list[str]
    extra_cost: float                         # true extra cost of the plan (reserve activation once)
    problems: list[str]


def apply_assignment(inst: Instance, rp: RepairProblem, x) -> RepairPlan:
    """Replay the chosen options as full duties and check them together."""
    x = np.asarray(x)
    rp.to_qubo().energy(x)  # reject malformed bitstrings before integer conversion
    x = x.astype(int)
    problems = []
    selected = set(np.flatnonzero(x))
    violated = [(k, l) for k, l in rp.conflicts if k in selected and l in selected]
    if violated:
        problems.append(f"pairwise repair conflicts selected: {violated}")
    assignment: dict[str, str] = {}
    for t, g in zip(rp.trips, rp.groups):
        chosen = [k for k in g if x[k]]
        if len(chosen) != 1:
            problems.append(f"trip {t}: {len(chosen)} options chosen")
            continue
        o = rp.options[chosen[0]]
        assignment[t] = o.target or "cancel"
    extra: dict[str, list[str]] = defaultdict(list)
    for t, target in assignment.items():
        if target != "cancel":
            extra[target].append(t)

    duties: dict[str, Optional[Column]] = {}
    total = 0.0
    occupied = defaultdict(list, {s: list(v) for s, v in rp.blocked.items()})
    for b, col in rp.base_plans.items():
        if b not in extra:
            for ch in col.charges:
                occupied[ch.station].append((ch.start_min, ch.end_min))
    for target, trips in sorted(extra.items()):
        if target.startswith("R"):
            plan = plan_duty(inst, trips, earliest_start=rp.event.at_min, blocked=occupied, col_id=target)
            if plan is not None:
                total += rp.settings["reserve_cost"] + plan.cost - plan.vehicle_cost
        else:
            col = rp.base_plans[target]
            plan = _replan(inst, col, col.trips + trips, rp.event, occupied, target)
            before = _replan(inst, col, col.trips, rp.event, occupied, target) or col
            if plan is not None:
                total += plan.cost - before.cost
        duties[target] = plan
        if plan is None:
            problems.append(f"{target}: new duty with {trips} is infeasible")
        else:
            for ch in plan.charges:
                occupied[ch.station].append((ch.start_min, ch.end_min))
    cancelled = [t for t, v in assignment.items() if v == "cancel"]
    total += rp.settings["cancel_cost"] * len(cancelled)
    capacity_ok = _capacity_ok(inst, [c for c in duties.values() if c is not None]
                               + [c for b, c in rp.base_plans.items() if b not in duties])
    if not capacity_ok:
        problems.append("charger capacity exceeded")
    one_each = len(assignment) == len(rp.trips)
    joint = one_each and not violated and all(d is not None for d in duties.values()) and capacity_ok
    return RepairPlan(feasible=one_each and joint, jointly_feasible=joint, capacity_ok=capacity_ok,
                      assignment=assignment, duties=duties, cancelled=cancelled,
                      extra_cost=total if joint else math.inf, problems=problems)


def cancel_all_assignment(inst: Instance, rp: RepairProblem) -> tuple[np.ndarray, RepairPlan]:
    """Return the guaranteed one-hot fallback that cancels every affected trip."""
    x = np.zeros(rp.n, dtype=int)
    for trip, group in zip(rp.trips, rp.groups):
        cancel = [k for k in group if rp.options[k].kind == "cancel"]
        if len(cancel) != 1:
            raise ValueError(f"trip {trip}: expected exactly one cancel option")
        x[cancel[0]] = 1
    plan = apply_assignment(inst, rp, x)
    if not plan.feasible:
        raise ValueError("cancel-all fallback is not feasible: " + "; ".join(plan.problems))
    return x, plan


def _capacity_ok(inst: Instance, duties: list[Column]) -> bool:
    cap: dict[str, int] = defaultdict(int)
    for c in inst.chargers:
        cap[c.station] += 1
    events: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for d in duties:
        for ch in d.charges:
            events[ch.station] += [(ch.start_min, 1), (ch.end_min, -1)]
    for s, ev in events.items():
        level = 0
        for _, delta in sorted(ev, key=lambda e: (e[0], e[1])):
            level += delta
            if level > cap[s]:
                return False
    return True

"""Classical baselines (package D).

Diesel scenarios of the assignment (no battery) are polynomial:

    S1  minimum number of buses            minimum path cover = trips - maximum matching
    S2  buses + deadhead kilometres         minimum-cost flow
    S3  deadhead km with at most v_max      minimum-cost flow with a capacity on the depot

All three are solved as the same flow model, written as an LP over the trip
connection graph (`Instance.arcs`). Its constraint matrix is totally unimodular,
so the LP optimum is integral; `solve_flow` checks that. S1 is also solved
independently by Hopcroft-Karp matching as a cross-check.

Electric (E-VSP, battery + charging) baselines:

    greedy_electric      first-fit dispatcher: each trip goes to the bus where it is
                         cheapest to insert (battery replayed by plan_duty), else a new bus
    battery_check        which diesel blocks are still feasible with a battery
    greedy_repair        disruption repair: cheapest option per affected trip, in time order
    exact_repair         true-cost optimum of a repair by replaying every one-hot assignment

    fleet = scenario2(inst)                 # FleetSolution: blocks, buses, deadhead km
    e = greedy_electric(inst)               # ElectricSolution: duties with charging
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .columns import Column
from .disruption import RepairPlan, RepairProblem, apply_assignment, plan_duty
from .instance import Instance


# --------------------------------------------------------------------------- diesel (flow)

@dataclass
class FleetSolution:
    method: str
    blocks: list[list[str]]                  # trip ids per bus, in time order
    vehicles: int
    deadhead_km: float                       # pull-out + between trips + pull-in
    cost: float                              # w_veh * vehicles + w_km * deadhead_km
    seconds: float = 0.0
    status: str = "optimal"
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"method": self.method, "vehicles": self.vehicles, "deadhead_km": round(self.deadhead_km, 2),
                "cost": round(self.cost, 2), "seconds": round(self.seconds, 3), "status": self.status, **self.extra}


def block_deadhead_km(inst: Instance, block: list[str]) -> float:
    tb, depot = inst.trip_by_id, inst.depot
    seq = [tb[t] for t in block]
    km = inst.dh_km(depot, seq[0].start) + inst.dh_km(seq[-1].end, depot)
    return km + sum(inst.dh_km(i.end, j.start) for i, j in zip(seq, seq[1:]))


def _fleet(inst: Instance, method: str, blocks: list[list[str]], w_veh: float, w_km: float,
           seconds: float = 0.0, **extra) -> FleetSolution:
    km = sum(block_deadhead_km(inst, b) for b in blocks)
    return FleetSolution(method=method, blocks=blocks, vehicles=len(blocks), deadhead_km=km,
                         cost=w_veh * len(blocks) + w_km * km, seconds=seconds, extra=dict(extra))


def solve_flow(inst: Instance, w_veh: float, w_km: float, vmax: Optional[int] = None,
               min_layover: Optional[float] = None, method: str = "min-cost flow") -> Optional[FleetSolution]:
    """Minimum-cost flow: every trip has one predecessor (another trip or the depot) and one successor.

    Variables: x_ij for every admissible connection, o_j pull-out before trip j, d_i pull-in after
    trip i. Cost: w_veh per pull-out plus w_km per deadhead km. `vmax` bounds sum(o) (scenario 3).
    Returns None if no schedule with at most `vmax` buses exists.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    t0 = time.time()
    trips = sorted(inst.trips, key=lambda t: (t.dep_min, t.id))
    pos = {t.id: k for k, t in enumerate(trips)}
    n = len(trips)
    arcs = inst.arcs(min_layover=min_layover)
    m = len(arcs)
    tb, depot = inst.trip_by_id, inst.depot
    cost = np.empty(m + 2 * n)
    cost[:m] = [w_km * inst.dh_km(tb[i].end, tb[j].start) for i, j in arcs]
    cost[m:m + n] = [w_veh + w_km * inst.dh_km(depot, t.start) for t in trips]
    cost[m + n:] = [w_km * inst.dh_km(t.end, depot) for t in trips]
    rows, cols = [], []
    for k, (i, j) in enumerate(arcs):
        rows += [pos[j], n + pos[i]]               # in-row of j, out-row of i
        cols += [k, k]
    for k in range(n):
        rows += [k, n + k]
        cols += [m + k, m + n + k]
    a_eq = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(2 * n, m + 2 * n)).tocsr()
    a_ub = b_ub = None
    if vmax is not None:
        a_ub = coo_matrix((np.ones(n), (np.zeros(n, dtype=int), np.arange(m, m + n))), shape=(1, m + 2 * n)).tocsr()
        b_ub = [vmax]
    res = linprog(cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=np.ones(2 * n), bounds=(0, 1), method="highs")
    if res.status == 2:                            # infeasible: fewer buses than the minimum fleet
        return None
    if res.status != 0:
        raise RuntimeError(f"flow LP failed: {res.message}")
    x = res.x
    if np.max(np.abs(x - np.round(x))) > 1e-6:     # cannot happen for a totally unimodular matrix
        raise RuntimeError("flow LP returned a fractional solution")
    x = np.round(x).astype(int)
    succ = {arcs[k][0]: arcs[k][1] for k in np.flatnonzero(x[:m])}
    blocks = []
    for k in np.flatnonzero(x[m:m + n]):           # follow every pull-out to its pull-in
        cur, block = trips[k].id, []
        while cur is not None:
            block.append(cur)
            cur = succ.get(cur)
        blocks.append(block)
    return _fleet(inst, method, blocks, w_veh, w_km, time.time() - t0, arcs=m,
                  **({} if vmax is None else {"vmax": vmax}))


def min_fleet_matching(inst: Instance, min_layover: Optional[float] = None) -> int:
    """Scenario 1 by Hopcroft-Karp: minimum path cover of the trip DAG = n - maximum matching."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    pos = {t.id: k for k, t in enumerate(inst.trips)}
    arcs = inst.arcs(min_layover=min_layover)
    n = len(inst.trips)
    if not arcs:
        return n
    g = coo_matrix((np.ones(len(arcs)), ([pos[i] for i, _ in arcs], [pos[j] for _, j in arcs])), shape=(n, n)).tocsr()
    match = maximum_bipartite_matching(g, perm_type="column")
    return n - int((match >= 0).sum())


def scenario1(inst: Instance) -> FleetSolution:
    """Minimum number of buses (deadhead km only breaks ties)."""
    s = solve_flow(inst, w_veh=1.0, w_km=1e-6, method="S1 min buses (flow)")
    s.cost = float(s.vehicles)
    s.extra["matching_fleet"] = min_fleet_matching(inst)
    return s


def scenario2(inst: Instance, w_veh: Optional[float] = None, w_km: Optional[float] = None) -> FleetSolution:
    """Buses + deadhead km with the instance's cost weights (vehicle 500, 1.5 per km by default)."""
    a = inst.assumptions
    return solve_flow(inst, a["vehicle_cost"] if w_veh is None else w_veh,
                      a["deadhead_cost_per_km"] if w_km is None else w_km, method="S2 buses + deadhead (flow)")


def scenario3(inst: Instance, vmax: int) -> Optional[FleetSolution]:
    """Least deadhead km with at most `vmax` buses (None if vmax is below the minimum fleet)."""
    return solve_flow(inst, w_veh=0.0, w_km=1.0, vmax=vmax, method=f"S3 deadhead, <= {vmax} buses (flow)")


def baseline_blocks(inst: Instance) -> FleetSolution:
    """The operator's real GTFS blocks, measured with our deadhead estimates."""
    a = inst.assumptions
    blocks = [list(b["trips"]) for b in inst.baseline_blocks]
    rep = inst.check_baseline()
    return _fleet(inst, "real GTFS blocks", blocks, a["vehicle_cost"], a["deadhead_cost_per_km"],
                  infeasible_pairs=rep["pairs_fail"])


def is_valid_cover(inst: Instance, blocks: list[list[str]], min_layover: Optional[float] = None) -> bool:
    """Every trip exactly once and every consecutive pair time-feasible."""
    tb = inst.trip_by_id
    seen = [t for b in blocks for t in b]
    if sorted(seen) != sorted(t.id for t in inst.trips):
        return False
    return all(inst.can_follow(tb[i], tb[j], min_layover) for b in blocks for i, j in zip(b, b[1:]))


# --------------------------------------------------------------------------- electric

@dataclass
class ElectricSolution:
    method: str
    duties: list[Column]
    seconds: float
    status: str = "ok"

    @property
    def vehicles(self) -> int:
        return len(self.duties)

    @property
    def cost(self) -> float:
        return sum(c.cost for c in self.duties)

    def summary(self) -> dict:
        return {"method": self.method, "vehicles": self.vehicles, "cost": round(self.cost, 2),
                "deadhead_km": round(sum(c.deadhead_km for c in self.duties), 2),
                "charging_stops": sum(len(c.charges) for c in self.duties),
                "grid_kwh": round(sum(c.grid_kwh_day for c in self.duties), 1),
                "seconds": round(self.seconds, 3), "status": self.status}


def _full_windows(inst: Instance, duties: list[Column]) -> dict[str, list[tuple[float, float]]]:
    """Time windows in which every connector of a station is already taken by `duties`."""
    cap: dict[str, int] = {}
    for c in inst.chargers:
        cap[c.station] = cap.get(c.station, 0) + 1
    events: dict[str, list[tuple[float, int]]] = {}
    for d in duties:
        for ch in d.charges:
            events.setdefault(ch.station, []).extend([(ch.start_min, 1), (ch.end_min, -1)])
    out: dict[str, list[tuple[float, float]]] = {}
    for st, ev in events.items():
        level, start = 0, None
        for t, delta in sorted(ev, key=lambda e: (e[0], e[1])):
            level += delta
            if level >= cap.get(st, 1) and start is None:
                start = t
            elif level < cap.get(st, 1) and start is not None:
                out.setdefault(st, []).append((start, t))
                start = None
    return out


def _plan(inst: Instance, trips: list[str], col_id: str,
          blocked: Optional[dict[str, list[tuple[float, float]]]] = None) -> Optional[Column]:
    """plan_duty restricted to the column model of package B: charging only between two trips."""
    plan = plan_duty(inst, trips, col_id=col_id, blocked=blocked)
    if plan is None:
        return None
    last = max(inst.trip_by_id[t].arr_min for t in trips)
    return None if any(ch.start_min >= last for ch in plan.charges) else plan


def greedy_electric(inst: Instance) -> ElectricSolution:
    """First-fit dispatcher: trips in departure order, each to the bus with the cheapest feasible insertion.

    Battery, deadheads and charging stops are replayed by `plan_duty` (greedy charging at the
    nearest station when SoC < 60 % or the next trip is out of reach), restricted to the column
    model of package B so that its duties are valid columns and the cost is comparable with the
    MILP. Stations whose connectors are all taken by other buses are blocked, so the schedule
    respects charger capacity. A new bus is opened only when no existing bus can take the trip. Typical of
    dispatching rules used in practice; no optimality guarantee.
    """
    t0 = time.time()
    buses: list[list[str]] = []
    plans: list[Column] = []
    for t in sorted(inst.trips, key=lambda t: (t.dep_min, t.id)):
        best = None
        for b, trips in enumerate(buses):
            if inst.trip_by_id[trips[-1]].arr_min > t.dep_min:
                continue
            plan = _plan(inst, trips + [t.id], f"g{b}", _full_windows(inst, plans[:b] + plans[b + 1:]))
            if plan is not None and (best is None or plan.cost - plans[b].cost < best[0]):
                best = (plan.cost - plans[b].cost, b, plan)
        if best is not None:
            _, b, plan = best
            buses[b].append(t.id)
            plans[b] = plan
            continue
        plan = _plan(inst, [t.id], f"g{len(buses)}", _full_windows(inst, plans))
        if plan is None:
            return ElectricSolution("greedy dispatcher", plans, time.time() - t0, status=f"trip {t.id} infeasible alone")
        buses.append([t.id])
        plans.append(plan)
    return ElectricSolution("greedy dispatcher", plans, time.time() - t0)


def battery_check(inst: Instance, blocks: list[list[str]]) -> dict:
    """Replay diesel blocks with the battery: how many stay feasible (charging allowed between trips)."""
    ok = [_plan(inst, b, "check") is not None for b in blocks]
    return {"blocks": len(blocks), "battery_feasible": int(sum(ok)),
            "infeasible_blocks": [k for k, f in enumerate(ok) if not f]}


# --------------------------------------------------------------------------- repair

def _cancel_all(rp: RepairProblem) -> np.ndarray:
    x = np.zeros(rp.n, dtype=int)
    for g in rp.groups:
        x[[k for k in g if rp.options[k].kind == "cancel"][0]] = 1
    return x


def greedy_repair(inst: Instance, rp: RepairProblem) -> tuple[np.ndarray, RepairPlan]:
    """Dispatcher rule: affected trips in time order, each takes its cheapest option that still replays."""
    x = _cancel_all(rp)
    for g in rp.groups:                            # groups are in departure order
        for k in sorted(g, key=lambda k: rp.options[k].cost):
            trial = x.copy()
            trial[g] = 0
            trial[k] = 1
            if apply_assignment(inst, rp, trial).feasible:
                x = trial
                break
    return x, apply_assignment(inst, rp, x)


def exact_repair(inst: Instance, rp: RepairProblem, max_states: int = 20_000) -> Optional[tuple[np.ndarray, RepairPlan]]:
    """True-cost optimum: replay every one-hot assignment (reserve activation counted once per bus).

    None if the one-hot subspace is larger than `max_states`.
    """
    from .xy_qaoa import OneHotSpace

    if rp.subspace_size > max_states:
        return None
    space = OneHotSpace.from_problem(rp)
    best = None
    for s in range(space.size):
        if not space.feasible[s]:
            continue
        x = space.bits(s)
        plan = apply_assignment(inst, rp, x)
        if plan.feasible and (best is None or plan.extra_cost < best[1].extra_cost - 1e-9):
            best = (x, plan)
    return best

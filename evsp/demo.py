"""One-call pipeline behind the demo app and the pitch figures (package G).

    data = load_demo_data()                               # instance + columns + optimal schedule
    rep = solve_disruption(data, Disruption.bus_failure(data.schedule, "k4", 6 * 60))
    rep.before, rep.after, rep.moved, rep.cancelled       # schedules for the Gantt charts
    rep.rows                                              # solver comparison (P(optimum), P(feasible))
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import load_instance
from .columns import Column, ColumnSet, load_columns
from .disruption import (Disruption, RepairPlan, RepairProblem, Schedule, apply_assignment, build_repair_problem,
                         cancel_all_assignment)
from .qaoa import circuit_metrics, run_qaoa, shots_for_confidence
from .qubo import build_qubo, variant_weights
from .solvers import Landscape, simulated_annealing
from .xy_qaoa import XYQAOA, OneHotSpace, best_valid_state, run_xy_qaoa, xy_qaoa_circuit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLUMNS = ROOT / "columns" / "kcm_small_b20_bat200.json"
DEFAULT_INSTANCE = ROOT / "instances" / "kcm_small.json"


@dataclass
class DemoData:
    inst: object
    cs: ColumnSet
    schedule: Schedule
    columns_path: Path


def load_demo_data(columns: Path = DEFAULT_COLUMNS, instance: Path = DEFAULT_INSTANCE) -> DemoData:
    cs = load_columns(columns)
    inst = load_instance(instance)
    inst.assumptions.update(cs.assumptions)          # same battery / costs as the columns
    return DemoData(inst=inst, cs=cs, schedule=Schedule.from_columns(cs), columns_path=Path(columns))


def peak_schedules(data: DemoData) -> dict[str, dict[str, Column]]:
    """Exact E4 schedules for the same column pool with and without peak penalty."""
    from .solvers import index_to_bits

    result = {}
    for label, weight in (("R=0", 0.0), ("R>0", 1.0)):
        q = build_qubo(data.cs, variant_weights("E4", data.cs.assumptions, R=weight))
        land = Landscape.from_qubo(q)
        if not land.feasible.any():
            raise ValueError("no feasible schedule in the selected column pool")
        x = index_to_bits(int(np.argmin(np.where(land.feasible, land.energies, np.inf))), q.n)
        result[label] = {data.cs.columns[i].id: data.cs.columns[i]
                         for i, selected in enumerate(x) if selected}
    return result


@dataclass
class RepairReport:
    event: Disruption
    problem: RepairProblem
    space: OneHotSpace
    plan: RepairPlan                     # cheapest assignment that passes the full replay
    xy_plan: RepairPlan                  # most likely XY-QAOA answer, replayed (cancel-all if it fails)
    xy_is_optimal: bool
    before: dict[str, Column]
    after: dict[str, Optional[Column]]
    moved: dict[str, str]
    cancelled: list[str]
    rows: list[dict]                     # method comparison
    circuits: dict
    seconds: dict = field(default_factory=dict)
    penalty_source: str = ""             # "computed", "precomputed <file>" or "skipped: ..."
    valid_rank: Optional[int] = 0        # energy rank of the replayed optimum (None = cancel-all fallback)


def solve_disruption(data: DemoData, event: Disruption, reserves: int = 3, max_bus_options: int = 3,
                     p_max: int = 3, penalty_qaoa: str = "auto", penalty_max_n: int = 16,
                     sa_reads: int = 200, precomputed: Optional[Path] = None, seed: int = 0,
                     max_states: int = 100_000) -> RepairReport:
    """Exact optimum, SA, XY-mixer QAOA and (if affordable) penalty QAOA for one disruption.

    penalty_qaoa: "auto" runs the X-mixer QAOA when n <= penalty_max_n, otherwise uses a matching
    precomputed result from scripts/run_repair.py if available; "always" runs it up to 22 qubits;
    "never" skips it. Raises ValueError when the one-hot subspace has more than max_states
    states (the live demo would take minutes; use scripts/run_repair.py instead).
    """
    inst, sched = data.inst, data.schedule
    t0 = time.time()
    rp = build_repair_problem(inst, sched, event, max_bus_options=max_bus_options, reserves=reserves)
    if rp.subspace_size > max_states:
        raise ValueError(f"{rp.n} qubits, {rp.subspace_size} one-hot states: too large for the live demo "
                         f"(limit {max_states}); lower reserves / options or use scripts/run_repair.py")
    space = OneHotSpace.from_problem(rp)
    valid = best_valid_state(inst, rp, space)
    if valid is None:
        _, plan = cancel_all_assignment(inst, rp)
        valid_rank = None
    else:
        _, plan, valid_rank = valid
    seconds = {"build": time.time() - t0}
    rows = []

    q = rp.to_qubo()
    t0 = time.time()
    sa = simulated_annealing(q, reads=sa_reads, sweeps=500, seed=seed)
    tol = 1e-6 * max(1.0, abs(space.optimum_energy))
    rows.append({"method": "simulated annealing", "p_optimal": float(np.mean(sa.energies <= space.optimum_energy + tol)),
                 "p_feasible": None, "p": None})
    seconds["sa"] = time.time() - t0

    t0 = time.time()
    xy = run_xy_qaoa(space, p_max=p_max)
    bxy = max(xy.layers, key=lambda L: L.p_optimal)
    xy_probs = XYQAOA(space).probabilities(bxy.gammas, bxy.betas)
    seconds["xy_qaoa"] = time.time() - t0

    penalty_row, source = None, ""
    if penalty_qaoa != "never":
        limit = 22 if penalty_qaoa == "always" else penalty_max_n
        if rp.n <= limit:
            t0 = time.time()
            land = Landscape.from_qubo(q)
            pq = run_qaoa(q, land, p_max=p_max, shots=1000, seed=seed)
            bp = max(pq.layers, key=lambda L: L.p_optimal)
            penalty_row = {"method": "QAOA, X mixer + penalties", "p_optimal": bp.p_optimal,
                           "p_feasible": bp.p_feasible, "p": bp.p}
            seconds["penalty_qaoa"] = time.time() - t0
            source = "computed"
        else:
            penalty_row, source = _precomputed_penalty(event, rp, precomputed)
    rows.insert(0, penalty_row or {"method": "QAOA, X mixer + penalties", "p_optimal": None, "p_feasible": None,
                                   "p": None})
    if penalty_row is None and not source:
        source = "skipped"
    rows.insert(1, {"method": "QAOA, XY mixer", "p_optimal": bxy.p_optimal, "p_feasible": bxy.p_feasible,
                    "p": bxy.p})
    for r in rows:
        r["shots_95"] = shots_for_confidence(r["p_optimal"]) if r["p_optimal"] is not None else None

    top = int(np.argmax(xy_probs))
    xy_plan = apply_assignment(inst, rp, space.bits(top))
    if not xy_plan.feasible:
        _, xy_plan = cancel_all_assignment(inst, rp)
    qc = xy_qaoa_circuit(rp, bxy.gammas, bxy.betas, space)
    circuits = {"xy_all": circuit_metrics(qc, "all"), "xy_heavy_hex": circuit_metrics(qc, "heavy-hex")}

    before = dict(sched.duties)
    after: dict[str, Optional[Column]] = {b: c for b, c in rp.base_plans.items() if not b.endswith("*")}
    failed_prefix = {b[:-1]: c for b, c in rp.base_plans.items() if b.endswith("*")}
    tb = inst.trip_by_id
    for b, c in failed_prefix.items():                 # failed bus keeps the trips it finished before the failure
        done = [t for t in c.trips if tb[t].arr_min <= event.at_min]
        if done:                                       # (base_plans already cut its charging at the failure)
            after[b] = Column(**{**c.__dict__, "trips": done, "end_min": event.at_min})
    after.update(plan.duties)
    owner = {t: b for b, c in before.items() for t in c.trips}
    moved = {t: v for t, v in plan.assignment.items() if v != "cancel" and v != owner.get(t)}
    return RepairReport(event=event, problem=rp, space=space, plan=plan, xy_plan=xy_plan,
                        xy_is_optimal=bool(xy_plan.feasible and space.optimum_mask()[top] and valid_rank == 0),
                        before=before, after=after, moved=moved,
                        cancelled=list(plan.cancelled), rows=rows, circuits=circuits, seconds=seconds,
                        penalty_source=source, valid_rank=valid_rank)


def _precomputed_penalty(event: Disruption, rp: RepairProblem, path: Optional[Path]):
    """Penalty-QAOA numbers from results/repair_<event>.json written by scripts/run_repair.py."""
    if path is None:
        if event.kind == "bus":
            tag = f"bus_{event.bus}_{int(event.at_min) // 60:02d}{int(event.at_min) % 60:02d}"
        else:
            a, b = int(event.at_min), int(event.until_min)
            tag = f"charger_{event.station}_{a // 60:02d}{a % 60:02d}-{b // 60:02d}{b % 60:02d}"
        path = ROOT / "results" / f"repair_{tag}.json"
    if not path.exists():
        return None, f"skipped: {rp.n} qubits is too slow live; run scripts/run_repair.py to precompute"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (raw.get("event") != event.__dict__ or raw.get("trips") != rp.trips
            or raw.get("n_qubits") != rp.n or raw.get("subspace_size") != rp.subspace_size
            or raw.get("options") != [o.__dict__ for o in rp.options]
            or raw.get("conflicts") != [list(pair) for pair in rp.conflicts]
            or "penalty_qaoa" not in raw):
        return None, f"skipped: {path.name} does not match this problem"
    best = max(raw["penalty_qaoa"]["layers"], key=lambda L: L["p_optimal"])
    return ({"method": "QAOA, X mixer + penalties", "p_optimal": best["p_optimal"], "p_feasible": best["p_feasible"],
             "p": best["p"]}, f"precomputed {path.name}")
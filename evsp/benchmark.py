"""Solution quality with a sample budget: QAOA vs simulated annealing (best of N samples).

QAOA and simulated annealing are both samplers: one QAOA shot or one SA run gives one
candidate repair. A dispatcher takes N candidates, replays them and keeps the cheapest.
For every method we build the distribution of the TRUE extra cost of one sample (each
repair is replayed with batteries and chargers; a sample that is not a valid repair
counts as the cancel-all fallback), then compute exactly for a budget of N samples

    P(best of N is the true optimum)  = 1 - (1 - p_opt)^N
    E[cost of the best of N]           from the cost distribution: P(min >= v) = P(X >= v)^N

    dists = sample_distributions(inst, rp, p_max=2, sa_reads=2000)
    table = best_of_n_table(dists, [1, 10, 100, 1000])

Distributions: XY-mixer QAOA and penalty QAOA (X mixer, n <= 18) from the exact
statevector, simulated annealing from `sa_reads` independent runs (empirical), and two
random baselines (uniform over valid one-hot assignments, uniform over all bitstrings).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .disruption import RepairProblem, apply_assignment, cancel_all_assignment
from .instance import Instance
from .xy_qaoa import OneHotSpace, XYQAOA, run_xy_qaoa


@dataclass
class SampleDistribution:
    method: str
    values: np.ndarray                 # true extra cost of each outcome (fallback for invalid samples)
    probs: np.ndarray                  # probability of each outcome
    p_valid: float                     # one sample is a valid repair
    p_optimal: float                   # one sample is a true-cost optimal repair
    seconds_per_sample: Optional[float] = None
    info: dict = field(default_factory=dict)


@dataclass
class RepairCosts:
    space: OneHotSpace
    costs: np.ndarray                  # true extra cost of every one-hot state (inf if it does not replay)
    optimum: float
    fallback: float                    # cancel every affected trip (always valid)
    index: dict                        # bit tuple -> one-hot state


def repair_costs(inst: Instance, rp: RepairProblem, max_states: int = 20_000) -> RepairCosts:
    """Replay every one-hot assignment once (true cost, reserve activation counted once per bus)."""
    space = OneHotSpace.from_problem(rp)
    if space.size > max_states:
        raise ValueError(f"{space.size} one-hot states (max {max_states})")
    costs = np.full(space.size, np.inf)
    index = {}
    for s in range(space.size):
        x = space.bits(s)
        index[tuple(int(v) for v in x)] = s
        if space.feasible[s]:
            plan = apply_assignment(inst, rp, x)
            if plan.feasible:
                costs[s] = plan.extra_cost
    _, fallback = cancel_all_assignment(inst, rp)
    return RepairCosts(space=space, costs=costs, optimum=float(costs.min()), fallback=float(fallback.extra_cost),
                       index=index)


def _distribution(method: str, rc: RepairCosts, state_probs: np.ndarray, invalid_mass: float,
                  **kw) -> SampleDistribution:
    """state_probs: probability of every one-hot state; invalid_mass: probability of anything else."""
    valid = np.isfinite(rc.costs)
    values = np.concatenate([rc.costs[valid], [rc.fallback]])
    probs = np.concatenate([state_probs[valid], [invalid_mass + state_probs[~valid].sum()]])
    tol = 1e-6 * max(1.0, abs(rc.optimum))
    p_opt = float(state_probs[valid & (rc.costs <= rc.optimum + tol)].sum())
    return SampleDistribution(method=method, values=values, probs=probs / probs.sum(),
                              p_valid=float(state_probs[valid].sum()), p_optimal=p_opt, **kw)


def best_of_n(dist: SampleDistribution, n: int, optimum: float) -> dict:
    """Exact statistics of the cheapest of n independent samples."""
    order = np.argsort(dist.values, kind="stable")
    v, p = dist.values[order], dist.probs[order]
    tail = np.concatenate([np.cumsum(p[::-1])[::-1], [0.0]])      # P(X >= v_k)
    p_min = np.clip(tail[:-1], 0, 1) ** n - np.clip(tail[1:], 0, 1) ** n
    expected = float(p_min @ v)
    return {"method": dist.method, "n": n, "expected_best_cost": expected,
            "expected_gap": max(0.0, expected - optimum) if expected - optimum > -1e-6 else expected - optimum, "p_found_optimum": 1.0 - (1.0 - dist.p_optimal) ** n,
            "p_any_valid": 1.0 - (1.0 - dist.p_valid) ** n}


def sample_distributions(inst: Instance, rp: RepairProblem, p_max: int = 2, sa_reads: int = 2000,
                         sa_sweeps: int = 500, penalty_max_n: int = 18, seed: int = 0,
                         rc: Optional[RepairCosts] = None) -> tuple[RepairCosts, list[SampleDistribution]]:
    from .qaoa import FastQAOA, run_qaoa
    from .solvers import Landscape, simulated_annealing

    rc = rc or repair_costs(inst, rp)
    space, n = rc.space, rp.n
    dists = []

    xy = run_xy_qaoa(space, p_max=p_max)
    layer = xy.layers[-1]
    probs = XYQAOA(space).probabilities(layer.gammas, layer.betas)
    dists.append(_distribution(f"XY-mixer QAOA (p={layer.p})", rc, probs, 0.0, info={"p": layer.p}))

    q = rp.to_qubo()
    if n <= penalty_max_n:
        land = Landscape.from_qubo(q)
        pen = run_qaoa(q, land, p_max=p_max, shots=1000, seed=seed).layers[-1]
        full = FastQAOA(land).probabilities(pen.gammas, pen.betas)
        idx = np.array([space.full_index(s) for s in range(space.size)])
        sp = full[idx]
        dists.append(_distribution(f"penalty QAOA, X mixer (p={pen.p})", rc, sp, float(1.0 - sp.sum()),
                                   info={"p": pen.p}))

    sa = simulated_annealing(q, reads=sa_reads, sweeps=sa_sweeps, seed=seed)
    counts = np.zeros(space.size)
    other = 0
    for row in sa.states:
        s = rc.index.get(tuple(int(v) for v in row))
        if s is None:
            other += 1
        else:
            counts[s] += 1
    dists.append(_distribution(f"simulated annealing ({sa_sweeps} sweeps)", rc, counts / sa_reads, other / sa_reads,
                               seconds_per_sample=sa.seconds / sa_reads, info={"reads": sa_reads}))

    dists.append(_distribution("random valid assignment", rc, np.full(space.size, 1.0 / space.size), 0.0))
    dists.append(_distribution("random bitstring", rc, np.full(space.size, 2.0 ** -n), float(1 - space.size * 2.0 ** -n)))
    return rc, dists


def best_of_n_table(dists: list[SampleDistribution], budgets: list[int], optimum: float) -> list[dict]:
    return [best_of_n(d, n, optimum) for d in dists for n in budgets]


def run_quality_benchmark(inst: Instance, rp: RepairProblem, budgets=(1, 10, 100, 1000), **kw) -> dict:
    """Everything for one repair problem, JSON-ready."""
    from .classical_baselines import greedy_repair

    t0 = time.time()
    rc, dists = sample_distributions(inst, rp, **kw)
    _, greedy = greedy_repair(inst, rp)
    return {"qubits": rp.n, "valid_states": rc.space.size, "true_optimum": rc.optimum, "fallback": rc.fallback,
            "greedy": greedy.extra_cost,
            "per_sample": [{"method": d.method, "p_valid": d.p_valid, "p_optimal": d.p_optimal,
                            "seconds_per_sample": d.seconds_per_sample, **d.info} for d in dists],
            "best_of_n": best_of_n_table(dists, list(budgets), rc.optimum), "seconds": time.time() - t0}

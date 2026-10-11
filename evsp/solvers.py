"""Shared tools for packages D/E: exact landscape, simulated annealing, tabu, repair.

Bit order everywhere is little-endian, the same as Qiskit: state index
    idx = sum_k x_k * 2^k,
so variable (qubit) k is bit k of the index.

    land = Landscape.from_qubo(q)          # energy + feasibility of all 2^n states (n <= ~24)
    land.optimum_energy, land.p_optimal(probs), land.p_feasible(probs)
    sa = simulated_annealing(q, reads=100, seed=0)
    x_fixed = repair(q, x)                 # make an infeasible bitstring feasible, if possible
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .qubo import QUBO

ENERGY_TOL = 1e-6


def index_to_bits(idx: np.ndarray | int, n: int) -> np.ndarray:
    """State index (or array of indices) -> bit array [.., n], bit k = variable k."""
    idx = np.asarray(idx, dtype=np.int64)
    return ((idx[..., None] >> np.arange(n, dtype=np.int64)) & 1).astype(np.int8)


def bits_to_index(x) -> int:
    return int(sum(int(b) << k for k, b in enumerate(np.asarray(x).astype(int))))


# --------------------------------------------------------------------------- landscape

@dataclass
class Landscape:
    """Energy and feasibility of every bitstring; exact reference for small QUBOs."""

    n: int
    energies: np.ndarray            # [2^n] true QUBO energy (offset included)
    feasible: np.ndarray            # [2^n] bool, every trip once, no conflict, fleet size (E3)
    optimum_energy: float
    optimum_indices: np.ndarray     # all states with the minimum energy
    best_feasible_energy: float

    @classmethod
    def from_qubo(cls, q: QUBO, max_n: int = 24, chunk: int = 1 << 16) -> "Landscape":
        n = q.n
        if n > max_n:
            raise ValueError(f"{n} variables: the full landscape needs 2^{n} states (max {max_n})")
        m = q.matrix()
        total = 1 << n
        energies = np.empty(total)
        feasible = np.empty(total, dtype=bool)
        cover = np.zeros((n, len(q._cover)), dtype=np.int16)        # variable x trip incidence
        for i, cols in enumerate(q._cover):
            cover[cols, i] = 1
        conflicts = np.array(q._conflicts, dtype=np.int64).reshape(-1, 2)
        fleet = q.weights.V if (q.weights.V is not None and q.meta.get("M")) else None
        for start in range(0, total, chunk):
            ids = np.arange(start, min(total, start + chunk), dtype=np.int64)
            x = index_to_bits(ids, n).astype(float)
            energies[start:start + len(ids)] = np.einsum("bi,ij,bj->b", x, m, x) + q.offset
            ok = np.all(x @ cover == 1, axis=1)
            if len(conflicts):
                ok &= ~np.any((x[:, conflicts[:, 0]] > 0) & (x[:, conflicts[:, 1]] > 0), axis=1)
            if fleet is not None:
                ok &= x.sum(axis=1) == fleet
            feasible[start:start + len(ids)] = ok
        e_min = float(energies.min())
        opt = np.flatnonzero(energies <= e_min + ENERGY_TOL * max(1.0, abs(e_min)))
        best_feas = float(energies[feasible].min()) if feasible.any() else math.inf
        return cls(n=n, energies=energies, feasible=feasible, optimum_energy=e_min,
                   optimum_indices=opt, best_feasible_energy=best_feas)

    # ---------------------------------------------------------------- distribution metrics
    def is_optimal(self, idx) -> np.ndarray:
        e = self.energies[np.asarray(idx)]
        return e <= self.optimum_energy + ENERGY_TOL * max(1.0, abs(self.optimum_energy))

    def p_optimal(self, probs: np.ndarray) -> float:
        return float(probs[self.optimum_indices].sum())

    def p_feasible(self, probs: np.ndarray) -> float:
        return float(probs[self.feasible].sum())

    def expected_energy(self, probs: np.ndarray) -> float:
        return float(probs @ self.energies)

    def approximation_ratio(self, probs: np.ndarray) -> float:
        """(E_max - <E>) / (E_max - E_min): 1 = always optimal, 0 = always the worst state."""
        e_max = float(self.energies.max())
        span = e_max - self.optimum_energy
        return float((e_max - self.expected_energy(probs)) / span) if span > 0 else 1.0

    def uniform_p_optimal(self) -> float:
        """P(optimum) of random guessing, the baseline QAOA has to beat."""
        return len(self.optimum_indices) / len(self.energies)

    def uniform_p_feasible(self) -> float:
        return float(self.feasible.mean())


# --------------------------------------------------------------------------- simulated annealing

@dataclass
class SAResult:
    best_x: np.ndarray
    best_energy: float
    energies: np.ndarray            # final energy of every read
    p_optimal: Optional[float]      # share of reads ending in the optimum (needs a landscape)
    p_feasible: Optional[float]
    seconds: float
    params: dict = field(default_factory=dict)
    states: Optional[np.ndarray] = None   # [reads, n] final bitstring of every read


def simulated_annealing(q: QUBO, reads: int = 100, sweeps: int = 1000, t_start: Optional[float] = None,
                        t_end: Optional[float] = None, seed: int = 0,
                        landscape: Optional[Landscape] = None) -> SAResult:
    """Single-flip Metropolis annealing on the same QUBO, run for `reads` independent chains.

    Temperatures default to the coefficient scale: t_start = max |coef|, t_end = 1e-3 * min |coef|.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    n = q.n
    m = q.matrix()
    sym = m + m.T - np.diag(np.diag(m))          # off-diagonal symmetric couplings, diagonal = linear
    lin = np.diag(m).copy()
    off = sym - np.diag(np.diag(sym))
    coefs = np.abs(np.concatenate([lin, off[np.triu_indices(n, 1)]]))
    coefs = coefs[coefs > 1e-12]
    t_start = t_start if t_start is not None else float(coefs.max())
    t_end = t_end if t_end is not None else float(coefs.min()) * 1e-3
    temps = np.geomspace(t_start, t_end, sweeps)

    x = rng.integers(0, 2, size=(reads, n)).astype(float)
    field_ = x @ off + lin                       # local field: dE for flipping 0 -> 1
    for t in temps:
        for k in rng.permutation(n):
            de = (1 - 2 * x[:, k]) * field_[:, k]
            accept = (de <= 0) | (rng.random(reads) < np.exp(-np.clip(de, 0, None) / t))
            flip = np.flatnonzero(accept)
            if flip.size:
                delta = 1 - 2 * x[flip, k]
                x[flip, k] += delta
                field_[flip] += delta[:, None] * off[k]
    finals = np.einsum("bi,ij,bj->b", x, m, x) + q.offset
    best = int(np.argmin(finals))
    p_opt = p_feas = None
    if landscape is not None:
        idx = np.array([bits_to_index(row) for row in x.astype(int)])
        p_opt = float(np.mean(landscape.is_optimal(idx)))
        p_feas = float(np.mean(landscape.feasible[idx]))
    return SAResult(best_x=x[best].astype(int), best_energy=float(finals[best]), energies=finals,
                    p_optimal=p_opt, p_feasible=p_feas, seconds=time.time() - t0,
                    params={"reads": reads, "sweeps": sweeps, "t_start": t_start, "t_end": t_end, "seed": seed},
                    states=x.astype(np.int8))


@dataclass
class TabuResult:
    best_x: np.ndarray
    best_energy: float
    seconds: float
    params: dict


def tabu_search(q: QUBO, reads: int = 20, steps: int = 500, tenure: int = 5,
                seed: int = 0) -> TabuResult:
    """Multi-start single-flip tabu search, retaining the best visited state.

    A tabu move is allowed if it improves the chain's best energy (aspiration).
    Tenure is capped at n - 1 so there is always an admissible move. This solver
    uses only QUBO coefficients; the exact landscape is not an optimization oracle.
    """
    if q.n < 1 or reads < 1 or steps < 1 or tenure < 1:
        raise ValueError("tabu search needs variables and positive reads, steps and tenure")
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    tenure = min(tenure, q.n - 1)
    matrix = q.matrix()
    linear = np.diag(matrix)
    off = matrix + matrix.T - 2 * np.diag(linear)
    x = rng.integers(0, 2, size=(reads, q.n))
    best_x = x.copy()
    best_energy = q.energy(x)
    tabu_until = np.zeros_like(x)
    rows = np.arange(reads)
    for step in range(steps):
        energy = q.energy(x)
        delta = (1 - 2 * x) * (linear + x @ off)
        candidates = energy[:, None] + delta
        allowed = (tabu_until <= step) | (candidates < best_energy[:, None] - 1e-9)
        moves = np.argmin(np.where(allowed, candidates, np.inf), axis=1)
        x[rows, moves] = 1 - x[rows, moves]
        tabu_until[rows, moves] = step + tenure + 1
        energy = q.energy(x)
        improved = energy < best_energy
        best_x[improved] = x[improved]
        best_energy[improved] = energy[improved]
    winner = int(np.argmin(best_energy))
    return TabuResult(best_x[winner], float(best_energy[winner]), time.perf_counter() - t0,
                      {"reads": reads, "steps": steps, "tenure": tenure, "seed": seed})


# --------------------------------------------------------------------------- repair

def repair(q: QUBO, x) -> Optional[np.ndarray]:
    """Greedy classical repair of a bitstring; returns a feasible bitstring or None.

    1. drop columns while a trip is covered twice or a conflict is violated
       (the column involved in most violations first, the most expensive on ties),
    2. cover missing trips with columns that only cover missing trips and add no
       conflict, cheapest per covered trip first.
    Fleet size (E3) is not repaired. Results after repair are reported separately.
    """
    x = np.asarray(x).astype(int).copy()
    n = q.n
    covers = [set() for _ in range(n)]
    for i, cols in enumerate(q._cover):
        for k in cols:
            covers[k].add(i)
    conflict_of = [set() for _ in range(n)]
    for a, b in q._conflicts:
        conflict_of[a].add(b)
        conflict_of[b].add(a)
    costs = q._costs if q._costs is not None else q.linear

    def violations(k: int) -> int:
        count = sum(1 for i in covers[k] if sum(x[j] for j in q._cover[i]) > 1)
        return count + sum(1 for j in conflict_of[k] if x[j])

    while True:
        bad = [(violations(k), costs[k], k) for k in range(n) if x[k] and violations(k)]
        if not bad:
            break
        x[max(bad)[2]] = 0
    missing = {i for i, cols in enumerate(q._cover) if not any(x[k] for k in cols)}
    while missing:
        options = [(costs[k] / len(covers[k]), k) for k in range(n)
                   if not x[k] and covers[k] and covers[k] <= missing
                   and not any(x[j] for j in conflict_of[k])]
        if not options:
            return None
        k = min(options)[1]
        x[k] = 1
        missing -= covers[k]
    return x

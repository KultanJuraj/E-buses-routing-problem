"""QAOA with an XY mixer for one-hot problems (package F).

Every affected trip is a group of qubits with exactly one 1 ("one-hot"). The XY
mixer exp(-i beta (XX + YY) / 2) on pairs inside a group only moves the 1 inside
the group, and the start state is a product of W states, so the circuit never
leaves the feasible one-hot subspace: no one-hot penalty is needed.

That subspace has prod_g m_g states (m_g options of trip g) instead of 2^n, so
`XYQAOA` simulates it exactly even when n is far beyond a full statevector
(40 qubits with 135 000 one-hot states in the charger-outage scenario).
`xy_qaoa_circuit` is the same ansatz in Qiskit (W states, RZ, RZZ,
XXPlusYYGate); tests check it against the simulator.

    sub = OneHotSpace.from_problem(rp)          # all one-hot assignments + their energies
    res = run_xy_qaoa(sub, p_max=3)
    qc = xy_qaoa_circuit(rp, res.layers[-1].gammas, res.layers[-1].betas, sub)
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
import numpy as np

from .disruption import RepairProblem
from .qaoa import _interp, shots_for_confidence

ENERGY_TOL = 1e-6


# --------------------------------------------------------------------------- feasible subspace

@dataclass
class OneHotSpace:
    """Every assignment with exactly one option per trip, in mixed-radix order."""

    groups: list[list[int]]
    sizes: np.ndarray                 # options per group
    strides: np.ndarray               # state = sum choice_g * stride_g
    choices: np.ndarray               # [S, G] chosen option position per group
    energies: np.ndarray              # [S] QUBO energy (one-hot penalty is 0 here)
    feasible: np.ndarray              # [S] no pairwise conflict violated
    n_qubits: int

    @classmethod
    def from_problem(cls, rp: RepairProblem, max_states: int = 5_000_000) -> "OneHotSpace":
        sizes = np.array([len(g) for g in rp.groups], dtype=np.int64)
        total = int(np.prod(sizes))
        if total > max_states:
            raise ValueError(f"one-hot subspace has {total} states (max {max_states})")
        strides = np.ones(len(sizes), dtype=np.int64)
        for g in range(len(sizes) - 2, -1, -1):
            strides[g] = strides[g + 1] * sizes[g + 1]
        idx = np.arange(total, dtype=np.int64)
        choices = (idx[:, None] // strides[None, :]) % sizes[None, :]
        var = np.array([rp.groups[g][c] for g in range(len(sizes)) for c in range(sizes[g])])
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        chosen_vars = var[offsets[None, :] + choices]          # [S, G] qubit index set to 1
        costs = np.array([o.cost for o in rp.options], dtype=float)
        energies = costs[chosen_vars].sum(axis=1)
        q_conf = 2.0 * max(float(costs.max(initial=0.0)), 1.0)
        bad = np.zeros(total, dtype=bool)
        onehot = np.zeros((total, rp.n), dtype=bool)
        onehot[np.arange(total)[:, None], chosen_vars] = True
        for k, l in rp.conflicts:
            hit = onehot[:, k] & onehot[:, l]
            energies = energies + q_conf * hit
            bad |= hit
        return cls(groups=[list(g) for g in rp.groups], sizes=sizes, strides=strides, choices=choices,
                   energies=energies, feasible=~bad, n_qubits=rp.n)

    @property
    def size(self) -> int:
        return len(self.energies)

    @property
    def optimum_energy(self) -> float:
        return float(self.energies.min())

    def optimum_mask(self) -> np.ndarray:
        e = self.optimum_energy
        return self.energies <= e + ENERGY_TOL * max(1.0, abs(e))

    def bits(self, state: int) -> np.ndarray:
        """Qubit vector of one subspace state."""
        x = np.zeros(self.n_qubits, dtype=int)
        for g, c in enumerate(self.choices[state]):
            x[self.groups[g][c]] = 1
        return x

    def full_index(self, state: int) -> int:
        """Index in the 2^n computational basis (little-endian, as Qiskit)."""
        return int(sum(1 << int(k) for k in np.flatnonzero(self.bits(state))))


# --------------------------------------------------------------------------- simulator

def ring_pairs(m: int) -> list[tuple[int, int]]:
    """XY-mixer pairs inside a group of m options (ring; a single pair for m = 2)."""
    if m < 2:
        return []
    if m == 2:
        return [(0, 1)]
    return [(i, (i + 1) % m) for i in range(m)]


class XYQAOA:
    """Exact QAOA statevector restricted to the one-hot subspace."""

    def __init__(self, space: OneHotSpace):
        self.space = space
        e = space.energies
        self.mean = float(e.mean())
        self.std = float(e.std()) or 1.0
        self.h_std = (e - self.mean) / self.std
        self.order = np.argsort(e, kind="stable")
        self.evals = 0
        # index pairs (state with option a, same state with option b) for every mixer pair
        self.moves = []
        for g, m in enumerate(space.sizes):
            for a, b in ring_pairs(int(m)):
                src = np.flatnonzero(space.choices[:, g] == a)
                self.moves.append((src, src + (b - a) * space.strides[g]))

    def state(self, gammas, betas) -> np.ndarray:
        psi = np.full(self.space.size, 1 / math.sqrt(self.space.size), dtype=complex)   # product of W states
        for g, b in zip(gammas, betas):
            psi *= np.exp(-1j * g * self.h_std)
            c, s = math.cos(b), -1j * math.sin(b)            # exp(-i b (XX+YY)/2) on {|10>, |01>}
            for src, dst in self.moves:
                a0, a1 = psi[src], psi[dst]
                psi[src], psi[dst] = c * a0 + s * a1, s * a0 + c * a1
        self.evals += 1
        return psi

    def probabilities(self, gammas, betas) -> np.ndarray:
        p = np.abs(self.state(gammas, betas)) ** 2
        return p / p.sum()

    def cvar(self, probs: np.ndarray, alpha: float) -> float:
        if alpha >= 1:
            return float(probs @ self.h_std)
        p = probs[self.order]
        cum = np.cumsum(p)
        cut = int(np.searchsorted(cum, alpha))
        before = cum[cut - 1] if cut > 0 else 0.0
        e = self.h_std[self.order]
        return float((p[:cut] @ e[:cut] + (alpha - before) * e[cut]) / alpha)


# --------------------------------------------------------------------------- optimization

@dataclass
class XYLayer:
    p: int
    gammas: list[float]
    betas: list[float]
    objective: float
    expected_energy: float
    p_optimal: float
    p_feasible: float
    amplification_subspace: float      # vs uniform guessing among one-hot states
    amplification_full: float          # vs uniform guessing among all 2^n bitstrings
    shots_for_optimum_95: float
    evaluations: int
    seconds: float


@dataclass
class XYResult:
    n_qubits: int
    subspace_size: int
    optimum_energy: float
    n_optimal: int
    alpha: float
    layers: list[XYLayer]
    hamiltonian_scale: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def run_xy_qaoa(space: OneHotSpace, p_max: int = 3, alpha: float = 0.2, maxiter: int = 150,
                grid: tuple[int, int] = (8, 6), grid_starts: int = 3, verbose: bool = False) -> XYResult:
    """Same optimization scheme as evsp.qaoa.run_qaoa (grid, INTERP, zero-layer start, COBYLA)."""
    from scipy.optimize import minimize

    sim = XYQAOA(space)
    opt = space.optimum_mask()
    n_opt = int(opt.sum())

    def objective(theta):
        p = len(theta) // 2
        return sim.cvar(sim.probabilities(theta[:p], theta[p:]), alpha)

    layers, theta = [], None
    for p in range(1, p_max + 1):
        t0, e0 = time.time(), sim.evals
        if p == 1:
            gs = np.linspace(0.1, math.pi, grid[0])
            bs = np.linspace(0.1, math.pi - 0.1, grid[1])
            scores = sorted((objective(np.array([g, b])), g, b) for g in gs for b in bs)
            starts = [np.array([g, b]) for _, g, b in scores[:grid_starts]]
        else:
            gp, bp = theta[:p - 1], theta[p - 1:]
            starts = [np.concatenate([_interp(gp), _interp(bp)]), np.concatenate([gp, [0.0], bp, [0.0]])]
        best = None
        for s in starts:
            r = minimize(objective, s, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.1})
            if best is None or r.fun < best.fun:
                best = r
        theta = best.x
        g, b = theta[:p], theta[p:]
        probs = sim.probabilities(g, b)
        p_opt = float(probs[opt].sum())
        layer = XYLayer(p=p, gammas=[float(v) for v in g], betas=[float(v) for v in b],
                        objective=float(best.fun), expected_energy=float(probs @ space.energies),
                        p_optimal=p_opt, p_feasible=float(probs[space.feasible].sum()),
                        amplification_subspace=p_opt / (n_opt / space.size),
                        amplification_full=p_opt / (n_opt / 2 ** space.n_qubits),
                        shots_for_optimum_95=shots_for_confidence(p_opt),
                        evaluations=sim.evals - e0, seconds=time.time() - t0)
        layers.append(layer)
        if verbose:
            print(f"  XY p={p}: P(opt) {p_opt:.4f} ({layer.amplification_subspace:.1f}x random one-hot, "
                  f"~{layer.shots_for_optimum_95:.0f} shots for 95 %), P(feasible) {layer.p_feasible:.3f}, "
                  f"{layer.evaluations} evals, {layer.seconds:.1f}s")
    return XYResult(n_qubits=space.n_qubits, subspace_size=space.size, optimum_energy=space.optimum_energy,
                    n_optimal=n_opt, alpha=alpha, layers=layers,
                    hamiltonian_scale={"mean": sim.mean, "std": sim.std})


# --------------------------------------------------------------------------- Qiskit circuit

def _w_state(qc, qubits: list[int]) -> None:
    """|10..0> -> uniform superposition of the m one-hot states (CRY + CX cascade)."""
    m = len(qubits)
    qc.x(qubits[0])
    for k in range(1, m):
        p = 1.0 / (m - k + 1)                     # keep 1/(remaining) of the weight on the current qubit
        theta = 2 * math.acos(math.sqrt(p))
        qc.cry(theta, qubits[k - 1], qubits[k])
        qc.cx(qubits[k], qubits[k - 1])


def xy_qaoa_circuit(rp: RepairProblem, gammas, betas, space: OneHotSpace, measure: bool = True):
    """XY-mixer QAOA as a Qiskit circuit; qubit k = option k of the repair problem.

    Cost layer: costs and conflict penalties only (the one-hot penalty is constant in
    the subspace), RZ / RZZ angles scaled by the subspace energy std, as in XYQAOA.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate

    std = XYQAOA(space).std
    costs = np.array([o.cost for o in rp.options], dtype=float)
    q_conf = 2.0 * max(float(costs.max(initial=0.0)), 1.0)
    # Ising form of sum c_k y_k + q sum y_k y_l with y = (1 - z) / 2
    h = -costs / 2.0
    J = {}
    for k, l in rp.conflicts:
        J[(k, l)] = J.get((k, l), 0.0) + q_conf / 4.0
        h[k] -= q_conf / 4.0
        h[l] -= q_conf / 4.0
    qc = QuantumCircuit(rp.n, name=f"xy_qaoa_p{len(gammas)}")
    for g in rp.groups:
        _w_state(qc, g)
    for gamma, beta in zip(gammas, betas):
        for (k, l), v in J.items():
            qc.rzz(2 * gamma * v / std, k, l)
        for k, v in enumerate(h):
            if abs(v) > 1e-12:
                qc.rz(2 * gamma * v / std, k)
        for g in rp.groups:
            for a, b in ring_pairs(len(g)):
                qc.append(XXPlusYYGate(2 * beta), [g[a], g[b]])
    if measure:
        qc.measure_all()
    return qc

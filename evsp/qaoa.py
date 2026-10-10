"""QAOA for the E-VSP QUBO (package E).

Two engines with the same mathematics:

* `FastQAOA` - exact statevector simulation in numpy. The cost Hamiltonian is
  diagonal, so its 2^n energies are computed once (`Landscape`) and one QAOA layer
  is a phase multiply plus n single-qubit RX rotations. Used for the angle
  optimization loop (hundreds of evaluations), where a generic circuit
  simulator would be far slower.
* `qaoa_circuit` - the same ansatz as a Qiskit circuit (RZ, RZZ, RX gates) for
  sampling with Qiskit primitives, transpilation metrics and noisy simulation.
  `tests/test_qaoa.py` checks that both give identical probabilities.

The cost layer uses the standardized Hamiltonian H' = (H - mean) / std, so the
angles gamma are O(1) whatever the cost units; the ground state is unchanged.

    q = build_qubo(cs, variant_weights("E2", cs.assumptions))
    land = Landscape.from_qubo(q)
    res = run_qaoa(q, land, p_max=3)             # p = 1..3, CVaR + COBYLA, INTERP warm start
    layer = res.layers[-1]; layer.p_optimal, layer.approx_ratio
    qc = qaoa_circuit(q, layer.gammas, layer.betas, land)    # Qiskit circuit for sampling / hardware
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from .qubo import QUBO
from .solvers import Landscape, index_to_bits, repair


# --------------------------------------------------------------------------- simulator

class FastQAOA:
    """Exact QAOA statevector on the diagonal cost Hamiltonian of a Landscape."""

    def __init__(self, landscape: Landscape, precision: str = "single"):
        self.land = landscape
        self.n = landscape.n
        e = landscape.energies
        self.mean = float(e.mean())
        self.std = float(e.std()) or 1.0
        self.h_std = (e - self.mean) / self.std          # standardized cost, gamma ~ O(1)
        self.order = np.argsort(e, kind="stable")        # for CVaR
        # complex64 halves memory traffic (~2x faster); probabilities are renormalized in float64
        self.dtype = np.complex64 if precision == "single" else np.complex128
        self._h = self.h_std.astype(np.float32 if precision == "single" else np.float64)
        self.evals = 0

    def state(self, gammas, betas) -> np.ndarray:
        n = self.n
        psi = np.full(1 << n, 1 / math.sqrt(1 << n), dtype=self.dtype)
        for g, b in zip(gammas, betas):
            psi *= np.exp((-1j * g) * self._h).astype(self.dtype, copy=False)   # cost layer exp(-i g H')
            c, s = self.dtype(math.cos(b)), self.dtype(-1j * math.sin(b))      # RX(2b) = exp(-i b X)
            for k in range(n):                                                 # mixer on every qubit
                v = psi.reshape(-1, 2, 1 << k)
                a0, a1 = v[:, 0, :].copy(), v[:, 1, :]
                v[:, 0, :] = c * a0 + s * a1
                v[:, 1, :] = s * a0 + c * a1
        self.evals += 1
        return psi

    def probabilities(self, gammas, betas) -> np.ndarray:
        p = np.abs(self.state(gammas, betas)).astype(np.float64) ** 2
        return p / p.sum()

    def cvar(self, probs: np.ndarray, alpha: float) -> float:
        """Mean standardized energy of the best `alpha` probability mass (alpha = 1: expectation)."""
        if alpha >= 1:
            return float(probs @ self.h_std)
        p = probs[self.order]
        cum = np.cumsum(p)
        cut = int(np.searchsorted(cum, alpha))
        mass_before = cum[cut - 1] if cut > 0 else 0.0
        e = self.h_std[self.order]
        return float((p[:cut] @ e[:cut] + (alpha - mass_before) * e[cut]) / alpha)


# --------------------------------------------------------------------------- results

@dataclass
class LayerResult:
    p: int
    gammas: list[float]
    betas: list[float]
    objective: float                 # CVaR of the standardized energy at the optimum angles
    expected_energy: float           # <H> in cost units
    approx_ratio: float              # (E_max - <E>) / (E_max - E_min)
    p_optimal: float                 # probability of measuring the ground state
    p_feasible: float                # probability of measuring any feasible schedule
    amplification: float             # p_optimal / P(optimum) of uniform random guessing
    shots_for_optimum_95: float      # measurements needed to see the optimum with 95 % probability
    evaluations: int
    seconds: float
    sampled: dict = field(default_factory=dict)


@dataclass
class QAOAResult:
    n_qubits: int
    alpha: float
    optimum_energy: float
    uniform_p_optimal: float
    uniform_p_feasible: float
    layers: list[LayerResult]
    hamiltonian_scale: dict

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- optimization

def _interp(angles: np.ndarray) -> np.ndarray:
    """INTERP warm start (Zhou et al. 2020): p angles -> p + 1 angles."""
    p = len(angles)
    ext = np.concatenate([[0.0], angles, [0.0]])
    return np.array([(i / p) * ext[i] + ((p - i) / p) * ext[i + 1] for i in range(p + 1)])


def _layer_metrics(sim: FastQAOA, gammas, betas, alpha: float) -> dict:
    probs = sim.probabilities(gammas, betas)
    land = sim.land
    p_opt = land.p_optimal(probs)
    return {
        "objective": sim.cvar(probs, alpha),
        "expected_energy": land.expected_energy(probs),
        "approx_ratio": land.approximation_ratio(probs),
        "p_optimal": p_opt,
        "p_feasible": land.p_feasible(probs),
        "amplification": p_opt / land.uniform_p_optimal(),
        "shots_for_optimum_95": shots_for_confidence(p_opt),
        "probs": probs,
    }


def shots_for_confidence(p: float, confidence: float = 0.95) -> float:
    """Number of measurements after which an outcome of probability p is seen with `confidence`."""
    if p <= 0:
        return math.inf
    if p >= 1:
        return 1.0
    return math.ceil(math.log(1 - confidence) / math.log(1 - p))


def run_qaoa(q: QUBO, landscape: Optional[Landscape] = None, p_max: int = 2, alpha: float = 0.2,
             maxiter: int = 150, grid: tuple[int, int] = (8, 6), grid_starts: int = 3, shots: int = 10_000,
             seed: int = 0, verbose: bool = False) -> QAOAResult:
    """Optimize QAOA angles for p = 1..p_max and report exact and sampled metrics.

    p = 1 is optimized from the `grid_starts` best points of a (gamma, beta) grid. Every next depth
    is optimized from two starts and keeps the better one: the INTERP
    extrapolation of the previous optimum, and the previous optimum padded with a
    zero layer (which reproduces depth p - 1 exactly, so the objective never gets
    worse with depth). The objective is CVaR_alpha of the standardized energy,
    minimized with COBYLA.
    """
    from scipy.optimize import minimize

    land = landscape or Landscape.from_qubo(q)
    sim = FastQAOA(land)
    rng = np.random.default_rng(seed)

    def objective(theta: np.ndarray) -> float:
        p = len(theta) // 2
        return sim.cvar(sim.probabilities(theta[:p], theta[p:]), alpha)

    layers: list[LayerResult] = []
    theta = None
    for p in range(1, p_max + 1):
        t0, e0 = time.time(), sim.evals
        if p == 1:
            gs = np.linspace(0.1, math.pi, grid[0])
            bs = np.linspace(0.1, math.pi - 0.1, grid[1])
            scores = [(objective(np.array([g, b])), g, b) for g in gs for b in bs]
            starts = [np.array([g0, b0]) for _, g0, b0 in sorted(scores)[:grid_starts]]
        else:
            g_prev, b_prev = theta[:p - 1], theta[p - 1:]
            starts = [np.concatenate([_interp(g_prev), _interp(b_prev)]),
                      np.concatenate([g_prev, [0.0], b_prev, [0.0]])]
        best = None
        for start in starts:
            res = minimize(objective, start, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.1})
            if best is None or res.fun < best.fun:
                best = res
        theta = best.x
        gammas, betas = theta[:p], theta[p:]
        m = _layer_metrics(sim, gammas, betas, alpha)
        sampled = sample_metrics(q, land, m.pop("probs"), shots=shots, rng=rng)
        layer = LayerResult(p=p, gammas=[float(v) for v in gammas], betas=[float(v) for v in betas],
                            evaluations=sim.evals - e0, seconds=time.time() - t0, sampled=sampled,
                            **{k: float(v) for k, v in m.items()})
        layers.append(layer)
        if verbose:
            print(f"  p={p}: P(opt) {layer.p_optimal:.4f} ({layer.amplification:.0f}x random, "
                  f"~{layer.shots_for_optimum_95:.0f} shots for 95 %), "
                  f"P(feasible) {layer.p_feasible:.3f}, approx ratio {layer.approx_ratio:.4f}, "
                  f"{layer.evaluations} evals, {layer.seconds:.1f}s")
    return QAOAResult(n_qubits=q.n, alpha=alpha, optimum_energy=land.optimum_energy,
                      uniform_p_optimal=land.uniform_p_optimal(), uniform_p_feasible=land.uniform_p_feasible(),
                      layers=layers, hamiltonian_scale={"mean": sim.mean, "std": sim.std})


def sample_metrics(q: QUBO, land: Landscape, probs: np.ndarray, shots: int = 1000,
                   rng: Optional[np.random.Generator] = None) -> dict:
    """What a real run with `shots` measurements would report (raw and after classical repair)."""
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(len(probs), size=shots, p=probs)
    return counts_metrics(q, land, {int(i): int(c) for i, c in zip(*np.unique(idx, return_counts=True))})


def counts_metrics(q: QUBO, land: Landscape, counts: dict[int, int]) -> dict:
    """Metrics from measurement counts {state index: count}."""
    shots = sum(counts.values())
    idx = np.array(list(counts), dtype=np.int64)
    c = np.array(list(counts.values()), dtype=float)
    feas = land.feasible[idx]
    opt = land.is_optimal(idx)
    best_raw = float(land.energies[idx[feas]].min()) if feas.any() else math.inf
    # repair the 20 most frequent infeasible bitstrings
    repaired_best = best_raw
    n_repaired = 0
    for i in idx[~feas][np.argsort(-c[~feas])][:20]:
        x = repair(q, index_to_bits(i, q.n))
        if x is not None:
            n_repaired += 1
            repaired_best = min(repaired_best, float(q.energy(x)))
    top = idx[np.argmax(c)]
    return {
        "shots": shots,
        "frac_feasible": float(c[feas].sum() / shots),
        "frac_optimal": float(c[opt].sum() / shots),
        "best_feasible_energy": best_raw,
        "found_optimum": bool(opt.any()),
        "best_after_repair": repaired_best,
        "optimum_after_repair": bool(repaired_best <= land.optimum_energy + 1e-6 * max(1.0, abs(land.optimum_energy))),
        "repaired_strings": n_repaired,
        "most_frequent_energy": float(land.energies[top]),
        "distinct_states": int(len(idx)),
    }


# --------------------------------------------------------------------------- Qiskit

def qaoa_circuit(q: QUBO, gammas, betas, landscape: Optional[Landscape] = None, measure: bool = True,
                 std: Optional[float] = None):
    """The same ansatz as FastQAOA as a Qiskit circuit.

    Cost layer exp(-i gamma (H - mean) / std): RZ(2 gamma h_k / std) and RZZ(2 gamma J_kl / std)
    from the Ising form (global phase dropped). Mixer: RX(2 beta) on every qubit.
    Qubit k = variable k; read measured strings with `qubo.bitstring_to_x`.
    """
    from qiskit import QuantumCircuit

    if std is None:
        std = FastQAOA(landscape).std if landscape is not None else float(np.std(_all_energies_or_fail(q)))
    h, J, _ = q.to_ising()
    n = q.n
    qc = QuantumCircuit(n, name=f"qaoa_p{len(gammas)}")
    qc.h(range(n))
    for g, b in zip(gammas, betas):
        for (k, l), v in J.items():
            if abs(v) > 1e-12:
                qc.rzz(2 * g * v / std, k, l)
        for k, v in enumerate(h):
            if abs(v) > 1e-12:
                qc.rz(2 * g * v / std, k)
        qc.rx(2 * b, range(n))
    if measure:
        qc.measure_all()
    return qc


def _all_energies_or_fail(q: QUBO):
    return Landscape.from_qubo(q).energies


def sample_with_qiskit(qc, shots: int = 1000, seed: Optional[int] = None, noise_2q: float = 0.0,
                       noise_1q: float = 0.0) -> dict[int, int]:
    """Run the circuit with Qiskit and return {state index: count}.

    Without noise this uses qiskit's StatevectorSampler. With noise it needs
    qiskit-aer: depolarizing error `noise_1q` on 1-qubit and `noise_2q` on 2-qubit gates.
    """
    if noise_1q or noise_2q:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error

        nm = NoiseModel()
        if noise_1q:
            nm.add_all_qubit_quantum_error(depolarizing_error(noise_1q, 1), ["rz", "rx", "h", "sx", "x"])
        if noise_2q:
            nm.add_all_qubit_quantum_error(depolarizing_error(noise_2q, 2), ["rzz", "cx"])
        sim = AerSimulator(noise_model=nm, seed_simulator=seed)
        job = sim.run(transpile(qc, basis_gates=["h", "rz", "rx", "rzz", "cx", "sx", "x", "measure"]),
                      shots=shots)
        raw = job.result().get_counts()
    else:
        from qiskit.primitives import StatevectorSampler

        sampler = StatevectorSampler(seed=seed)
        raw = sampler.run([qc], shots=shots).result()[0].data.meas.get_counts()
    return {int(bits.replace(" ", ""), 2): int(c) for bits, c in raw.items()}


def circuit_metrics(qc, coupling: str = "all") -> dict:
    """Depth and 2-qubit gate count after transpiling to {rz, sx, x, cx}.

    coupling: "all" (all-to-all, e.g. trapped ions) or "heavy-hex" (IBM-style
    lattice, adds the SWAPs a real superconducting device would need).
    """
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    cm = None
    if coupling == "heavy-hex":
        d = 3
        while True:
            cm = CouplingMap.from_heavy_hex(d)
            if cm.size() >= qc.num_qubits:
                break
            d += 2
    elif coupling != "all":
        raise ValueError("coupling must be 'all' or 'heavy-hex'")
    tq = transpile(qc.remove_final_measurements(inplace=False), basis_gates=["rz", "sx", "x", "cx"],
                   coupling_map=cm, optimization_level=1, seed_transpiler=0)
    ops = tq.count_ops()
    return {"coupling": coupling, "qubits": qc.num_qubits, "physical_qubits": tq.num_qubits,
            "depth": tq.depth(), "cx": int(ops.get("cx", 0)), "ops": {k: int(v) for k, v in ops.items()}}

from pathlib import Path

import numpy as np
import pytest

from evsp import load_instance
from evsp.columns import generate_columns, select_columns, solve_milp
from evsp.qaoa import (FastQAOA, _interp, circuit_metrics, counts_metrics, qaoa_circuit, run_qaoa,
                       sample_with_qiskit, shots_for_confidence)
from evsp.qubo import brute_force, build_qubo, variant_weights
from evsp.solvers import Landscape, bits_to_index, index_to_bits, repair, simulated_annealing

ROOT = Path(__file__).resolve().parents[1]


def _qubo(name, budget, variant="E2", battery=None):
    path = ROOT / "instances" / f"kcm_{name}.json"
    if not path.exists():
        pytest.skip("run scripts/build_instances.py first")
    inst = load_instance(path)
    if battery:
        inst.assumptions["battery_kwh"] = battery
    cs = generate_columns(inst)
    idx = select_columns(cs, budget=budget).indices
    return cs, idx, build_qubo(cs, variant_weights(variant, cs.assumptions), indices=idx)


@pytest.fixture(scope="module")
def tiny8():
    return _qubo("tiny", 8)


@pytest.fixture(scope="module")
def small8():
    return _qubo("small", 8, variant="E4", battery=200)     # has charging terms


@pytest.fixture(scope="module")
def tiny20():
    return _qubo("tiny", 20)


# ---------------------------------------------------------------- landscape

def test_bit_order_round_trip():
    for i in [0, 1, 6, 1023]:
        assert bits_to_index(index_to_bits(i, 10)) == i
    assert list(index_to_bits(6, 4)) == [0, 1, 1, 0]


def test_landscape_matches_brute_force_and_milp(tiny20):
    cs, idx, q = tiny20
    land = Landscape.from_qubo(q)
    x, e = brute_force(q)
    assert land.optimum_energy == pytest.approx(e)
    assert land.feasible[bits_to_index(x)]
    assert land.optimum_energy == pytest.approx(solve_milp(cs, idx).objective)
    assert land.best_feasible_energy == pytest.approx(land.optimum_energy)


def test_feasible_mask_agrees_with_evaluate(small8):
    _, _, q = small8
    land = Landscape.from_qubo(q)
    for i in np.random.default_rng(0).integers(0, 1 << q.n, size=200):
        assert land.feasible[i] == q.evaluate(index_to_bits(i, q.n))["feasible"]


# ---------------------------------------------------------------- simulator vs Qiskit

@pytest.mark.parametrize("fixture", ["tiny8", "small8"])
def test_fast_simulator_equals_qiskit_statevector(request, fixture):
    from qiskit.quantum_info import Statevector

    _, _, q = request.getfixturevalue(fixture)
    land = Landscape.from_qubo(q)
    sim = FastQAOA(land, precision="double")
    rng = np.random.default_rng(1)
    for p in (1, 2, 3):
        g, b = rng.uniform(0, 3, p), rng.uniform(0, 3, p)
        p_fast = sim.probabilities(g, b)
        p_qiskit = Statevector(qaoa_circuit(q, g, b, land, measure=False)).probabilities()
        assert np.abs(p_fast - p_qiskit).max() < 1e-10


def test_single_precision_is_close(tiny8):
    _, _, q = tiny8
    land = Landscape.from_qubo(q)
    g, b = [0.8, 0.3], [0.5, 1.2]
    p64 = FastQAOA(land, precision="double").probabilities(g, b)
    p32 = FastQAOA(land).probabilities(g, b)
    assert np.abs(p64 - p32).max() < 1e-5 and p32.sum() == pytest.approx(1.0)


def test_zero_angles_give_uniform_distribution(tiny8):
    _, _, q = tiny8
    sim = FastQAOA(Landscape.from_qubo(q))
    assert np.allclose(sim.probabilities([0.0], [0.0]), 1 / (1 << q.n))


def test_cvar(tiny8):
    _, _, q = tiny8
    sim = FastQAOA(Landscape.from_qubo(q))
    probs = sim.probabilities([0.7], [0.4])
    assert sim.cvar(probs, 1.0) == pytest.approx(probs @ sim.h_std)
    assert sim.cvar(probs, 1e-9) == pytest.approx(sim.h_std.min(), abs=1e-6)
    assert sim.cvar(probs, 0.1) <= sim.cvar(probs, 0.5) <= sim.cvar(probs, 1.0)


def test_interp_warm_start():
    assert np.allclose(_interp(np.array([0.5])), [0.5, 0.5])
    assert np.allclose(_interp(np.array([0.2, 0.6])), [0.2, 0.4, 0.6])


# ---------------------------------------------------------------- optimization

def test_qaoa_beats_random_guessing_and_depth_never_hurts(tiny8):
    _, _, q = tiny8
    land = Landscape.from_qubo(q)
    res = run_qaoa(q, land, p_max=3, maxiter=80, shots=2000)
    objectives = [L.objective for L in res.layers]
    assert all(b <= a + 1e-9 for a, b in zip(objectives, objectives[1:]))
    best = max(L.p_optimal for L in res.layers)
    assert best > 20 * land.uniform_p_optimal()
    assert any(L.sampled["found_optimum"] for L in res.layers)


def test_done_criterion_tiny_20_qubits(tiny20):
    """Package E done criterion: QAOA on tiny (20 qubits) gives the optimum a nonzero,
    better-than-random probability."""
    _, _, q = tiny20
    land = Landscape.from_qubo(q)
    layer = run_qaoa(q, land, p_max=1, maxiter=40, grid=(5, 4), grid_starts=1, shots=20_000).layers[0]
    assert layer.p_optimal > 10 * land.uniform_p_optimal()
    assert layer.shots_for_optimum_95 < 1e5


def test_shots_for_confidence():
    assert shots_for_confidence(1.0) == 1
    assert shots_for_confidence(0.0) == float("inf")
    assert shots_for_confidence(0.05) == 59        # 1 - 0.95^59 >= 0.95


# ---------------------------------------------------------------- classical baselines

def test_simulated_annealing_finds_optimum(tiny20):
    _, _, q = tiny20
    land = Landscape.from_qubo(q)
    sa = simulated_annealing(q, reads=50, sweeps=300, seed=0, landscape=land)
    assert sa.best_energy == pytest.approx(land.optimum_energy)
    assert sa.p_optimal > 0


def test_repair(small8):
    _, _, q = small8
    land = Landscape.from_qubo(q)
    rng = np.random.default_rng(3)
    opt = index_to_bits(land.optimum_indices[0], q.n)
    assert np.array_equal(repair(q, opt), opt)                    # feasible input unchanged
    for i in rng.integers(0, 1 << q.n, size=100):
        x = repair(q, index_to_bits(i, q.n))
        assert x is None or q.evaluate(x)["feasible"]


# ---------------------------------------------------------------- Qiskit sampling and circuits

def test_qiskit_sampling_matches_probabilities(tiny8):
    _, _, q = tiny8
    land = Landscape.from_qubo(q)
    g, b = [0.8], [0.4]
    probs = FastQAOA(land, precision="double").probabilities(g, b)
    counts = sample_with_qiskit(qaoa_circuit(q, g, b, land), shots=20_000, seed=7)
    freq = np.zeros(1 << q.n)
    for i, c in counts.items():
        freq[i] = c / 20_000
    assert np.abs(freq - probs).max() < 0.02
    m = counts_metrics(q, land, counts)
    assert m["shots"] == 20_000 and 0 <= m["frac_optimal"] <= m["frac_feasible"] <= 1


def test_noisy_sampling_runs(tiny8):
    pytest.importorskip("qiskit_aer")
    _, _, q = tiny8
    land = Landscape.from_qubo(q)
    counts = sample_with_qiskit(qaoa_circuit(q, [0.8], [0.4], land), shots=500, seed=1,
                                noise_1q=0.001, noise_2q=0.01)
    assert sum(counts.values()) == 500


def test_circuit_metrics(tiny8):
    _, _, q = tiny8
    land = Landscape.from_qubo(q)
    qc = qaoa_circuit(q, [0.8, 0.2], [0.4, 0.3], land)
    full, hh = circuit_metrics(qc, "all"), circuit_metrics(qc, "heavy-hex")
    assert full["cx"] > 0 and hh["cx"] >= full["cx"]
    assert hh["physical_qubits"] >= q.n

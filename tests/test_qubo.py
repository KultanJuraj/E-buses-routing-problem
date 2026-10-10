from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from evsp import load_instance
from evsp.columns import generate_columns, select_columns, solve_milp
from evsp.qubo import Weights, bitstring_to_x, brute_force, build_qubo, tune_weights, variant_weights

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ["E1", "E2", "E3", "E4"]


def _selection(name, battery=None, budget=20):
    path = ROOT / "instances" / f"kcm_{name}.json"
    if not path.exists():
        pytest.skip("run scripts/build_instances.py first")
    inst = load_instance(path)
    if battery:
        inst.assumptions["battery_kwh"] = battery
    cs = generate_columns(inst)
    sel = select_columns(cs, budget=budget)
    return cs, sel.indices


@pytest.fixture(scope="module")
def tiny():
    return _selection("tiny")


@pytest.fixture(scope="module")
def small():
    return _selection("small", battery=200)       # has charging, conflicts and tight pairs


def _weights(cs, variant, idx):
    v = solve_milp(cs, idx).n_vehicles if variant == "E3" else None
    return variant_weights(variant, cs.assumptions, V=v)


def _direct_energy(cs, idx, q, x):
    """H evaluated straight from its definition, independent of the QUBO expansion."""
    w = q.weights
    cols = [cs.columns[k] for k in idx]
    e = sum(xk * (w.w_veh + w.w_dh * c.deadhead_km + w.w_en * c.energy_cost) for xk, c in zip(x, cols))
    for t in cs.trip_ids:
        e += q.meta["P"] * (sum(xk for xk, c in zip(x, cols) if t in c.trips) - 1) ** 2
    conflicts, tight = cs.pairs(idx)
    e += q.meta["Q"] * sum(x[k] * x[l] for k, l in conflicts)
    e += w.w_rob * sum(r * x[k] * x[l] for k, l, r in tight)
    if w.R:
        p, _, _ = cs.power_tensor(idx)
        p = p / w.peak_unit_kw
        if w.peak_scope == "total":
            p = p.sum(axis=1, keepdims=True)
        e += w.R * float((np.einsum("k,kst->st", np.asarray(x, float), p) ** 2).sum())
    if q.meta["M"] and w.V is not None:
        e += q.meta["M"] * (sum(x) - w.V) ** 2
    return e


@pytest.mark.parametrize("variant", VARIANTS)
def test_expansion_matches_definition(small, variant):
    cs, idx = small
    q = build_qubo(cs, _weights(cs, variant, idx), indices=idx)
    rng = np.random.default_rng(0)
    for x in rng.integers(0, 2, size=(25, q.n)):
        assert q.energy(x) == pytest.approx(_direct_energy(cs, idx, q, x), rel=1e-9, abs=1e-6)


def test_peak_scope_total(small):
    cs, idx = small
    w = variant_weights("E4", cs.assumptions)
    w.peak_scope = "total"
    q = build_qubo(cs, w, indices=idx)
    x = np.ones(q.n, dtype=int)
    assert q.energy(x) == pytest.approx(_direct_energy(cs, idx, q, x), rel=1e-9)


@pytest.mark.parametrize("variant", VARIANTS)
def test_ising_is_the_same_function(small, variant):
    cs, idx = small
    q = build_qubo(cs, _weights(cs, variant, idx), indices=idx)
    h, J, c = q.to_ising()
    for x in np.random.default_rng(1).integers(0, 2, size=(20, q.n)):
        z = 1 - 2 * x
        e = c + h @ z + sum(v * z[k] * z[l] for (k, l), v in J.items())
        assert e == pytest.approx(q.energy(x), rel=1e-9, abs=1e-6)


@pytest.mark.parametrize("fixture,variant", [("tiny", "E1"), ("tiny", "E2"), ("small", "E1"), ("small", "E2")])
def test_ground_state_equals_milp(request, fixture, variant):
    cs, idx = request.getfixturevalue(fixture)
    q = build_qubo(cs, variant_weights(variant, cs.assumptions), indices=idx)
    x, e = brute_force(q)
    rep = q.evaluate(x)
    milp = solve_milp(cs, idx)
    assert rep["feasible"]
    assert rep["n_vehicles"] == milp.n_vehicles
    if variant == "E2":                      # E2 objective is exactly the MILP objective
        assert e == pytest.approx(milp.objective, rel=1e-9)


@pytest.mark.parametrize("variant", VARIANTS)
def test_auto_penalties_give_feasible_ground_state(small, variant):
    cs, idx = small
    q = build_qubo(cs, _weights(cs, variant, idx), indices=idx)
    x, _ = brute_force(q)
    rep = q.evaluate(x)
    assert rep["feasible"], rep
    if variant == "E3":
        assert rep["n_vehicles"] == q.weights.V


def test_too_small_penalty_breaks_feasibility(tiny):
    """Sanity check of the penalty logic: with P tiny the solver drops trips."""
    cs, idx = tiny
    w = variant_weights("E2", cs.assumptions)
    w.P = 1.0
    x, _ = brute_force(build_qubo(cs, w, indices=idx))
    assert not build_qubo(cs, w, indices=idx).evaluate(x)["feasible"]


def test_scaled_keeps_ground_state(small):
    cs, idx = small
    q = build_qubo(cs, variant_weights("E4", cs.assumptions), indices=idx)
    qs = q.scaled()
    assert max(np.abs(qs.linear).max(), max(abs(v) for v in qs.quadratic.values())) == pytest.approx(1.0)
    assert np.array_equal(brute_force(q)[0], brute_force(qs)[0])


def test_bitstring_order():
    assert list(bitstring_to_x("0011")) == [1, 1, 0, 0]       # qubit 0 is the rightmost character


def test_qiskit_operator_diagonal(small):
    pytest.importorskip("qiskit")
    cs, idx = small
    sub = idx[:10]                                            # 2^10 diagonal, cheap
    q = build_qubo(cs, variant_weights("E4", cs.assumptions), indices=sub)
    op, const = q.to_sparse_pauli_op()
    diag = np.real(op.to_matrix(sparse=True).diagonal())
    for x in np.random.default_rng(2).integers(0, 2, size=(30, q.n)):
        index = int(sum(int(b) << k for k, b in enumerate(x)))
        assert diag[index] + const == pytest.approx(q.energy(x), rel=1e-9, abs=1e-6)


def test_dimod_bqm(small):
    pytest.importorskip("dimod")
    cs, idx = small
    q = build_qubo(cs, variant_weights("E4", cs.assumptions), indices=idx)
    bqm = q.to_bqm()
    for x in np.random.default_rng(3).integers(0, 2, size=(10, q.n)):
        assert bqm.energy({k: int(v) for k, v in enumerate(x)}) == pytest.approx(q.energy(x), rel=1e-9)


def test_quadratic_program(small):
    pytest.importorskip("qiskit_optimization")
    cs, idx = small
    q = build_qubo(cs, variant_weights("E2", cs.assumptions), indices=idx)
    qp = q.to_quadratic_program()
    x = brute_force(q)[0]
    assert qp.objective.evaluate(x) == pytest.approx(q.energy(x), rel=1e-9)


def test_coverage_pairs_only_between_columns_sharing_a_trip(tiny):
    cs, idx = tiny
    w = variant_weights("E1", cs.assumptions)
    q = build_qubo(cs, w, indices=idx)
    for k, l in combinations(range(q.n), 2):
        shared = set(cs.columns[idx[k]].trips) & set(cs.columns[idx[l]].trips)
        assert ((k, l) in q.quadratic) == bool(shared)


# ---------------------------------------------------------------- review fixes (PR #4)

@pytest.mark.parametrize("variant", VARIANTS)
def test_normalization_only_changes_units(small, variant):
    """normalize_costs divides the whole H (costs, R, w_rob, penalties) by one factor."""
    cs, idx = small
    raw = build_qubo(cs, _weights(cs, variant, idx), indices=idx)
    w = _weights(cs, variant, idx)
    w.normalize_costs = True
    norm = build_qubo(cs, w, indices=idx)
    scale = norm.meta["cost_scale"]
    assert scale > 1
    assert np.max(norm._costs / scale) <= 1 + 1e-12
    for x in np.random.default_rng(4).integers(0, 2, size=(20, raw.n)):
        assert norm.energy(x) * scale == pytest.approx(raw.energy(x), rel=1e-9, abs=1e-6)
    assert np.array_equal(brute_force(raw)[0], brute_force(norm)[0])


def test_normalization_keeps_explicit_penalties_in_cost_units(small):
    cs, idx = small
    w = variant_weights("E4", cs.assumptions)
    w.P, w.Q = 5000.0, 7000.0
    raw = build_qubo(cs, w, indices=idx)
    w2 = variant_weights("E4", cs.assumptions)
    w2.P, w2.Q, w2.normalize_costs = 5000.0, 7000.0, True
    norm = build_qubo(cs, w2, indices=idx)
    assert norm.meta["P"] * norm.meta["cost_scale"] == pytest.approx(raw.meta["P"])
    assert norm.meta["Q"] * norm.meta["cost_scale"] == pytest.approx(raw.meta["Q"])


def test_energy_basis_kwh(small):
    cs, idx = small
    w = variant_weights("E2", cs.assumptions)
    w.energy_basis = "kwh"
    q = build_qubo(cs, w, indices=idx)
    cols = [cs.columns[k] for k in idx]
    expected = [w.w_veh + w.w_dh * c.deadhead_km + w.w_en * (c.grid_kwh_day + c.refill_kwh) for c in cols]
    assert q._costs == pytest.approx(expected)
    assert q.evaluate(brute_force(q)[0])["feasible"]


def test_tune_weights_keeps_e2_comparable_with_milp(small):
    cs, idx = small
    w, trials = tune_weights(cs, variant_weights("E2", cs.assumptions), idx)
    assert w.R == 0 and {t["R"] for t in trials} == {0.0}
    assert sum(t["selected"] for t in trials) == 1
    x, e = brute_force(build_qubo(cs, w, indices=idx))
    assert e == pytest.approx(solve_milp(cs, idx).objective, rel=1e-9)


def test_tune_weights_never_trades_vehicles_for_peak(small):
    cs, idx = small
    w, trials = tune_weights(cs, variant_weights("E4", cs.assumptions), idx)
    feasible = [t for t in trials if t["feasible"]]
    chosen = next(t for t in trials if t["selected"])
    assert chosen["feasible"] and len({t["R"] for t in trials}) > 1
    assert chosen["n_vehicles"] == min(t["n_vehicles"] for t in feasible)
    assert chosen["peak_kw_total"] == min(t["peak_kw_total"] for t in feasible
                                         if t["n_vehicles"] == chosen["n_vehicles"])


def test_weights_validation():
    with pytest.raises(ValueError):
        Weights(peak_scope="grid")
    with pytest.raises(ValueError):
        Weights(energy_basis="joules")
    with pytest.raises(ValueError):
        Weights(R=-1.0)


def test_optional_packages_are_not_required_for_import():
    """qubo.py must import without dimod / qiskit-optimization (they are used lazily)."""
    import subprocess
    import sys
    code = ("import sys; sys.modules['dimod'] = None; sys.modules['qiskit_optimization'] = None; "
            "import evsp.qubo")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

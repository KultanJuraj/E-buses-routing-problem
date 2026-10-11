from pathlib import Path
import json

import numpy as np
import pytest

from evsp import load_instance
from evsp.columns import load_columns
from evsp.disruption import Disruption, Schedule, apply_assignment, build_repair_problem, plan_duty
from evsp.qaoa import FastQAOA, run_qaoa
from evsp.qubo import brute_force
from evsp.solvers import Landscape, bits_to_index
from evsp.xy_qaoa import OneHotSpace, XYQAOA, _w_state, ring_pairs, run_xy_qaoa, xy_qaoa_circuit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def setup():
    cols = ROOT / "columns" / "kcm_small_b20_bat200.json"
    inst_path = ROOT / "instances" / "kcm_small.json"
    if not cols.exists() or not inst_path.exists():
        pytest.skip("run build_instances.py and build_columns.py --battery 200 first")
    cs = load_columns(cols)
    inst = load_instance(inst_path)
    inst.assumptions.update(cs.assumptions)
    return inst, Schedule.from_columns(cs)


def _problem(setup, bus, at_min, reserves=3):
    inst, sched = setup
    return build_repair_problem(inst, sched, Disruption.bus_failure(sched, bus, at_min), reserves=reserves)


@pytest.fixture(scope="module")
def demo(setup):                     # 20 qubits, 480 one-hot states
    return _problem(setup, "k4", 6 * 60)


@pytest.fixture(scope="module")
def small(setup):                    # 9 qubits, for full-circuit checks
    return _problem(setup, "k190", 6 * 60 + 30, reserves=2)


# ---------------------------------------------------------------- duty replay

def test_plan_duty_reproduces_the_schedule(setup):
    inst, sched = setup
    for bus, col in sched.duties.items():
        plan = plan_duty(inst, col.trips, vehicle_cost=col.vehicle_cost)
        assert plan is not None, bus
        assert plan.cost == pytest.approx(col.cost, abs=1e-3)          # columns file rounds to 4 decimals
        assert plan.final_soc == pytest.approx(col.final_soc, abs=1e-3)


def test_plan_duty_rejects_impossible_sequences(setup):
    inst, sched = setup
    a, b = sched.duties["k4"].trips[:2], sched.duties["k190"].trips[:2]
    assert plan_duty(inst, [a[0], b[0]]) is None                      # overlapping trips
    first = inst.trip_by_id[a[0]]
    assert plan_duty(inst, [a[0]], earliest_start=first.dep_min) is None   # cannot pull out in time


def test_blocked_station_is_avoided(setup):
    inst, sched = setup
    col = sched.duties["k4"]
    station = col.charges[0].station
    plan = plan_duty(inst, col.trips, blocked={station: [(0, 2000)]})
    assert plan is None or all(ch.station != station for ch in plan.charges)


# ---------------------------------------------------------------- repair problem

def test_bus_failure_problem(setup, demo):
    inst, sched = setup
    tb = inst.trip_by_id
    expected = [t for t in sched.duties["k4"].trips if tb[t].dep_min >= 6 * 60]
    assert demo.trips == expected
    for t, g in zip(demo.trips, demo.groups):
        kinds = [demo.options[k].kind for k in g]
        assert kinds[-1] == "cancel" and all(demo.options[k].trip == t for k in g)
    assert "k4" not in {o.target for o in demo.options}


def test_every_option_and_conflict_is_real(setup, demo):
    inst, _ = setup
    for k, l in demo.conflicts:
        ok, ol = demo.options[k], demo.options[l]
        assert ok.target is not None and ol.target is not None and ok.trip != ol.trip
    # each single option replays; each non-conflicting pair on one bus replays
    for k, o in enumerate(demo.options):
        if o.kind != "cancel":
            x = np.zeros(demo.n, dtype=int)
            for g in demo.groups:
                x[g[-1]] = 1                                   # everything else cancelled
            x[[g for g in demo.groups if k in g][0][-1]] = 0
            x[k] = 1
            assert apply_assignment(inst, demo, x).jointly_feasible, o.name


def test_qubo_matches_one_hot_space(demo):
    q = demo.to_qubo()
    space = OneHotSpace.from_problem(demo)
    for s in np.random.default_rng(0).integers(0, space.size, 50):
        assert q.energy(space.bits(s)) == pytest.approx(space.energies[s], rel=1e-9)
    x, e = brute_force(q)
    assert e == pytest.approx(space.optimum_energy)
    land = Landscape.from_qubo(q)
    assert int(land.feasible.sum()) == int(space.feasible.sum())


def test_optimal_repair_is_a_valid_schedule(setup, demo):
    inst, _ = setup
    space = OneHotSpace.from_problem(demo)
    plan = apply_assignment(inst, demo, space.bits(int(np.argmin(space.energies))))
    assert plan.feasible and plan.jointly_feasible and plan.capacity_ok
    assert set(plan.assignment) == set(demo.trips)


def test_charger_outage(setup):
    inst, sched = setup
    ev = Disruption.charger_outage("S3", 7 * 60, 9 * 60)
    rp = build_repair_problem(inst, sched, ev, reserves=1)
    assert rp.trips
    for col in rp.base_plans.values():
        assert not any(ch.station == "S3" and ch.start_min < 9 * 60 and 7 * 60 < ch.end_min for ch in col.charges)


# ---------------------------------------------------------------- XY mixer

def test_ring_pairs():
    assert ring_pairs(1) == [] and ring_pairs(2) == [(0, 1)]
    assert ring_pairs(4) == [(0, 1), (1, 2), (2, 3), (3, 0)]


@pytest.mark.parametrize("m", [1, 2, 3, 5])
def test_w_state(m):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit(m)
    _w_state(qc, list(range(m)))
    probs = Statevector(qc).probabilities()
    one_hot = [1 << k for k in range(m)]
    assert np.allclose(probs[one_hot], 1 / m) and probs.sum() == pytest.approx(1)


def test_xy_simulator_equals_qiskit_circuit(small):
    from qiskit.quantum_info import Statevector

    space = OneHotSpace.from_problem(small)
    sim = XYQAOA(space)
    full = [space.full_index(s) for s in range(space.size)]
    rng = np.random.default_rng(1)
    for p in (1, 2):
        g, b = rng.uniform(0, 3, p), rng.uniform(0, 3, p)
        p_sim = sim.probabilities(g, b)
        p_qk = Statevector(xy_qaoa_circuit(small, g, b, space, measure=False)).probabilities()
        assert np.abs(p_qk[full] - p_sim).max() < 1e-10
        assert p_qk[full].sum() == pytest.approx(1.0)          # never leaves the one-hot subspace


def test_xy_zero_angles_uniform(demo):
    space = OneHotSpace.from_problem(demo)
    assert np.allclose(XYQAOA(space).probabilities([0.0], [0.0]), 1 / space.size)


def test_done_criterion_xy_beats_penalty_qaoa(setup):
    """Package F done criterion: on the same subproblem the XY mixer gives clearly higher
    P(feasible) and P(optimum) than penalty QAOA with the X mixer."""
    rp = _problem(setup, "k585", 7 * 60)                     # 14 qubits: penalty QAOA is cheap
    space = OneHotSpace.from_problem(rp)
    xy = run_xy_qaoa(space, p_max=1, maxiter=60).layers[0]
    q = rp.to_qubo()
    land = Landscape.from_qubo(q)
    pen = run_qaoa(q, land, p_max=1, maxiter=60, shots=1000).layers[0]
    assert xy.p_feasible > 3 * pen.p_feasible
    assert xy.p_optimal > 3 * pen.p_optimal
    # the penalty QUBO and the XY subspace agree on the optimum
    assert land.optimum_energy == pytest.approx(space.optimum_energy)
    idx = bits_to_index(space.bits(int(np.argmin(space.energies))))
    assert land.is_optimal(idx)


def test_penalty_simulator_agrees_with_xy_space_energies(small):
    q = small.to_qubo()
    land = Landscape.from_qubo(q)
    space = OneHotSpace.from_problem(small)
    full = [space.full_index(s) for s in range(space.size)]
    assert np.allclose(land.energies[full], space.energies)
    assert FastQAOA(land).probabilities([0.0], [0.0]).sum() == pytest.approx(1.0)


# ---------------------------------------------------------------- event-time consistency (PR #6)

def test_history_before_the_event_is_kept(setup):
    from evsp.disruption import _history_matches

    inst, sched = setup
    for ev in (Disruption.charger_outage("S3", 7 * 60, 9 * 60), Disruption.bus_failure(sched, "k4", 6 * 60)):
        rp = build_repair_problem(inst, sched, ev, reserves=1)
        for bus, col in rp.base_plans.items():
            if not bus.endswith("*"):
                assert _history_matches(sched.duties[bus], col, inst, ev.at_min), (ev.describe(), bus)


def test_charger_outage_reroutes_instead_of_rewriting_the_past(setup):
    inst, sched = setup
    ev = Disruption.charger_outage("S3", 7 * 60, 9 * 60)
    rp = build_repair_problem(inst, sched, ev, reserves=1, max_bus_options=2)
    k4 = rp.base_plans["k4"]                      # stranded at 24 % SoC after 07:13: charges after the outage
    assert all(ch.start_min >= 7 * 60 for ch in k4.charges)
    assert k4.trips == sched.duties["k4"].trips[:2]
    space = OneHotSpace.from_problem(rp)
    from evsp.xy_qaoa import best_valid_state
    state, plan, rank = best_valid_state(inst, rp, space)
    assert rank == 0 and plan.feasible and not plan.cancelled


def test_trip_in_progress_can_only_be_cancelled(setup):
    inst, sched = setup
    rp = build_repair_problem(inst, sched, Disruption.bus_failure(sched, "k4", 6 * 60 + 30), reserves=2)
    running = [t for t in sched.duties["k4"].trips if inst.trip_by_id[t].dep_min < 6 * 60 + 30]
    assert running[-1] in rp.trips
    g = rp.groups[rp.trips.index(running[-1])]
    assert [rp.options[k].kind for k in g] == ["cancel"]


def test_same_bus_conflicts_are_replayed_pairs(setup, demo):
    inst, sched = setup
    strict = build_repair_problem(inst, sched, Disruption.bus_failure(sched, "k4", 6 * 60), reserves=3,
                                  one_trip_per_bus=True)
    assert set(demo.conflicts) < set(strict.conflicts)
    allowed = [(k, l) for k in range(demo.n) for l in range(k + 1, demo.n)
               if demo.options[k].target and demo.options[k].target == demo.options[l].target
               and demo.options[k].trip != demo.options[l].trip and (k, l) not in set(demo.conflicts)]
    assert allowed                                # some bus can take two affected trips
    for k, l in allowed:                          # and every such pair really replays
        x = np.zeros(demo.n, dtype=int)
        for g in demo.groups:
            x[g[-1]] = 1
        for v in (k, l):
            x[[g for g in demo.groups if v in g][0][-1]] = 0
            x[v] = 1
        assert apply_assignment(inst, demo, x).feasible


def test_malformed_bitstrings_are_rejected(demo):
    q = demo.to_qubo()
    with pytest.raises(ValueError):
        q.energy(np.full(demo.n, 0.7))
    with pytest.raises(ValueError):
        q.energy(np.zeros(demo.n + 1))
    with pytest.raises(ValueError):
        apply_assignment(None, demo, np.full(demo.n, 0.5))


def test_cancel_all_fallback(setup, demo):
    from evsp.disruption import cancel_all_assignment

    inst, _ = setup
    x, plan = cancel_all_assignment(inst, demo)
    assert plan.feasible and sorted(plan.cancelled) == sorted(demo.trips) and x.sum() == len(demo.trips)


def test_stranded_bus_keeps_the_trips_it_already_drove(setup):
    inst, sched = setup
    ev = Disruption.charger_outage("S3", 7 * 60 + 30, 8 * 60 + 30)    # k4 is charging at S3 at 07:30
    rp = build_repair_problem(inst, sched, ev, reserves=1, max_bus_options=1)
    assert "k4*" in rp.base_plans and "k4" not in rp.base_plans
    tb = inst.trip_by_id
    assert all(tb[t].dep_min >= ev.at_min for t in rp.trips)
    assert set(rp.trips) >= {t for t in sched.duties["k4"].trips if tb[t].dep_min >= ev.at_min}


def test_precomputed_result_checks_the_actual_problem(setup, demo, tmp_path):
    """results/ is not versioned, so the test writes its own file in the format of scripts/run_repair.py."""
    from evsp.demo import _precomputed_penalty

    _, sched = setup
    event = Disruption.bus_failure(sched, "k4", 6 * 60)
    raw = {"event": event.__dict__, "trips": demo.trips, "n_qubits": demo.n, "subspace_size": demo.subspace_size,
           "options": [o.__dict__ for o in demo.options], "conflicts": demo.conflicts,
           "penalty_qaoa": {"layers": [{"p": 1, "p_optimal": 0.01, "p_feasible": 0.2},
                                       {"p": 2, "p_optimal": 0.03, "p_feasible": 0.3}]}}
    result = tmp_path / "repair_bus_k4_0600.json"
    result.write_text(json.dumps(raw, default=float), encoding="utf-8")
    row, source = _precomputed_penalty(event, demo, result)
    assert row["p"] == 2 and row["p_optimal"] == 0.03 and "precomputed" in source

    raw["conflicts"] = [[9, 14]]
    result.write_text(json.dumps(raw, default=float), encoding="utf-8")
    assert _precomputed_penalty(event, demo, result)[0] is None
    assert _precomputed_penalty(event, demo, tmp_path / "missing.json")[0] is None


def test_soc_includes_charging_after_the_last_trip(setup, demo):
    from evsp.viz import soc_timeline
    from evsp.xy_qaoa import best_valid_state

    inst, _ = setup
    plan = best_valid_state(inst, demo, OneHotSpace.from_problem(demo))[1]
    charged_home = [c for c in plan.duties.values()
                    if any(ch.start_min >= inst.trip_by_id[c.trips[-1]].arr_min for ch in c.charges)]
    assert charged_home
    for col in charged_home:
        times, soc = soc_timeline(inst, col)
        assert times == sorted(times)
        assert soc[-1] == pytest.approx(col.final_soc)
        assert min(soc) == pytest.approx(col.min_soc)
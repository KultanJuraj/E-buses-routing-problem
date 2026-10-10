from pathlib import Path

import pytest

from evsp import load_instance
from evsp.classical_baselines import (baseline_blocks, battery_check, block_deadhead_km, exact_repair, greedy_electric,
                            greedy_repair, is_valid_cover, min_fleet_matching, scenario1, scenario2, scenario3,
                            solve_flow)
from evsp.columns import generate_columns, solve_milp, validate_column
from evsp.disruption import Disruption, Schedule, _capacity_ok, build_repair_problem
from evsp.xy_qaoa import OneHotSpace, best_valid_state

ROOT = Path(__file__).resolve().parents[1]


def _inst(name, battery=None):
    path = ROOT / "instances" / f"kcm_{name}.json"
    if not path.exists():
        pytest.skip("run scripts/build_instances.py first")
    inst = load_instance(path)
    if battery is not None:
        inst.assumptions["battery_kwh"] = battery
    return inst


# ---------------------------------------------------------------- diesel scenarios

@pytest.mark.parametrize("name", ["tiny", "small", "medium"])
def test_flow_fleet_equals_matching(name):
    inst = _inst(name)
    s1 = scenario1(inst)
    assert s1.vehicles == s1.extra["matching_fleet"] == min_fleet_matching(inst)
    assert is_valid_cover(inst, s1.blocks)


def test_scenario2_is_a_valid_minimum_cost_cover():
    inst = _inst("medium")
    s1, s2 = scenario1(inst), scenario2(inst)
    assert is_valid_cover(inst, s2.blocks)
    a = inst.assumptions
    assert s2.deadhead_km == pytest.approx(sum(block_deadhead_km(inst, b) for b in s2.blocks))
    assert s2.cost == pytest.approx(a["vehicle_cost"] * s2.vehicles + a["deadhead_cost_per_km"] * s2.deadhead_km)
    assert s2.vehicles == s1.vehicles              # 500 per bus dominates the deadhead cost here
    real = baseline_blocks(inst)                   # any valid cover costs at least as much
    assert s2.cost <= real.cost + 1e-6


def test_scenario3_tradeoff_and_infeasibility():
    inst = _inst("large")
    v0 = scenario1(inst).vehicles
    assert scenario3(inst, v0 - 1) is None
    kms = [scenario3(inst, v).deadhead_km for v in (v0, v0 + 2, v0 + 6)]
    assert kms[0] >= kms[1] >= kms[2] and kms[0] > kms[2]
    s = scenario3(inst, v0 + 2)
    assert s.vehicles <= v0 + 2 and is_valid_cover(inst, s.blocks)


def test_flow_lp_is_integral_on_weighted_costs():
    inst = _inst("small")
    s = solve_flow(inst, w_veh=3.0, w_km=7.0)      # raises if the LP were fractional
    assert is_valid_cover(inst, s.blocks)


# ---------------------------------------------------------------- electric

@pytest.mark.parametrize("name", ["tiny", "small"])
def test_greedy_electric_is_valid_and_not_better_than_milp(name):
    inst = _inst(name, battery=200)
    g = greedy_electric(inst)
    assert g.status == "ok"
    assert sorted(t for c in g.duties for t in c.trips) == sorted(t.id for t in inst.trips)
    assert all(validate_column(inst, c) == [] for c in g.duties)      # valid columns of package B
    assert _capacity_ok(inst, g.duties)
    sol = solve_milp(generate_columns(inst))
    assert g.cost >= sol.objective - 1e-6
    assert g.vehicles >= scenario1(inst).vehicles                     # diesel fleet is a lower bound


def test_battery_check_of_diesel_optimum():
    inst = _inst("small", battery=200)
    rep = battery_check(inst, scenario2(inst).blocks)
    assert rep["blocks"] == 5 and rep["battery_feasible"] < 5        # the diesel optimum needs repair
    big = _inst("small")
    assert battery_check(big, scenario2(big).blocks)["battery_feasible"] == 5


# ---------------------------------------------------------------- repair

@pytest.fixture(scope="module")
def demo():
    from evsp.columns import load_columns

    cols = ROOT / "columns" / "kcm_small_b20_bat200.json"
    if not cols.exists():
        pytest.skip("run build_columns.py --battery 200 first")
    cs = load_columns(cols)
    inst = _inst("small")
    inst.assumptions.update(cs.assumptions)
    return inst, Schedule.from_columns(cs)


@pytest.mark.parametrize("bus,at", [("k4", 6 * 60), ("k4", 11 * 60), ("k585", 7 * 60)])
def test_greedy_and_exact_repair(demo, bus, at):
    inst, sched = demo
    rp = build_repair_problem(inst, sched, Disruption.bus_failure(sched, bus, at), reserves=3)
    _, greedy = greedy_repair(inst, rp)
    _, exact = exact_repair(inst, rp)
    _, qubo, _ = best_valid_state(inst, rp, OneHotSpace.from_problem(rp))
    assert greedy.feasible and exact.feasible
    assert exact.extra_cost <= qubo.extra_cost + 1e-6
    assert exact.extra_cost <= greedy.extra_cost + 1e-6


def test_exact_repair_respects_state_limit(demo):
    inst, sched = demo
    rp = build_repair_problem(inst, sched, Disruption.bus_failure(sched, "k4", 6 * 60), reserves=3)
    assert exact_repair(inst, rp, max_states=10) is None

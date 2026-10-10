from pathlib import Path

import pytest

from evsp import load_instance
from evsp.columns import (check_schedule, generate_columns, load_columns, select_columns, solve_milp,
                          validate_column)

ROOT = Path(__file__).resolve().parents[1]


def _instance(name, battery=None):
    path = ROOT / "instances" / f"kcm_{name}.json"
    if not path.exists():
        pytest.skip("run scripts/build_instances.py first")
    inst = load_instance(path)
    if battery:
        inst.assumptions["battery_kwh"] = battery
    return inst


@pytest.fixture(scope="module")
def tiny():
    inst = _instance("tiny")
    return inst, generate_columns(inst)


@pytest.fixture(scope="module")
def small_low_battery():
    inst = _instance("small", battery=200)      # forces charging stops
    return inst, generate_columns(inst)


def test_every_column_replays_feasible(tiny, small_low_battery):
    for inst, cs in (tiny, small_low_battery):
        for col in cs.columns:
            assert validate_column(inst, col) == [], col.id


def test_every_trip_has_a_single_trip_column(tiny):
    inst, cs = tiny
    singles = {c.trips[0] for c in cs.columns if c.n_trips == 1}
    assert singles == {t.id for t in inst.trips}


def test_tiny_done_criterion(tiny):
    """Done criterion of package B: the columns build a schedule covering all 8 trips."""
    inst, cs = tiny
    sel = select_columns(cs, budget=20)
    assert len(sel.indices) <= 20
    sol = solve_milp(cs, sel.indices)
    assert check_schedule(cs, sol.columns)["ok"]
    covered = sorted(t for k in sol.columns for t in cs.columns[k].trips)
    assert covered == sorted(t.id for t in inst.trips)
    assert sol.n_vehicles == len(inst.baseline_blocks) == 2


def test_selection_keeps_a_feasible_schedule(small_low_battery):
    _, cs = small_low_battery
    sel = select_columns(cs, budget=20)
    assert set(sel.anchor) <= set(sel.indices)
    assert check_schedule(cs, sel.anchor)["ok"]
    assert check_schedule(cs, solve_milp(cs, sel.indices).columns)["ok"]


def test_low_battery_needs_charging(small_low_battery):
    _, cs = small_low_battery
    sol = solve_milp(cs)
    assert any(cs.columns[k].charges for k in sol.columns)
    assert check_schedule(cs, sol.columns)["over_capacity"] == []


def test_conflicts_are_real_overlaps(small_low_battery):
    _, cs = small_low_battery
    idx = select_columns(cs, budget=40).indices
    conflicts, tight = cs.pairs(idx)
    for a, b in conflicts:
        ca, cb = cs.columns[idx[a]], cs.columns[idx[b]]
        assert any(x.station == y.station and x.start_min < y.end_min and y.start_min < x.end_min
                   for x in ca.charges for y in cb.charges)
    for a, b, r in tight:
        assert 0 < r <= cs.buffer_min


def test_power_profile_matches_energy(small_low_battery):
    _, cs = small_low_battery
    for col in cs.columns:
        kwh_from_power = sum(kw * cs.slot_min / 60 for prof in col.power_by_station.values() for kw in prof.values())
        assert kwh_from_power == pytest.approx(col.grid_kwh_day, rel=1e-6, abs=1e-6)
        assert sum(col.energy_by_slot.values()) == pytest.approx(col.grid_kwh_day, rel=1e-6, abs=1e-6)


def test_json_round_trip(tmp_path, small_low_battery):
    _, cs = small_low_battery
    idx = select_columns(cs, budget=20).indices
    path = cs.to_json(tmp_path / "cols.json", indices=idx)
    back = load_columns(path)
    assert [c.trips for c in back.columns] == [cs.columns[k].trips for k in idx]
    assert back.pairs(list(range(len(idx)))) == cs.pairs(idx)
    assert back.costs() == pytest.approx(cs.costs(idx), rel=1e-6)

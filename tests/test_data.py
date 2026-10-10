from pathlib import Path

import pytest

from evsp.data.gtfs_kcm import FT_PER_KM, haversine_km, hms_to_min
from evsp.instance import load_instance

ROOT = Path(__file__).resolve().parents[1]


def test_hms_past_midnight():
    assert hms_to_min("05:30:00") == 330
    assert hms_to_min("25:10:30") == pytest.approx(1510.5)


def test_haversine_one_degree_latitude():
    assert haversine_km(47.0, -122.0, 48.0, -122.0) == pytest.approx(111.2, rel=1e-3)


def test_feet_constant():
    assert FT_PER_KM == pytest.approx(3280.84)


@pytest.mark.parametrize("name", ["tiny", "small", "medium"])
def test_instance_is_consistent(name):
    path = ROOT / "instances" / f"kcm_{name}.json"
    if not path.exists():
        pytest.skip("run scripts/build_instances.py first")
    inst = load_instance(path)
    ids = {t.id for t in inst.trips}
    assert len(ids) == len(inst.trips)
    for t in inst.trips:
        assert t.arr_min > t.dep_min
        assert t.dist_km > 0 and t.energy_kwh > 0
        assert t.start in inst.locations and t.end in inst.locations
    # every baseline trip is in the instance and the real blocks pass the time check
    assert all(tid in ids for b in inst.baseline_blocks for tid in b["trips"])
    assert inst.check_baseline()["pairs_fail"] == 0
    # arcs only go forward in time (graph A is acyclic)
    tb = inst.trip_by_id
    assert all(tb[i].arr_min < tb[j].dep_min for i, j in inst.arcs())

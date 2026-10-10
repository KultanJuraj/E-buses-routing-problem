"""Small on-disk GTFS fixtures exercise the parser without external downloads."""
import pandas as pd
import pytest

from evsp.data.gtfs_kcm import (
    Assumptions, build_instance, haversine_km, load_trip_table, save_instance,
)
from evsp import load_instance


@pytest.fixture
def feed(tmp_path):
    tables = {
        "routes.txt": pd.DataFrame([dict(route_id="r", route_short_name="150")]),
        "trips.txt": pd.DataFrame([dict(route_id="r", service_id="86021", trip_id="t",
                                       direction_id="0", block_id="b", shape_id="s")]),
        "stop_times.txt": pd.DataFrame([
            dict(trip_id="t", arrival_time="25:00:00", departure_time="25:00:00",
                 stop_id="A", stop_sequence=1, shape_dist_traveled=0),
            dict(trip_id="t", arrival_time="25:10:00", departure_time="25:10:00",
                 stop_id="B", stop_sequence=2, shape_dist_traveled=3280.84),
            dict(trip_id="t", arrival_time="25:20:00", departure_time="25:20:00",
                 stop_id="C", stop_sequence=3, shape_dist_traveled=6561.68),
        ]),
        "shapes.txt": pd.DataFrame([
            dict(shape_id="s", shape_pt_lat=47.0, shape_pt_lon=-122.0, shape_pt_sequence=1),
            dict(shape_id="s", shape_pt_lat=47.01, shape_pt_lon=-122.0, shape_pt_sequence=2),
            dict(shape_id="s", shape_pt_lat=47.02, shape_pt_lon=-122.0, shape_pt_sequence=3),
        ]),
    }

    def write():
        for name, table in tables.items():
            table.to_csv(tmp_path / name, index=False)
        return tmp_path

    return tables, write


@pytest.mark.parametrize("missing_endpoint", [0, 2])
def test_missing_endpoint_distance_uses_geographical_fallback(feed, missing_endpoint):
    tables, write = feed
    tables["stop_times.txt"]["shape_dist_traveled"] = tables["stop_times.txt"]["shape_dist_traveled"].astype(float)
    tables["stop_times.txt"].loc[missing_endpoint, "shape_dist_traveled"] = float("nan")
    row = load_trip_table(write()).iloc[0]
    assert row["dist_km"] == pytest.approx(1.3 * haversine_km(47.0, -122.0, 47.02, -122.0))
    assert row["start_stop"] == "A"
    assert row["end_stop"] == "C"


def test_gtfs_to_validated_instance(feed, tmp_path):
    _, write = feed
    trips = load_trip_table(write())
    assert trips.iloc[0]["dist_km"] == pytest.approx(2)
    assert trips.iloc[0]["dep_min"] == 1500
    assumptions = Assumptions(kwh_per_km=1.5)
    raw = build_instance(trips, [{"name": "Candidate", "lat": 47.01, "lon": -122.0}], assumptions)
    inst = load_instance(save_instance(raw, tmp_path / "output.json"))
    assert inst.trips[0].energy_kwh == 3
    assert inst.assumptions["peak_scope"] == "global"
    assert all(c.availability == "assumed" for c in inst.chargers)
    assert inst.tariff[-1]["start_min"] >= inst.trips[0].arr_min
    assert inst.check_baseline()["pairs_fail"] == 0


def test_invalid_shape_endpoint_is_not_replaced_with_middle_point(feed):
    tables, write = feed
    tables["shapes.txt"].loc[0, "shape_pt_lon"] = float("nan")
    with pytest.raises(ValueError, match="Invalid GTFS.*t"):
        load_trip_table(write())


def test_impossible_connection_is_rejected(feed, tmp_path):
    _, write = feed
    trips = load_trip_table(write())
    second = trips.iloc[0].copy()
    second["trip_id"] = "t2"
    second["dep_min"] = 1510
    second["arr_min"] = 1530
    trips = pd.concat([trips, second.to_frame().T], ignore_index=True)
    inst = load_instance(save_instance(build_instance(trips, []), tmp_path / "overlap.json"))
    assert not inst.can_follow(inst.trips[0], inst.trips[1])
    assert inst.arcs() == []
    assert inst.check_baseline()["pairs_fail"] == 1

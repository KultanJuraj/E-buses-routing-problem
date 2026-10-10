import copy
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from evsp import load_instance, validate_instance
from evsp.data.gtfs_kcm import build_instance, hms_to_min, save_instance, select_trips

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw():
    return json.loads((ROOT / "instances/kcm_tiny.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("path,value,expected", [
    (("schema_version",), "2.0", "schema_version"),
    (("trips", 0, "start"), "missing", "trips[0].start"),
    (("trips", 0, "arr_min"), 0, "trips[0].arr_min"),
    (("trips", 0, "dist_km"), -1, "dist_km"),
    (("trips", 0, "energy_kwh"), 99999, "energy_kwh"),
    (("trips", 0, "energy_kwh"), float("nan"), "energy_kwh"),
    (("chargers", 0, "power_kw"), 0, "power_kw"),
    (("chargers", 0, "location"), "missing", "chargers[0].location"),
    (("assumptions", "soc_start"), 0, "soc_start"),
    (("assumptions", "soc_min"), 1, "soc_max"),
    (("assumptions", "charge_efficiency"), 1.1, "charge_efficiency"),
    (("assumptions", "slot_min"), 0, "slot_min"),
    (("assumptions", "peak_scope"), "station", "peak_scope"),
    (("tariff", 1, "start_min"), 0, "tariff[1]"),
    (("baseline_blocks", 0, "trips", 0), "missing", "baseline_blocks[0].trips[0]"),
])
def test_invalid_values_have_path(raw, path, value, expected):
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError) as error:
        validate_instance(raw)
    assert expected in str(error.value)


@pytest.mark.parametrize("field", ["trips", "chargers", "locations", "baseline_blocks"])
def test_duplicate_ids_rejected_before_loading(raw, field):
    raw[field].append(copy.deepcopy(raw[field][0]))
    with pytest.raises(ValueError, match="duplicate ID"):
        validate_instance(raw)


@pytest.mark.parametrize("kind", ["row", "column", "diagonal", "negative", "infinite"])
def test_invalid_deadhead(raw, kind):
    origin, destination = list(raw["deadhead"])[:2]
    if kind == "row":
        del raw["deadhead"][origin]
    elif kind == "column":
        del raw["deadhead"][origin][destination]
    elif kind == "diagonal":
        raw["deadhead"][origin][origin]["km"] = 1
    else:
        raw["deadhead"][origin][destination]["min"] = -1 if kind == "negative" else float("inf")
    with pytest.raises(ValueError, match="deadhead"):
        validate_instance(raw)


def test_missing_version_and_unknown_fields(raw):
    del raw["schema_version"]
    with pytest.raises(ValueError, match="schema_version"):
        validate_instance(raw)
    raw["schema_version"] = "1.0"
    raw["trips"][0]["energy_kw"] = 1
    with pytest.raises(ValueError, match="energy_kw"):
        validate_instance(raw)


def test_invalid_json_reports_file_and_line(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"trips":', encoding="utf-8")
    with pytest.raises(ValueError, match="broken.json: invalid JSON at line"):
        load_instance(path)


def test_round_trip_and_deadhead_energy(raw, tmp_path):
    path = save_instance(raw, tmp_path / "instance.json")
    first = load_instance(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    second = load_instance(save_instance(saved, tmp_path / "second.json"))
    assert saved == raw
    assert asdict(first) == asdict(second)
    for origin in first.locations:
        for destination in first.locations:
            assert first.dh_energy_kwh(origin, destination) == pytest.approx(
                first.dh_km(origin, destination) * first.assumptions["kwh_per_km"])
    assert first.dh_energy_kwh(first.depot, first.depot) == 0


def test_invalid_save_does_not_overwrite(raw, tmp_path):
    path = save_instance(raw, tmp_path / "instance.json")
    before = path.read_bytes()
    raw["trips"][0]["energy_kwh"] = -1
    with pytest.raises(ValueError):
        save_instance(raw, path)
    assert path.read_bytes() == before


def test_empty_selection():
    empty = select_trips(pd.DataFrame({"route": ["150"], "dep_min": [0], "trip_id": ["1"]}), routes=["999"])
    with pytest.raises(ValueError, match="No trips selected"):
        build_instance(empty, [])


@pytest.mark.parametrize("value", ["-1:00:00", "24:60:00", "12:00:60", "invalid"])
def test_invalid_times(value):
    with pytest.raises(ValueError):
        hms_to_min(value)


def test_large_diagnostics_use_same_rules_as_arcs():
    inst = load_instance(ROOT / "instances/kcm_large.json")
    report = inst.check_baseline()
    arcs = set(inst.arcs())
    expected = {(i, j) for block in inst.baseline_blocks
                for i, j in zip(block["trips"], block["trips"][1:]) if (i, j) not in arcs}
    actual = {(f["from_trip"], f["to_trip"]) for f in report["failures"]}
    assert actual == expected
    assert len(actual) == report["pairs_fail"] == 49
    assert inst.check_baseline(min_layover=0)["pairs_fail"] == 34
    for failure in report["failures"]:
        assert failure["shortfall_min"] > 0
        assert failure["shortfall_min"] == pytest.approx(failure["required_min"] - failure["available_min"])

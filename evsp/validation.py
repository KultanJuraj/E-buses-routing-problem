"""Versioned structural and semantic checks, before IDs become dictionary keys."""
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_VERSION = "1.0"
_SCHEMA = json.loads(Path(__file__).with_name("instance.schema.json").read_text(encoding="utf-8"))
_VALIDATOR = Draft202012Validator(_SCHEMA)


def validate_instance(raw: dict) -> None:
    """Raise ValueError with a JSON path for the first invalid value.

    Baseline time infeasibility is diagnostic, not a schema error. Directed road
    matrices are allowed; symmetry and a particular travel speed are not required.
    """
    def fail(path, message):
        raise ValueError(f"{path}: {message}")

    def finite(value, path="$"):
        if isinstance(value, float) and not math.isfinite(value):
            fail(path, "must be finite")
        if isinstance(value, dict):
            for key, child in value.items():
                finite(child, f"{path}[{key!r}]")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                finite(child, f"{path}[{index}]")

    finite(raw)
    error = next(_VALIDATOR.iter_errors(raw), None)
    if error:
        path = "$" + "".join(f"[{part!r}]" for part in error.absolute_path)
        fail(path, error.message)

    def unique(rows, key, path):
        seen = set()
        for index, row in enumerate(rows):
            if row[key] in seen:
                fail(f"{path}[{index}].{key}", f"duplicate ID {row[key]!r}")
            seen.add(row[key])
        return seen

    locations = unique(raw["locations"], "id", "$.locations")
    unique(raw["trips"], "id", "$.trips")
    unique(raw["chargers"], "id", "$.chargers")
    unique(raw["baseline_blocks"], "block_id", "$.baseline_blocks")
    if sum(l["type"] == "depot" for l in raw["locations"]) != 1:
        fail("$.locations", "exactly one depot is required")
    a = raw["assumptions"]
    if not a["soc_min"] < a["soc_max"]:
        fail("$.assumptions.soc_max", "must exceed soc_min")
    if not a["soc_min"] <= a["soc_start"] <= a["soc_max"]:
        fail("$.assumptions.soc_start", "must be between soc_min and soc_max")
    for i, trip in enumerate(raw["trips"]):
        for key in ("start", "end"):
            if trip[key] not in locations:
                fail(f"$.trips[{i}].{key}", "unknown location")
        if trip["arr_min"] <= trip["dep_min"]:
            fail(f"$.trips[{i}].arr_min", "must exceed dep_min")
        # Both distance and energy are independently rounded to three decimals.
        tolerance = 0.0005 * (1 + a["kwh_per_km"]) + 1e-9
        if abs(trip["energy_kwh"] - trip["dist_km"] * a["kwh_per_km"]) > tolerance:
            fail(f"$.trips[{i}].energy_kwh", "inconsistent with dist_km * kwh_per_km")
    for i, charger in enumerate(raw["chargers"]):
        if charger["location"] not in locations:
            fail(f"$.chargers[{i}].location", "unknown location")
    dh = raw["deadhead"]
    if set(dh) != locations:
        fail("$.deadhead", "rows must match all location IDs")
    for origin, row in dh.items():
        if set(row) != locations:
            fail(f"$.deadhead[{origin!r}]", "columns must match all location IDs")
        if row[origin] != {"km": 0, "min": 0}:
            fail(f"$.deadhead[{origin!r}][{origin!r}]", "diagonal must be zero")
        for destination, leg in row.items():
            if leg["km"] > 0 and leg["min"] == 0:
                fail(f"$.deadhead[{origin!r}][{destination!r}].min", "positive distance requires positive travel time")
    for i, slot in enumerate(raw["tariff"]):
        if slot["t"] != i or slot["start_min"] != i * a["slot_min"]:
            fail(f"$.tariff[{i}]", "slots must be contiguous and ordered from midnight")
    if len(raw["tariff"]) * a["slot_min"] < max(t["arr_min"] for t in raw["trips"]):
        fail("$.tariff", "does not cover the last trip arrival")
    trips = {t["id"]: t for t in raw["trips"]}
    seen = set()
    for i, block in enumerate(raw["baseline_blocks"]):
        previous = -1
        for j, tid in enumerate(block["trips"]):
            path = f"$.baseline_blocks[{i}].trips[{j}]"
            if tid not in trips:
                fail(path, "unknown trip")
            if tid in seen:
                fail(path, "trip occurs in multiple baseline blocks")
            seen.add(tid)
            trip = trips[tid]
            if trip["block_id"] != block["block_id"]:
                fail(path, "block_id differs from trip.block_id")
            if trip["dep_min"] < previous:
                fail(path, "trips must be ordered by departure")
            previous = trip["dep_min"]
    if seen != set(trips):
        fail("$.baseline_blocks", "must cover every trip exactly once")

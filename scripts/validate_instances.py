"""Validate existing JSON instances and write all baseline time failures."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evsp.instance import load_instance  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, default=ROOT / "instances")
    args = parser.parse_args()
    reports = []
    paths = sorted(args.instances.glob("kcm_*.json"))
    if not paths:
        parser.error(f"No kcm_*.json instances found in {args.instances}")
    for path in paths:
        inst = load_instance(path)
        report = inst.check_baseline()
        reports.append(report)
        print(f"{inst.name}: valid schema, {report['pairs_fail']} infeasible baseline pairs "
              f"with {report['min_layover_min']} min layover")
    output = args.instances / "baseline_validation.json"
    output.write_text(json.dumps({"instances": reports}, indent=2, allow_nan=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

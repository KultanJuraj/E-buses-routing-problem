import json
import math

import main


def test_result_log_handles_unsampled_feasible_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    path = main.save_results({"daily_schedule": {"qaoa": {"best_feasible_energy": math.inf}}}, "k4", 360)
    assert path.name == "main_bus_k4_0600.json"
    assert json.loads(path.read_text(encoding="utf-8"))["daily_schedule"]["qaoa"]["best_feasible_energy"] is None

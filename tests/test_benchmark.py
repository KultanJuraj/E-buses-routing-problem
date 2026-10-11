from pathlib import Path

import numpy as np
import pytest

from evsp.benchmark import SampleDistribution, best_of_n, repair_costs, run_quality_benchmark, sample_distributions
from evsp.classical_baselines import exact_repair
from evsp.disruption import Disruption, build_repair_problem

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def event():
    from evsp.demo import DEFAULT_COLUMNS, load_demo_data

    if not DEFAULT_COLUMNS.exists():
        pytest.skip("run build_columns.py --battery 200 first")
    d = load_demo_data()
    rp = build_repair_problem(d.inst, d.schedule, Disruption.bus_failure(d.schedule, "k585", 7 * 60), reserves=3)
    return d.inst, rp


def test_best_of_n_formulas():
    dist = SampleDistribution("toy", values=np.array([1.0, 2.0, 5.0]), probs=np.array([0.2, 0.3, 0.5]),
                              p_valid=1.0, p_optimal=0.2)
    one = best_of_n(dist, 1, optimum=1.0)
    assert one["expected_best_cost"] == pytest.approx(0.2 * 1 + 0.3 * 2 + 0.5 * 5)
    two = best_of_n(dist, 2, optimum=1.0)
    # P(min=5) = .25, P(min>=2) = .64, P(min=2) = .64-.25, P(min=1) = 1-.64
    assert two["expected_best_cost"] == pytest.approx(0.36 * 1 + 0.39 * 2 + 0.25 * 5)
    assert two["p_found_optimum"] == pytest.approx(1 - 0.8 ** 2)
    costs = [best_of_n(dist, n, 1.0)["expected_best_cost"] for n in (1, 2, 5, 20)]
    assert costs == sorted(costs, reverse=True)


def test_best_of_n_matches_monte_carlo(event):
    inst, rp = event
    rc, dists = sample_distributions(inst, rp, p_max=1, sa_reads=300, penalty_max_n=0)
    rng = np.random.default_rng(0)
    for dist in dists[:2]:
        sims = [dist.values[rng.choice(len(dist.values), 10, p=dist.probs)].min() for _ in range(4000)]
        assert np.mean(sims) == pytest.approx(best_of_n(dist, 10, rc.optimum)["expected_best_cost"], rel=0.02)


def test_true_costs_and_distributions(event):
    inst, rp = event
    rc = repair_costs(inst, rp)
    _, exact = exact_repair(inst, rp)
    assert rc.optimum == pytest.approx(exact.extra_cost)
    assert rc.fallback > rc.optimum
    _, dists = sample_distributions(inst, rp, p_max=1, sa_reads=200, rc=rc)
    names = [d.method for d in dists]
    assert names[0].startswith("XY") and any(n.startswith("penalty") for n in names)
    for d in dists:
        assert d.probs.sum() == pytest.approx(1.0) and 0 <= d.p_optimal <= d.p_valid <= 1 + 1e-9
    xy = dists[0]
    assert xy.p_valid == pytest.approx(1.0)                     # never leaves the valid assignments
    pen = next(d for d in dists if d.method.startswith("penalty"))
    assert xy.p_valid > pen.p_valid and xy.p_optimal > pen.p_optimal


def test_run_quality_benchmark(event):
    inst, rp = event
    r = run_quality_benchmark(inst, rp, budgets=(1, 100), p_max=1, sa_reads=200)
    assert r["qubits"] == rp.n and r["greedy"] >= r["true_optimum"] - 1e-6
    rows = {(x["method"], x["n"]): x for x in r["best_of_n"]}
    for (m, n), x in rows.items():
        if n == 100:
            assert x["expected_gap"] <= rows[(m, 1)]["expected_gap"] + 1e-9

from pathlib import Path

import pytest

pytest.importorskip("plotly")

from evsp.demo import DEFAULT_COLUMNS, DEFAULT_INSTANCE, load_demo_data, solve_disruption  # noqa: E402
from evsp.disruption import Disruption  # noqa: E402
from evsp.viz import (describe_sk, gantt_figure, hhmm, method_figure, power_figure, scaling_figure,  # noqa: E402
                      schedule_kpis, soc_figure, soc_timeline)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data():
    if not DEFAULT_COLUMNS.exists() or not DEFAULT_INSTANCE.exists():
        pytest.skip("run build_instances.py and build_columns.py --battery 200 first")
    return load_demo_data()


@pytest.fixture(scope="module")
def report(data):                     # 14 qubits: penalty QAOA is computed live
    return solve_disruption(data, Disruption.bus_failure(data.schedule, "k585", 7 * 60), p_max=1)


def test_hhmm():
    assert hhmm(0) == "00:00" and hhmm(7 * 60 + 5) == "07:05" and hhmm(59.6) == "01:00"


def test_soc_timeline_matches_columns(data):
    for bus, col in data.schedule.duties.items():
        t, s = soc_timeline(data.inst, col)
        assert t == sorted(t), bus
        assert t[0] == col.start_min and t[-1] == col.end_min
        assert s[-1] == pytest.approx(col.final_soc, abs=1e-3), bus
        assert min(s) == pytest.approx(col.min_soc, abs=1e-3), bus


def test_schedule_kpis(data):
    k = schedule_kpis(data.schedule.duties)
    assert k["buses"] == 5 and k["trips"] == len(data.inst.trips)
    assert k["cost"] == pytest.approx(sum(c.cost for c in data.schedule.duties.values()))
    assert k["peak_kw"] > 0 and k["charging_stops"] > 0


def test_schedule_figures(data):
    duties = data.schedule.duties
    g = gantt_figure(data.inst, duties)
    names = {tr.name for tr in g.data}
    assert {"spoj", "nabíjanie", "prázdny presun"} <= names
    n_trips = sum(1 for tr in g.data if tr.name == "spoj")
    assert n_trips == len(data.inst.trips)
    assert len(soc_figure(data.inst, duties).data) == len(duties)
    p = power_figure(duties)
    assert {tr.name for tr in p.data} == {ch.station for c in duties.values() for ch in c.charges}
    assert any("špička" in a.text for a in p.layout.annotations)
    assert power_figure({}).layout.annotations[0].text.startswith("žiadne")


def test_repair_report(data, report):
    rp = report.problem
    assert report.penalty_source == "computed"
    assert set(report.plan.assignment) == set(rp.trips)
    assert set(report.moved) | set(report.cancelled) <= set(rp.trips)
    assert all(b != "k585" for b in report.moved.values())
    # failed bus keeps only what it drove before 07:00; every surviving trip is driven exactly once
    tb = data.inst.trip_by_id
    assert all(tb[t].dep_min < 7 * 60 for t in report.after["k585"].trips)
    driven = [t for c in report.after.values() if c is not None for t in c.trips]
    assert len(driven) == len(set(driven))
    assert set(driven) | set(report.cancelled) == {t.id for t in data.inst.trips}
    rows = {r["method"]: r for r in report.rows}
    assert rows["QAOA, XY mixer"]["p_feasible"] > rows["QAOA, X mixer + pokuty"]["p_feasible"]
    assert rows["QAOA, XY mixer"]["p_optimal"] > rows["QAOA, X mixer + pokuty"]["p_optimal"]


def test_repair_figures(data, report):
    ev = report.event
    g = gantt_figure(data.inst, report.after, failed={ev.bus: ev.at_min}, moved=report.moved,
                     cancelled=report.cancelled)
    names = [tr.name for tr in g.data]
    assert "výpadok" in names
    assert names.count("presunutý spoj") == len(report.moved)
    assert names.count("zrušený spoj") == len(report.cancelled)
    m = method_figure(report.rows)
    assert len(m.data) == 2 and len(m.data[0].y) == len(m.data[1].y) == 3
    assert describe_sk(ev) == "porucha autobusu k585 o 07:00"
    out = gantt_figure(data.inst, report.before, outage=("S3", 420, 540))
    assert out.layout.shapes and out.layout.shapes[0].x0 == 420


def test_large_or_empty_events_are_rejected(data):
    with pytest.raises(ValueError):
        solve_disruption(data, Disruption.bus_failure(data.schedule, "k4", 22 * 60), penalty_qaoa="never")
    with pytest.raises(ValueError):
        solve_disruption(data, Disruption.charger_outage("S3", 420, 540), penalty_qaoa="never", max_states=10)


def test_scaling_figure():
    rows = [{"qubits": n, "sa_p_optimal": 0.5, "best_p_optimal": 0.1 / n, "uniform_p_optimal": 2.0 ** -n}
            for n in (8, 10, 12)]
    fig = scaling_figure(rows)
    assert len(fig.data) == 3 and fig.layout.yaxis.type == "log"


def test_streamlit_app(data):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "demo.py"), default_timeout=120).run()
    assert not at.exception
    assert [m.value for m in at.metric][:2] == ["5", "20"]
    at.selectbox[0].set_value("k585")
    at.select_slider[0].set_value(7 * 60)
    at.slider[2].set_value(1)                                  # p = 1 keeps the test fast
    [b for b in at.button if b.key == "go"][0].click().run()
    assert not at.exception and not at.error
    assert [h.value for h in at.subheader] == ["porucha autobusu k585 o 07:00"]
    assert len(at.success) + len(at.warning) == 1              # verdict on the XY-QAOA answer
    assert len(at.get("plotly_chart")) >= 6

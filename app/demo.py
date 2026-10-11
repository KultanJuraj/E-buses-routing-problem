"""Interactive demo (package G): the optimal e-bus schedule and a live disruption repair.

    pip install -r requirements.txt
    streamlit run app/demo.py

Tab "Schedule": the 5-bus schedule of columns/kcm_small_b20_bat200.json (trips, charging,
battery, grid load). Tab "Disruption": pick a bus and a time (or a charger and a window),
press "Break the bus" and the repair subproblem is solved exactly, with simulated
annealing, with XY-mixer QAOA and (up to 16 qubits, or from results/) with penalty QAOA.
Tab "Scaling": penalty QAOA on growing E-VSP instances (results/*_scaling.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from evsp.demo import DEFAULT_COLUMNS, DEFAULT_INSTANCE, load_demo_data, peak_schedules, solve_disruption  # noqa: E402
from evsp.disruption import Disruption  # noqa: E402
from evsp.viz import (describe_event, gantt_figure, hhmm, method_figure, peak_comparison_figure, power_figure,  # noqa: E402
                      scaling_figure, schedule_kpis, soc_figure)

st.set_page_config(page_title="E-bus schedule and quantum repair", layout="wide")


def chart(fig, key: str, where=st):
    # theme=None keeps the figure's own validated colors (Streamlit's theme would restyle them)
    where.plotly_chart(fig, key=key, theme=None)


@st.cache_resource(show_spinner="Loading the instance and duties…")
def data(columns: str, instance: str):
    return load_demo_data(Path(columns), Path(instance))


@st.cache_resource(show_spinner="Comparing the optima for R=0 and R>0…")
def peak_data(columns: str, instance: str):
    return peak_schedules(data(columns, instance))


@st.cache_resource(show_spinner="Repairing the schedule (exact, SA, XY-QAOA, penalty QAOA)…")
def repair(columns: str, instance: str, kind: str, who: str, a: int, b: int, reserves: int, options: int,
           p_max: int, penalty: str):
    d = data(columns, instance)
    ev = Disruption.bus_failure(d.schedule, who, a) if kind == "bus" else Disruption.charger_outage(who, a, b)
    return solve_disruption(d, ev, reserves=reserves, max_bus_options=options, p_max=p_max, penalty_qaoa=penalty)


def schedule_table(inst, duties) -> pd.DataFrame:
    rows = []
    for bus, c in sorted(duties.items(), key=lambda kv: kv[1].start_min):
        rows.append({"bus": bus, "pull-out": hhmm(c.start_min), "pull-in": hhmm(c.end_min), "trips": c.n_trips,
                     "charging": ", ".join(f"{ch.station} {hhmm(ch.start_min)}-{hhmm(ch.end_min)}"
                                           for ch in c.charges) or "-",
                     "deadhead km": round(c.deadhead_km, 1), "min SoC": f"{c.min_soc:.0%}",
                     "final SoC": f"{c.final_soc:.0%}", "cost": round(c.cost, 1)})
    return pd.DataFrame(rows)


def kpi_row(k: dict, ref: dict | None = None):
    cols = st.columns(5)
    items = [("buses", "buses", "{:.0f}"), ("trips", "trips", "{:.0f}"), ("deadhead km", "deadhead_km", "{:.0f}"),
             ("charging peak [kW]", "peak_kw", "{:.0f}"), ("cost", "cost", "{:.0f}")]
    for c, (label, key, fmt) in zip(cols, items):
        delta = None if ref is None else fmt.format(k[key] - ref[key])
        c.metric(label, fmt.format(k[key]), delta=delta, delta_color="inverse", border=True)


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Data")
    columns = st.text_input("duties (columns JSON)", str(DEFAULT_COLUMNS.relative_to(ROOT)))
    instance = st.text_input("instance", str(DEFAULT_INSTANCE.relative_to(ROOT)))
    columns_path, instance_path = str(ROOT / columns), str(ROOT / instance)
    if not Path(columns_path).exists() or not Path(instance_path).exists():
        st.error("File not found. Run scripts/build_instances.py and scripts/build_columns.py --battery 200.")
        st.stop()
    d = data(columns_path, instance_path)
    a = d.inst.assumptions
    st.caption(f"{len(d.inst.trips)} trips, {len(d.cs.columns)} duties in the pool, battery {a['battery_kwh']:.0f} kWh, "
               f"min SoC {a['soc_min']:.0%}")

st.title("Electric buses: schedule and repair after a disruption")
tab_plan, tab_fail, tab_scale = st.tabs(["Schedule", "Disruption", "Scaling"])

# --------------------------------------------------------------------------- schedule
with tab_plan:
    duties = d.schedule.duties
    kpi_row(schedule_kpis(duties))
    chart(gantt_figure(d.inst, duties, title="Optimal schedule (QUBO E4 / MILP)"), "g0")
    left, right = st.columns(2)
    chart(soc_figure(d.inst, duties), "soc0", left)
    chart(power_figure(duties), "pow0", right)
    peaks = peak_data(columns_path, instance_path)
    chart(peak_comparison_figure(peaks), "peak_compare")
    if set(peaks["R=0"]) == set(peaks["R>0"]):
        st.caption("This duty pool gives the same schedule for R=0 and R>0; "
                   "the peak does not drop at the chosen weight.")
    st.dataframe(schedule_table(d.inst, duties), hide_index=True)

# --------------------------------------------------------------------------- disruption
with tab_fail:
    stations = sorted({ch.station for c in d.schedule.duties.values() for ch in c.charges})
    c1, c2, c3 = st.columns([1.2, 2, 1.4])
    kind = c1.radio("event", ["bus failure", "charger outage"], key="kind")
    times = list(range(5 * 60, 22 * 60 + 1, 15))
    if kind == "bus failure":
        buses = sorted(d.schedule.duties, key=lambda b: d.schedule.duties[b].start_min)
        who = c2.selectbox("bus", buses, index=buses.index("k4") if "k4" in buses else 0)
        at = c2.select_slider("failure time", times, value=6 * 60, format_func=hhmm)
        ev_args = ("bus", who, at, at)
    else:
        who = c2.selectbox("station", stations, index=stations.index("S3") if "S3" in stations else 0)
        a_, b_ = c2.select_slider("outage window", times, value=(7 * 60, 9 * 60), format_func=hhmm)
        ev_args = ("charger", who, a_, b_)
    reserves = c3.slider("reserve buses", 0, 3, 3)
    options = c3.slider("other buses per trip", 1, 3, 3)
    p_max = c3.slider("QAOA depth p", 1, 3, 3)
    penalty = "auto" if c3.checkbox("compare with penalty QAOA", value=True) else "never"

    label = "Break the bus" if kind == "bus failure" else "Switch off the charger"
    if st.button(label, type="primary", icon="⚡", key="go"):
        st.session_state["event"] = (*ev_args, reserves, options, p_max, penalty)
    ev = st.session_state.get("event")
    if ev is None:
        st.info(f"Choose an event and press **{label}**.")
    else:
        try:
            rep = repair(columns_path, instance_path, *ev)
        except ValueError as e:
            if "no trips are affected" in str(e):
                st.info("No trip is affected: the bus has no more trips, or its charging can simply move "
                        "to another station.")
            else:
                st.warning(f"This event cannot be solved live: {e}")
            st.stop()
        e = rep.event
        st.subheader(describe_event(e))
        st.write(f"**{len(rep.problem.trips)} affected trips → {rep.problem.n} qubits**, "
                 f"{rep.space.size:,} valid one-hot assignments out of 2^{rep.problem.n} = "
                 f"{2 ** rep.problem.n:,} bitstrings.")
        kpi_row(schedule_kpis(rep.after), schedule_kpis(rep.before))
        failed = {e.bus: e.at_min} if e.kind == "bus" else None
        outage = (e.station, e.at_min, e.until_min) if e.kind == "charger" else None
        left, right = st.columns(2)
        chart(gantt_figure(d.inst, rep.before, title="Before the event", failed=failed, outage=outage), "gb", left)
        chart(gantt_figure(d.inst, rep.after, title="After the repair (optimum)", failed=failed,
                           moved=rep.moved, cancelled=rep.cancelled, outage=outage), "ga", right)
        moved = ", ".join(f"{t} → {b}" for t, b in rep.moved.items()) or "none"
        st.write(f"Reassigned trips: {moved}. Cancelled: {', '.join(rep.cancelled) or 'none'}. "
                 f"Extra cost {rep.plan.extra_cost:.0f}.")
        if rep.valid_rank is None:
            st.warning("No cheap assignment passed the full schedule check: all affected trips are cancelled "
                       "as the fallback.")
        elif rep.valid_rank:
            st.caption(f"The {rep.valid_rank} cheapest assignments of the pairwise model failed the full check.")
        verdict = "is the optimal repair" if rep.xy_is_optimal else \
            f"is not the optimum (cost {rep.xy_plan.extra_cost:.0f})"
        if rep.xy_is_optimal:
            st.success(f"The most likely XY-QAOA answer {verdict}.")
        else:
            st.warning(f"The most likely XY-QAOA answer {verdict}.")

        chart(method_figure(rep.rows), "m")
        def cell(v, fmt):
            return "–" if v is None else fmt.format(v)

        table = pd.DataFrame([{"method": r["method"], "p": cell(r["p"], "{}"),
                               "P(optimum)": cell(r["p_optimal"], "{:.4f}"),
                               "P(valid)": cell(r["p_feasible"], "{:.3f}"),
                               "shots for 95 %": cell(r["shots_95"], "{:.0f}")}
                              for r in rep.rows])
        st.dataframe(table, hide_index=True)
        notes = {"computed": "Penalty QAOA computed live.", "skipped": "Penalty QAOA switched off."}
        st.caption(notes.get(rep.penalty_source, f"Penalty QAOA: {rep.penalty_source}.")
                   + " P(optimum) of simulated annealing = share of 200 runs; QAOA = exact probability "
                     "from the statevector.")
        ca, ch = rep.circuits["xy_all"], rep.circuits["xy_heavy_hex"]
        st.caption(f"XY-QAOA circuit: {ca['qubits']} qubits, depth {ca['depth']}, {ca['cx']} CX "
                   f"(all-to-all); heavy-hex: {ch['physical_qubits']} physical qubits, depth {ch['depth']}, "
                   f"{ch['cx']} CX. Time: " + ", ".join(f"{k} {v:.1f}s" for k, v in rep.seconds.items()))

# --------------------------------------------------------------------------- scaling
with tab_scale:
    files = sorted((ROOT / "results").glob("*_scaling.json"))
    if not files:
        st.info("No scaling results yet. Run for example\n\n"
                "`python scripts/run_qaoa.py --columns columns/kcm_tiny_b20.json --variant E2 --scaling 8,10,12,16`")
    else:
        f = st.selectbox("file", files, format_func=lambda p: p.name)
        rows = json.loads(f.read_text(encoding="utf-8"))["scaling"]
        chart(scaling_figure(rows), "sc")
        flat = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows])
        st.dataframe(flat, hide_index=True)
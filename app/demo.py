"""Interactive demo (package G): the optimal e-bus schedule and a live disruption repair.

    pip install -r requirements.txt
    streamlit run app/demo.py

Tab "Rozvrh": the 5-bus schedule of columns/kcm_small_b20_bat200.json (trips, charging,
battery, grid load). Tab "Výpadok": pick a bus and a time (or a charger and a window),
press "Pokaz autobus" and the repair subproblem is solved exactly, with simulated
annealing, with XY-mixer QAOA and (up to 16 qubits, or from results/) with penalty QAOA.
Tab "Škálovanie": penalty QAOA on growing E-VSP instances (results/*_scaling.json).
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

from evsp.demo import DEFAULT_COLUMNS, DEFAULT_INSTANCE, load_demo_data, solve_disruption  # noqa: E402
from evsp.disruption import Disruption  # noqa: E402
from evsp.viz import (describe_sk, gantt_figure, hhmm, method_figure, power_figure, scaling_figure,  # noqa: E402
                      schedule_kpis, soc_figure)

st.set_page_config(page_title="E-bus rozvrh a kvantová oprava", layout="wide")


@st.cache_resource(show_spinner="Načítavam inštanciu a turnusy…")
def data(columns: str, instance: str):
    return load_demo_data(Path(columns), Path(instance))


@st.cache_resource(show_spinner="Riešim opravu rozvrhu (presne, SA, XY-QAOA, QAOA s pokutami)…")
def repair(columns: str, instance: str, kind: str, who: str, a: int, b: int, reserves: int, options: int,
           p_max: int, penalty: str):
    d = data(columns, instance)
    ev = Disruption.bus_failure(d.schedule, who, a) if kind == "bus" else Disruption.charger_outage(who, a, b)
    return solve_disruption(d, ev, reserves=reserves, max_bus_options=options, p_max=p_max, penalty_qaoa=penalty)


def schedule_table(inst, duties) -> pd.DataFrame:
    rows = []
    for bus, c in sorted(duties.items(), key=lambda kv: kv[1].start_min):
        rows.append({"autobus": bus, "výjazd": hhmm(c.start_min), "návrat": hhmm(c.end_min), "spoje": c.n_trips,
                     "nabíjania": ", ".join(f"{ch.station} {hhmm(ch.start_min)}-{hhmm(ch.end_min)}"
                                            for ch in c.charges) or "-",
                     "prázdne km": round(c.deadhead_km, 1), "min SoC": f"{c.min_soc:.0%}",
                     "SoC na konci": f"{c.final_soc:.0%}", "náklad": round(c.cost, 1)})
    return pd.DataFrame(rows)


def kpi_row(k: dict, ref: dict | None = None):
    cols = st.columns(5)
    items = [("autobusy", "buses", "{:.0f}"), ("spoje", "trips", "{:.0f}"), ("prázdne km", "deadhead_km", "{:.0f}"),
             ("špička nabíjania [kW]", "peak_kw", "{:.0f}"), ("náklad", "cost", "{:.0f}")]
    for c, (label, key, fmt) in zip(cols, items):
        delta = None if ref is None else fmt.format(k[key] - ref[key])
        c.metric(label, fmt.format(k[key]), delta=delta, delta_color="inverse", border=True)


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Dáta")
    columns = st.text_input("turnusy (columns JSON)", str(DEFAULT_COLUMNS.relative_to(ROOT)))
    instance = st.text_input("inštancia", str(DEFAULT_INSTANCE.relative_to(ROOT)))
    columns_path, instance_path = str(ROOT / columns), str(ROOT / instance)
    if not Path(columns_path).exists() or not Path(instance_path).exists():
        st.error("Súbor neexistuje. Spusti scripts/build_instances.py a scripts/build_columns.py --battery 200.")
        st.stop()
    d = data(columns_path, instance_path)
    a = d.inst.assumptions
    st.caption(f"{len(d.inst.trips)} spojov, {len(d.cs.columns)} turnusov v pule, batéria {a['battery_kwh']:.0f} kWh, "
               f"min SoC {a['soc_min']:.0%}")

st.title("Elektrobusy: rozvrh a oprava po výpadku")
tab_plan, tab_fail, tab_scale = st.tabs(["Rozvrh", "Výpadok", "Škálovanie"])

# --------------------------------------------------------------------------- schedule
with tab_plan:
    duties = d.schedule.duties
    kpi_row(schedule_kpis(duties))
    st.plotly_chart(gantt_figure(d.inst, duties, title="Optimálny rozvrh (QUBO E4 / MILP)"), key="g0")
    left, right = st.columns(2)
    left.plotly_chart(soc_figure(d.inst, duties), key="soc0")
    right.plotly_chart(power_figure(duties), key="pow0")
    st.dataframe(schedule_table(d.inst, duties), hide_index=True)

# --------------------------------------------------------------------------- disruption
with tab_fail:
    stations = sorted({ch.station for c in d.schedule.duties.values() for ch in c.charges})
    c1, c2, c3 = st.columns([1.2, 2, 1.4])
    kind = c1.radio("udalosť", ["porucha autobusu", "výpadok nabíjačky"], key="kind")
    times = list(range(5 * 60, 22 * 60 + 1, 15))
    if kind == "porucha autobusu":
        buses = sorted(d.schedule.duties, key=lambda b: d.schedule.duties[b].start_min)
        who = c2.selectbox("autobus", buses, index=buses.index("k4") if "k4" in buses else 0)
        at = c2.select_slider("čas poruchy", times, value=6 * 60, format_func=hhmm)
        ev_args = ("bus", who, at, at)
    else:
        who = c2.selectbox("stanica", stations, index=stations.index("S3") if "S3" in stations else 0)
        a_, b_ = c2.select_slider("okno výpadku", times, value=(7 * 60, 9 * 60), format_func=hhmm)
        ev_args = ("charger", who, a_, b_)
    reserves = c3.slider("záložné autobusy", 0, 3, 3)
    options = c3.slider("iné autobusy na spoj", 1, 3, 3)
    p_max = c3.slider("hĺbka QAOA p", 1, 3, 3)
    penalty = "auto" if c3.checkbox("porovnať s QAOA s pokutami", value=True) else "never"

    label = "Pokaz autobus" if kind == "porucha autobusu" else "Vypni nabíjačku"
    if st.button(label, type="primary", icon="⚡", key="go"):
        st.session_state["event"] = (*ev_args, reserves, options, p_max, penalty)
    ev = st.session_state.get("event")
    if ev is None:
        st.info(f"Vyber udalosť a stlač **{label}**.")
    else:
        try:
            rep = repair(columns_path, instance_path, *ev)
        except ValueError as e:
            if "no trips are affected" in str(e):
                st.info("Žiadny spoj nie je ohrozený: autobus už nemá ďalšie spoje, alebo stačí presmerovať "
                        "nabíjanie na inú stanicu.")
            else:
                st.warning(f"Túto udalosť sa nedá riešiť naživo: {e}")
            st.stop()
        e = rep.event
        st.subheader(describe_sk(e))
        st.write(f"**{len(rep.problem.trips)} ohrozených spojov → {rep.problem.n} qubitov**, "
                 f"{rep.space.size:,} prípustných one-hot priradení z 2^{rep.problem.n} = "
                 f"{2 ** rep.problem.n:,} bitových reťazcov.".replace(",", " "))
        kpi_row(schedule_kpis(rep.after), schedule_kpis(rep.before))
        failed = {e.bus: e.at_min} if e.kind == "bus" else None
        outage = (e.station, e.at_min, e.until_min) if e.kind == "charger" else None
        left, right = st.columns(2)
        left.plotly_chart(gantt_figure(d.inst, rep.before, title="Pred udalosťou", failed=failed, outage=outage),
                          key="gb")
        right.plotly_chart(gantt_figure(d.inst, rep.after, title="Po oprave (optimum)", failed=failed,
                                        moved=rep.moved, cancelled=rep.cancelled, outage=outage), key="ga")
        moved = ", ".join(f"{t} → {b}" for t, b in rep.moved.items()) or "nič"
        st.write(f"Presunuté spoje: {moved}. Zrušené: {', '.join(rep.cancelled) or 'žiadne'}. "
                 f"Dodatočný náklad {rep.plan.extra_cost:.0f}.")
        if rep.valid_rank is None:
            st.warning("Žiadne lacné priradenie neprešlo úplnou kontrolou rozvrhu: použité je zrušenie všetkých "
                       "ohrozených spojov.")
        elif rep.valid_rank:
            st.caption(f"{rep.valid_rank} najlacnejších priradení párového modelu neprešlo úplnou kontrolou.")
        verdict = "je optimálna oprava" if rep.xy_is_optimal else \
            f"nie je optimum (náklad {rep.xy_plan.extra_cost:.0f})"
        if rep.xy_is_optimal:
            st.success(f"Najčastejšia odpoveď XY-QAOA {verdict}.")
        else:
            st.warning(f"Najčastejšia odpoveď XY-QAOA {verdict}.")

        st.plotly_chart(method_figure(rep.rows), key="m")
        table = pd.DataFrame([{"metóda": r["method"], "p": r["p"],
                               "P(optimum)": None if r["p_optimal"] is None else round(r["p_optimal"], 4),
                               "P(prípustné)": None if r["p_feasible"] is None else round(r["p_feasible"], 3),
                               "meraní na 95 %": None if r["shots_95"] is None else round(r["shots_95"])}
                              for r in rep.rows])
        st.dataframe(table, hide_index=True)
        notes = {"computed": "QAOA s pokutami spočítané naživo.", "skipped": "QAOA s pokutami vypnuté."}
        st.caption(notes.get(rep.penalty_source, f"QAOA s pokutami: {rep.penalty_source}.")
                   + " P(optimum) simulated annealing = podiel z 200 behov; QAOA = presná pravdepodobnosť "
                     "zo stavového vektora.")
        ca, ch = rep.circuits["xy_all"], rep.circuits["xy_heavy_hex"]
        st.caption(f"XY-QAOA obvod: {ca['qubits']} qubitov, hĺbka {ca['depth']}, {ca['cx']} CX "
                   f"(všetko-so-všetkým); heavy-hex: {ch['physical_qubits']} fyzických qubitov, hĺbka {ch['depth']}, "
                   f"{ch['cx']} CX. Čas: " + ", ".join(f"{k} {v:.1f}s" for k, v in rep.seconds.items()))

# --------------------------------------------------------------------------- scaling
with tab_scale:
    files = sorted((ROOT / "results").glob("*_scaling.json"))
    if not files:
        st.info("Žiadne výsledky škálovania. Spusti napr.\n\n"
                "`python scripts/run_qaoa.py --columns columns/kcm_tiny_b20.json --variant E2 --scaling 8,10,12,16`")
    else:
        f = st.selectbox("súbor", files, format_func=lambda p: p.name)
        rows = json.loads(f.read_text(encoding="utf-8"))["scaling"]
        st.plotly_chart(scaling_figure(rows), key="sc")
        st.dataframe(pd.DataFrame(rows), hide_index=True)

"""Plotly figures for the schedule, the battery, the grid load and the solver comparison (package G).

    fig = gantt_figure(inst, duties)                    # buses x time: trips, deadheads, charging
    fig = soc_figure(inst, duties)                      # state of charge over the day
    fig = power_figure(duties, slot_min=15)             # charging power per station (stacked)
    fig = method_figure(rows)                           # P(optimum) and P(feasible) per method

Colors (validated categorical palette, light mode): trips blue, reassigned trips
orange, charging aqua, deadheads gray; the failed bus and cancelled trips use the
critical status red and always carry a text label, never color alone.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .columns import Column
from .instance import Instance

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, "#008300", "#6250d6", "#e34948"]
CRITICAL = "#d03b3b"
GRAY = "#a3a29c"
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e9e8e4", "#fcfcfb"


def hhmm(m: float) -> str:
    m = int(round(m))
    return f"{m // 60:02d}:{m % 60:02d}"


def _layout(fig, title: str, height: int, margin_r: int = 20):
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 16, "color": TEXT}},
        height=height, margin={"l": 70, "r": margin_r, "t": 56, "b": 48},
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font={"color": TEXT_2, "size": 12},
        hoverlabel={"bgcolor": "white", "font": {"color": TEXT}},
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 1, "xanchor": "right"},
        bargap=0.25,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def _time_axis(fig, lo: float, hi: float):
    start, end = int(math.floor(lo / 60)), int(math.ceil(hi / 60))
    ticks = list(range(start * 60, end * 60 + 1, 60))
    fig.update_xaxes(range=[lo - 10, hi + 10], tickvals=ticks, ticktext=[hhmm(t) for t in ticks],
                     title_text="čas")


def describe_sk(event) -> str:
    """Slovak label of a evsp.disruption.Disruption."""
    if event.kind == "bus":
        return f"porucha autobusu {event.bus} o {hhmm(event.at_min)}"
    return f"výpadok nabíjačky {event.station} {hhmm(event.at_min)}-{hhmm(event.until_min)}"


# --------------------------------------------------------------------------- battery replay

def soc_timeline(inst: Instance, col: Column) -> tuple[list[float], list[float]]:
    """(minutes, SoC fraction) points of a duty: depot -> trips (+ charging) -> depot."""
    a = inst.assumptions
    cap, kwh_km, eta = a["battery_kwh"], a["kwh_per_km"], a["charge_efficiency"]
    tb = inst.trip_by_id
    soc = a["soc_start"] * cap
    t_pts, s_pts = [col.start_min], [soc / cap]
    loc = inst.depot
    pending = sorted(col.charges, key=lambda c: c.start_min)
    for n, tid in enumerate(col.trips):
        t = tb[tid]
        if n > 0 and pending and pending[0].start_min < t.dep_min:
            ch = pending.pop(0)
            soc -= inst.dh_km(loc, ch.location) * kwh_km
            t_pts.append(ch.start_min)
            s_pts.append(soc / cap)
            soc = min(a["soc_max"] * cap, soc + eta * ch.power_kw * (ch.end_min - ch.start_min) / 60.0)
            t_pts.append(ch.end_min)
            s_pts.append(soc / cap)
            loc = ch.location
        soc -= inst.dh_km(loc, t.start) * kwh_km
        t_pts.append(t.dep_min)
        s_pts.append(soc / cap)
        soc -= t.energy_kwh
        t_pts.append(t.arr_min)
        s_pts.append(soc / cap)
        loc = t.end
    soc -= inst.dh_km(loc, inst.depot) * kwh_km
    t_pts.append(col.end_min)
    s_pts.append(soc / cap)
    return t_pts, s_pts


# --------------------------------------------------------------------------- schedule

def gantt_figure(inst: Instance, duties: dict[str, Column], title: str = "Rozvrh vozidiel",
                 failed: Optional[dict[str, float]] = None, moved: Optional[dict[str, str]] = None,
                 cancelled: Optional[list[str]] = None, outage: Optional[tuple[str, float, float]] = None,
                 height: Optional[int] = None):
    """One row per bus: trips (blue, orange if reassigned), charging (aqua, station label), deadheads (gray).

    failed: {bus: time} draws the failure point; moved: {trip: new bus}; cancelled: trips shown
    on the failed bus's row; outage: (station, from, to) shades the charger outage window.
    """
    import plotly.graph_objects as go

    tb = inst.trip_by_id
    failed, moved, cancelled = failed or {}, moved or {}, list(cancelled or [])
    rows = sorted(duties, key=lambda b: (b.startswith("R"), duties[b].start_min if duties[b] else 0))
    # A cancelled trip belongs to the failed bus's row; keep that bus visible even
    # when the repaired schedule no longer has a duty assigned to it.
    cancel_row = next(iter(failed), rows[0] if rows else None)
    if cancelled and cancel_row is not None and cancel_row not in rows:
        rows.append(cancel_row)
    fig = go.Figure()
    shown = set()

    def bar(row, start, end, color, name, hover, text="", pattern=None):
        fig.add_trace(go.Bar(
            y=[row], x=[end - start], base=[start], orientation="h", marker={
                "color": color, "line": {"color": SURFACE, "width": 2},
                "pattern": {"shape": pattern, "fgcolor": "white"} if pattern else None},
            name=name, legendgroup=name, showlegend=name not in shown, hovertemplate=hover + "<extra></extra>",
            text=text, textposition="inside", insidetextanchor="middle", textfont={"color": "white", "size": 11},
        ))
        shown.add(name)

    lo, hi = math.inf, -math.inf
    for bus in rows:
        col = duties.get(bus)
        if col is None:
            continue
        lo, hi = min(lo, col.start_min), max(hi, col.end_min)
        first = tb[col.trips[0]]
        bar(bus, col.start_min, first.dep_min, GRAY, "prázdny presun",
            f"<b>{bus}</b> výjazd z depa<br>{hhmm(col.start_min)}-{hhmm(first.dep_min)}")
        last = tb[col.trips[-1]]
        bar(bus, last.arr_min, col.end_min, GRAY, "prázdny presun",
            f"<b>{bus}</b> návrat do depa<br>{hhmm(last.arr_min)}-{hhmm(col.end_min)}")
        for tid in col.trips:
            t = tb[tid]
            is_moved = tid in moved
            bar(bus, t.dep_min, t.arr_min, ORANGE if is_moved else BLUE,
                "presunutý spoj" if is_moved else "spoj",
                f"<b>{bus}</b> spoj {tid} (linka {t.route})<br>{hhmm(t.dep_min)}-{hhmm(t.arr_min)}, "
                f"{t.dist_km:.1f} km, {t.energy_kwh:.0f} kWh" + ("<br>presunutý po výpadku" if is_moved else ""),
                text="presun" if is_moved else "")
        for ch in col.charges:
            bar(bus, ch.start_min, ch.end_min, AQUA, "nabíjanie",
                f"<b>{bus}</b> nabíjanie {ch.station}<br>{hhmm(ch.start_min)}-{hhmm(ch.end_min)}, "
                f"{ch.power_kw:.0f} kW, {ch.grid_kwh:.0f} kWh<br>SoC {ch.soc_before:.0%} → {ch.soc_after:.0%}",
                text=ch.station)
    if cancel_row is not None:
        for tid in cancelled:
            t = tb[tid]
            bar(cancel_row, t.dep_min, t.arr_min, CRITICAL, "zrušený spoj",
                f"<b>zrušený spoj {tid}</b><br>{hhmm(t.dep_min)}-{hhmm(t.arr_min)}", text="zrušené",
                pattern="/")
    for bus, at in failed.items():
        if bus in duties:
            fig.add_trace(go.Scatter(
                x=[at], y=[bus], mode="markers+text", marker={"symbol": "x", "size": 14, "color": CRITICAL,
                                                              "line": {"width": 2, "color": CRITICAL}},
                text=["výpadok"], textposition="top center", textfont={"color": CRITICAL},
                name="výpadok", hovertemplate=f"<b>{bus}</b> výpadok o {hhmm(at)}<extra></extra>"))
            lo = min(lo, at)
    for tid in cancelled:
        lo, hi = min(lo, tb[tid].dep_min), max(hi, tb[tid].arr_min)
    if outage:
        st, a, b = outage
        fig.add_vrect(x0=a, x1=b, fillcolor=CRITICAL, opacity=0.08, line_width=0, layer="below")
        fig.add_annotation(x=a, y=1.0, xref="x", yref="paper", xanchor="left", yanchor="bottom", showarrow=False,
                           text=f"{st} mimo prevádzky {hhmm(a)}-{hhmm(b)}", font={"color": CRITICAL, "size": 11})
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(rows)), title_text="")
    _time_axis(fig, lo, hi)
    return _layout(fig, title, height or 110 + 46 * len(rows))


def soc_figure(inst: Instance, duties: dict[str, Column], title: str = "Stav batérie (SoC)"):
    """SoC of every bus over the day; the dashed line is the minimum allowed SoC."""
    import plotly.graph_objects as go

    fig = go.Figure()
    lo, hi = math.inf, -math.inf
    for i, (bus, col) in enumerate(sorted(duties.items(), key=lambda kv: kv[1].start_min if kv[1] else 0)):
        if col is None:
            continue
        t, s = soc_timeline(inst, col)
        lo, hi = min(lo, t[0]), max(hi, t[-1])
        color = SERIES[i % len(SERIES)] if i < len(SERIES) else GRAY
        fig.add_trace(go.Scatter(
            x=t, y=[100 * v for v in s], mode="lines", name=bus, line={"color": color, "width": 2},
            customdata=[hhmm(v) for v in t], hovertemplate=f"<b>{bus}</b> %{{customdata}}: %{{y:.0f}} %<extra></extra>"))
        fig.add_annotation(x=t[-1], y=100 * s[-1], text=bus, showarrow=False, xanchor="left",
                           font={"color": TEXT_2, "size": 11}, xshift=4)
    soc_min = 100 * inst.assumptions["soc_min"]
    fig.add_hline(y=soc_min, line={"color": CRITICAL, "dash": "dash", "width": 1},
                  annotation_text=f"minimum {soc_min:.0f} %", annotation_font_color=CRITICAL,
                  annotation_position="bottom right")
    fig.update_yaxes(range=[0, 105], title_text="SoC [%]")
    _time_axis(fig, lo, hi)
    return _layout(fig, title, 380)


def power_figure(duties: dict[str, Column], slot_min: int = 15, title: str = "Odber zo siete pri nabíjaní"):
    """Charging power per 15-min slot, stacked by station; the peak is labeled."""
    import plotly.graph_objects as go

    load: dict[str, dict[int, float]] = {}
    for col in duties.values():
        if col is None:
            continue
        for st, prof in col.power_by_station.items():
            for t, kw in prof.items():
                load.setdefault(st, {}).setdefault(t, 0.0)
                load[st][t] += kw
    fig = go.Figure()
    if not load:
        fig.add_annotation(text="žiadne nabíjanie počas dňa", showarrow=False, x=0.5, y=0.5,
                           xref="paper", yref="paper", font={"color": TEXT_2})
        return _layout(fig, title, 260)
    slots = sorted({t for prof in load.values() for t in prof})
    t_lo, t_hi = slots[0], slots[-1]
    xs = list(range(t_lo, t_hi + 1))
    total = np.zeros(len(xs))
    for i, st in enumerate(sorted(load)):
        ys = np.array([load[st].get(t, 0.0) for t in xs])
        total += ys
        fig.add_trace(go.Bar(
            x=[t * slot_min + slot_min / 2 for t in xs], y=ys, width=slot_min * 0.9, name=st,
            marker={"color": SERIES[i % len(SERIES)], "line": {"color": SURFACE, "width": 1}},
            customdata=[f"{hhmm(t * slot_min)}-{hhmm((t + 1) * slot_min)}" for t in xs],
            hovertemplate=f"<b>{st}</b> %{{customdata}}: %{{y:.0f}} kW<extra></extra>"))
    k = int(np.argmax(total))
    fig.add_annotation(x=xs[k] * slot_min + slot_min / 2, y=total[k], text=f"špička {total[k]:.0f} kW",
                       showarrow=True, arrowhead=0, ay=-28, font={"color": TEXT})
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="výkon [kW]", rangemode="tozero")
    _time_axis(fig, t_lo * slot_min, (t_hi + 1) * slot_min)
    return _layout(fig, title, 320)


def schedule_kpis(duties: dict[str, Column], slot_min: int = 15) -> dict:
    """Headline numbers of a schedule."""
    cols = [c for c in duties.values() if c is not None]
    load: dict[int, float] = {}
    for c in cols:
        for prof in c.power_by_station.values():
            for t, kw in prof.items():
                load[t] = load.get(t, 0.0) + kw
    return {"buses": len(cols), "trips": sum(c.n_trips for c in cols),
            "deadhead_km": sum(c.deadhead_km for c in cols),
            "charging_kwh": sum(c.grid_kwh_day for c in cols),
            "peak_kw": max(load.values(), default=0.0), "cost": sum(c.cost for c in cols),
            "charging_stops": sum(len(c.charges) for c in cols)}


# --------------------------------------------------------------------------- solver comparison

def method_figure(rows: list[dict], title: str = "Ako často metóda nájde optimálny plán"):
    """Two small panels (no shared y-scale): P(optimum) and P(feasible) per method."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rows = [r for r in rows if r.get("p_optimal") is not None]
    names = [r["method"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("P(optimum)", "P(prípustné riešenie)"),
                        horizontal_spacing=0.22)
    colors = [ORANGE if "XY" in n else (BLUE if "QAOA" in n else GRAY) for n in names]
    fig.add_trace(go.Bar(y=names, x=[r["p_optimal"] for r in rows], orientation="h", marker_color=colors,
                         text=[f"{r['p_optimal']:.3f}" for r in rows], textposition="outside",
                         hovertemplate="%{y}: %{x:.4f}<extra></extra>", showlegend=False), row=1, col=1)
    feas = [r.get("p_feasible") for r in rows]          # not measured for SA: labeled, no bar
    fig.add_trace(go.Bar(y=names, x=[v or 0.0 for v in feas], orientation="h", marker_color=colors,
                         text=["nemerané" if v is None else f"{v:.2f}" for v in feas], textposition="outside",
                         hovertemplate="%{y}: %{x:.3f}<extra></extra>", showlegend=False), row=1, col=2)
    top = max([r["p_optimal"] for r in rows] + [1e-3])
    fig.update_xaxes(range=[0, top * 1.35], row=1, col=1)
    fig.update_xaxes(range=[0, 1.15], row=1, col=2)
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title, 120 + 48 * len(rows))


def scaling_figure(rows: list[dict], title: str = "QAOA s pokutami slabne s veľkosťou problému"):
    """P(optimum) vs qubits for QAOA, simulated annealing and random guessing (log scale)."""
    import plotly.graph_objects as go

    n = [r["qubits"] for r in rows]
    fig = go.Figure()
    series = [("simulated annealing", [r["sa_p_optimal"] for r in rows], GRAY),
              ("QAOA (najlepšie p ≤ 3)", [r["best_p_optimal"] for r in rows], BLUE),
              ("náhodný výber", [r["uniform_p_optimal"] for r in rows], TEXT_2)]
    for name, ys, color in series:
        fig.add_trace(go.Scatter(x=n, y=ys, mode="lines+markers", name=name,
                                 line={"color": color, "width": 2, "dash": "dot" if "náhod" in name else None},
                                 marker={"size": 8, "line": {"color": SURFACE, "width": 2}},
                                 hovertemplate=f"{name}, %{{x}} qubitov: %{{y:.2e}}<extra></extra>"))
        fig.add_annotation(x=n[-1], y=math.log10(max(ys[-1], 1e-12)), text=name, showarrow=False,
                           xanchor="left", xshift=6, font={"color": TEXT_2, "size": 11})
    fig.update_yaxes(type="log", title_text="P(optimum)", exponentformat="power")
    fig.update_xaxes(title_text="počet qubitov", tickvals=n)
    return _layout(fig, title, 380, margin_r=170)

"""QUBO builder for the E-VSP set-partitioning model (package C).

One binary variable y_k per selected column (duty). The Hamiltonian follows
the "Účelové funkcie E-VSP" document:

    H(y) = sum_k c_k y_k                                   costs (vehicle, deadhead, energy)
         + P sum_i (sum_k a_ik y_k - 1)^2                  every trip exactly once
         + Q sum_{(k,l) in C} y_k y_l                      one bus per 1-charger station at a time
         + R sum_s sum_t (sum_k p_kst y_k)^2               peak power (per station or grid-wide)
         + w_rob sum_{k != l} r_kl y_k y_l                 back-to-back charging on one charger
         + M (sum_k y_k - V)^2                             exact fleet size (variant E3)

with c_k = w_veh + w_dh D_k + w_en * energy_cost_k. After y_k^2 = y_k it is
stored as  E(y) = offset + sum_k lin_k y_k + sum_{k<l} quad_kl y_k y_l.

Variants (see `variant_weights`):
    E1  vehicles + coverage + conflicts
    E2  E1 + deadhead km + electricity cost (equals the MILP objective)
    E3  deadhead + electricity + peak + conflicts + fixed fleet V
    E4  E2 + peak + charger robustness   (main demo variant)

Usage:
    cs = load_columns("columns/kcm_tiny_b20.json")
    q = build_qubo(cs, variant_weights("E4", cs.assumptions))
    x, e = brute_force(q)                  # exact ground state for n <= ~24
    rep = q.evaluate(x)                    # feasible? which trips missing? peak kW?
    op, offset = q.to_sparse_pauli_op()    # Qiskit Hamiltonian for QAOA
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .columns import ColumnSet


# --------------------------------------------------------------------------- weights

@dataclass
class Weights:
    """Weights in cost units. P, Q, M = None means automatic (see `auto_penalties`)."""

    w_veh: float = 500.0             # per vehicle (column)
    w_dh: float = 1.5                # per deadhead km
    w_en: float = 1.0                # multiplier on the column's electricity cost
    R: float = 0.0                   # peak weight per (peak_unit_kw)^2 per slot
    w_rob: float = 0.0               # per minute of missing buffer between two charges
    P: Optional[float] = None        # coverage penalty
    Q: Optional[float] = None        # charger conflict penalty
    M: float = 0.0                   # fleet-size penalty (E3)
    V: Optional[int] = None          # fleet size for E3
    peak_unit_kw: float = 100.0      # power is measured in units of this many kW
    peak_scope: str = "station"      # "station": sum_s sum_t (.)^2 | "total": sum_t (sum_s .)^2
    penalty_factor: float = 2.0      # P = Q = factor * max_k(largest cost a column can add)
    variant: str = "custom"

    def to_dict(self) -> dict:
        return asdict(self)


def variant_weights(variant: str, assumptions: Optional[dict] = None, V: Optional[int] = None,
                    R: float = 1.0, w_rob: float = 1.0) -> Weights:
    """Preset weights for E1-E4; costs default to the instance assumptions."""
    a = assumptions or {}
    base = dict(w_veh=a.get("vehicle_cost", 500.0), w_dh=a.get("deadhead_cost_per_km", 1.5), w_en=1.0)
    v = variant.upper()
    if v == "E1":
        return Weights(**{**base, "w_dh": 0.0, "w_en": 0.0}, variant=v)
    if v == "E2":
        return Weights(**base, variant=v)
    if v == "E3":
        if V is None:
            raise ValueError("E3 needs the fleet size V")
        return Weights(**{**base, "w_veh": 0.0}, R=R, V=V, M=-1.0, variant=v)   # M auto (<0)
    if v == "E4":
        return Weights(**base, R=R, w_rob=w_rob, variant=v)
    raise ValueError(f"unknown variant {variant!r} (use E1, E2, E3 or E4)")


# --------------------------------------------------------------------------- model

@dataclass
class QUBO:
    """E(y) = offset + lin . y + sum_{(k,l)} quad[(k,l)] y_k y_l,  k < l."""

    linear: np.ndarray
    quadratic: dict[tuple[int, int], float]
    offset: float
    names: list[str]                       # column ids, variable k <-> names[k]
    weights: Weights
    meta: dict = field(default_factory=dict)
    # data kept to evaluate solutions term by term
    _cover: list[list[int]] = field(default_factory=list, repr=False)       # trip -> columns
    _conflicts: list[tuple[int, int]] = field(default_factory=list, repr=False)
    _tight: list[tuple[int, int, float]] = field(default_factory=list, repr=False)
    _costs: Optional[np.ndarray] = field(default=None, repr=False)
    _power: Optional[np.ndarray] = field(default=None, repr=False)          # [K, S, T] in peak units
    _stations: list[str] = field(default_factory=list, repr=False)
    _trip_ids: list[str] = field(default_factory=list, repr=False)

    @property
    def n(self) -> int:
        return len(self.linear)

    # ---------------------------------------------------------------- energy
    def matrix(self) -> np.ndarray:
        """Upper-triangular Q with the linear terms on the diagonal (E = y^T Q y + offset)."""
        m = np.diag(self.linear.astype(float))
        for (k, l), v in self.quadratic.items():
            m[k, l] += v
        return m

    def energy(self, x) -> np.ndarray | float:
        """Energy of one bitstring (shape [n]) or many (shape [B, n])."""
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        x = np.atleast_2d(x)
        m = self.matrix()
        e = np.einsum("bi,ij,bj->b", x, m, x) + self.offset
        return float(e[0]) if single else e

    # ---------------------------------------------------------------- reading a solution
    def evaluate(self, x) -> dict:
        """Break a bitstring down into the terms of H and check feasibility."""
        x = np.asarray(x).astype(int)
        chosen = [k for k in range(self.n) if x[k]]
        cover_count = [sum(x[k] for k in cols) for cols in self._cover]
        missing = [self._trip_ids[i] for i, c in enumerate(cover_count) if c == 0]
        double = [self._trip_ids[i] for i, c in enumerate(cover_count) if c > 1]
        conflicts = [(k, l) for k, l in self._conflicts if x[k] and x[l]]
        tight = sum(r for k, l, r in self._tight if x[k] and x[l])
        load = np.einsum("k,kst->st", x.astype(float), self._power) if self._power is not None and self._power.size else None
        peak_kw = ({s: float(load[i].max() * self.weights.peak_unit_kw) for i, s in enumerate(self._stations)}
                   if load is not None else {})
        total_peak_kw = float(load.sum(axis=0).max() * self.weights.peak_unit_kw) if load is not None else 0.0
        return {
            "feasible": not (missing or double or conflicts) and (self.weights.V is None or len(chosen) == self.weights.V
                                                                    or self.weights.M == 0),
            "columns": [self.names[k] for k in chosen],
            "n_vehicles": len(chosen),
            "energy": self.energy(x),
            "cost": float(self._costs @ x) if self._costs is not None else None,
            "missing_trips": missing,
            "double_trips": double,
            "violated_conflicts": conflicts,
            "tight_minutes": tight,
            "peak_kw_by_station": peak_kw,
            "peak_kw_total": total_peak_kw,
        }

    # ---------------------------------------------------------------- conversions
    def to_ising(self) -> tuple[np.ndarray, dict[tuple[int, int], float], float]:
        """y = (1 - z) / 2,  z in {-1, +1}:  E = c + sum h_i z_i + sum J_ij z_i z_j."""
        h = -self.linear / 2.0
        const = self.offset + self.linear.sum() / 2.0
        J: dict[tuple[int, int], float] = {}
        for (k, l), v in self.quadratic.items():
            J[(k, l)] = J.get((k, l), 0.0) + v / 4.0
            h[k] -= v / 4.0
            h[l] -= v / 4.0
            const += v / 4.0
        return h, J, const

    def to_sparse_pauli_op(self):
        """Qiskit operator H_C with E(y) = <y|H_C|y> + offset.

        Variable k is qubit k. Qiskit prints bitstrings little-endian, so bit k of
        a measured string s is s[::-1][k]; `bitstring_to_x` does this for you.
        """
        from qiskit.quantum_info import SparsePauliOp

        h, J, const = self.to_ising()
        terms = [("Z", [k], float(v)) for k, v in enumerate(h) if abs(v) > 1e-12]
        terms += [("ZZ", [k, l], float(v)) for (k, l), v in J.items() if abs(v) > 1e-12]
        if not terms:
            terms = [("I", [0], 0.0)]
        return SparsePauliOp.from_sparse_list(terms, num_qubits=self.n).simplify(), const

    def to_quadratic_program(self):
        """qiskit-optimization QuadraticProgram (minimize), one binary per column."""
        from qiskit_optimization import QuadraticProgram

        qp = QuadraticProgram(name=self.meta.get("name", "evsp"))
        for name in self.names:
            qp.binary_var(name)
        qp.minimize(constant=self.offset,
                    linear={self.names[k]: float(v) for k, v in enumerate(self.linear)},
                    quadratic={(self.names[k], self.names[l]): float(v) for (k, l), v in self.quadratic.items()})
        return qp

    def to_bqm(self):
        """dimod BinaryQuadraticModel (for simulated annealing / D-Wave)."""
        import dimod

        return dimod.BinaryQuadraticModel(
            {k: float(v) for k, v in enumerate(self.linear)},
            {kl: float(v) for kl, v in self.quadratic.items()},
            float(self.offset), dimod.BINARY)

    def scaled(self) -> "QUBO":
        """Same QUBO divided by its largest |coefficient| (same ground state, nicer for QAOA angles)."""
        s = max(np.abs(self.linear).max(initial=0.0), max((abs(v) for v in self.quadratic.values()), default=0.0))
        s = s or 1.0
        out = QUBO(linear=self.linear / s, quadratic={kl: v / s for kl, v in self.quadratic.items()},
                   offset=self.offset / s, names=self.names, weights=self.weights,
                   meta={**self.meta, "scale": s * self.meta.get("scale", 1.0)},
                   _cover=self._cover, _conflicts=self._conflicts, _tight=self._tight, _costs=self._costs,
                   _power=self._power, _stations=self._stations, _trip_ids=self._trip_ids)
        return out

    def stats(self) -> dict:
        nz = [abs(v) for v in self.quadratic.values() if abs(v) > 1e-12]
        coeffs = [abs(v) for v in self.linear if abs(v) > 1e-12] + nz
        return {
            "n_variables": self.n,
            "n_quadratic": len(nz),
            "density": len(nz) / max(1, self.n * (self.n - 1) / 2),
            "coef_min": float(min(coeffs)) if coeffs else 0.0,
            "coef_max": float(max(coeffs)) if coeffs else 0.0,
            "P": self.meta.get("P"), "Q": self.meta.get("Q"), "M": self.meta.get("M"),
        }

    # ---------------------------------------------------------------- io
    def to_json(self, path: str | Path) -> Path:
        out = {
            "names": self.names,
            "offset": self.offset,
            "linear": [float(v) for v in self.linear],
            "quadratic": [[k, l, float(v)] for (k, l), v in sorted(self.quadratic.items())],
            "weights": self.weights.to_dict(),
            "meta": self.meta,
            "stats": self.stats(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        return path


def bitstring_to_x(bitstring: str) -> np.ndarray:
    """Qiskit measurement string (qubit 0 rightmost) -> variable vector x[k]."""
    return np.array([int(b) for b in bitstring[::-1]], dtype=int)


# --------------------------------------------------------------------------- builder

def auto_penalties(costs: np.ndarray, peak_self: np.ndarray, w: Weights, tight_max: np.ndarray) -> float:
    """Penalty = factor * the most one column can add to the objective by itself.

    Removing a column saves at most its own cost terms, so a coverage or conflict
    violation must cost more than that. Verified by brute force in the tests.
    """
    worst = costs + w.R * peak_self + w.w_rob * tight_max
    return w.penalty_factor * float(max(worst.max(initial=0.0), 1e-9))


def build_qubo(cs: ColumnSet, weights: Optional[Weights] = None, indices: Optional[list[int]] = None,
               name: Optional[str] = None) -> QUBO:
    """Build the QUBO over `indices` (default: every column in the set)."""
    w = weights or Weights()
    idx = list(range(len(cs.columns))) if indices is None else list(indices)
    n = len(idx)
    cols = [cs.columns[k] for k in idx]

    # ---- linear costs c_k
    costs = np.array([w.w_veh + w.w_dh * c.deadhead_km + w.w_en * c.energy_cost for c in cols], dtype=float)

    # ---- power tensor in peak units
    power, stations, n_slots = cs.power_tensor(idx)
    power = power / w.peak_unit_kw
    if w.peak_scope == "total":
        power_terms = power.sum(axis=1, keepdims=True)        # [K, 1, T]
    elif w.peak_scope == "station":
        power_terms = power
    else:
        raise ValueError("peak_scope must be 'station' or 'total'")
    peak_self = (power_terms ** 2).sum(axis=(1, 2)) if power_terms.size else np.zeros(n)

    conflicts, tight = cs.pairs(idx)
    tight_max = np.zeros(n)
    for k, l, r in tight:
        tight_max[k] = max(tight_max[k], r)
        tight_max[l] = max(tight_max[l], r)

    base_pen = auto_penalties(costs, peak_self, w, tight_max)
    P = w.P if w.P is not None else base_pen
    Q = w.Q if w.Q is not None else base_pen
    M = base_pen if w.M < 0 else w.M

    lin = costs.copy()
    quad: dict[tuple[int, int], float] = defaultdict(float)
    offset = 0.0

    def add(k: int, l: int, v: float):
        if k == l:
            lin[k] += v
        else:
            quad[(min(k, l), max(k, l))] += v

    # ---- coverage: P (sum_k a_ik y_k - 1)^2 = P(-sum y + 2 sum_{k<l} y y + 1)
    row = {t: r for r, t in enumerate(cs.trip_ids)}
    cover: list[list[int]] = [[] for _ in cs.trip_ids]
    for pos, c in enumerate(cols):
        for t in c.trips:
            cover[row[t]].append(pos)
    for members in cover:
        for a_ in members:
            lin[a_] -= P
        for x_, a_ in enumerate(members):
            for b_ in members[x_ + 1:]:
                add(a_, b_, 2 * P)
        offset += P

    # ---- conflicts
    for k, l in conflicts:
        add(k, l, Q)

    # ---- peak: R sum_{s,t} (sum_k p y)^2
    if w.R and power_terms.size:
        lin += w.R * peak_self
        flat = power_terms.reshape(n, -1)
        gram = flat @ flat.T
        for k in range(n):
            for l in range(k + 1, n):
                if gram[k, l] > 1e-12:
                    add(k, l, 2 * w.R * gram[k, l])

    # ---- robustness on chargers
    if w.w_rob:
        for k, l, r in tight:
            add(k, l, w.w_rob * r)

    # ---- fleet size: M (sum y - V)^2 = M(sum y (1 - 2V) + 2 sum_{k<l} y y + V^2)
    if M and w.V is not None:
        lin += M * (1 - 2 * w.V)
        for k in range(n):
            for l in range(k + 1, n):
                add(k, l, 2 * M)
        offset += M * w.V ** 2

    meta = {"name": name or cs.instance_name, "variant": w.variant, "P": P, "Q": Q, "M": M,
            "n_trips": len(cs.trip_ids), "n_conflicts": len(conflicts), "n_tight": len(tight),
            "stations": stations, "n_slots": n_slots, "scale": 1.0}
    return QUBO(linear=lin, quadratic={kl: v for kl, v in quad.items() if abs(v) > 1e-12}, offset=offset,
                names=[c.id for c in cols], weights=w, meta=meta, _cover=cover, _conflicts=conflicts,
                _tight=tight, _costs=costs, _power=power_terms, _stations=stations if w.peak_scope == "station"
                else ["TOTAL"], _trip_ids=list(cs.trip_ids))


# --------------------------------------------------------------------------- exact solver

def brute_force(q: QUBO, max_n: int = 26, chunk: int = 1 << 16) -> tuple[np.ndarray, float]:
    """Exact ground state by enumerating all 2^n bitstrings (vectorized, chunked)."""
    n = q.n
    if n > max_n:
        raise ValueError(f"{n} variables is too many for brute force (max {max_n})")
    m = q.matrix()
    best_e, best_x = np.inf, None
    total = 1 << n
    bits = np.arange(n, dtype=np.int64)
    for start in range(0, total, chunk):
        ids = np.arange(start, min(total, start + chunk), dtype=np.int64)
        x = ((ids[:, None] >> bits) & 1).astype(float)
        e = np.einsum("bi,ij,bj->b", x, m, x)
        i = int(np.argmin(e))
        if e[i] < best_e:
            best_e, best_x = float(e[i]), x[i].astype(int)
    return best_x, best_e + q.offset

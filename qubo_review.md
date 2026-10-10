## Review: work package C

The repository now has a QUBO builder in `evsp/qubo.py`, a command-line entry point in `scripts/build_qubo.py`, and tests in `tests/test_qubo.py`. **The core QUBO expansion and format conversions are implemented, but some choices differ from the markdown, and the weight-tuning and validation workflow is incomplete.**

### What’s implemented

- One binary variable per selected duty. The builder adds the trip-coverage penalty, charger-overlap conflicts, optional peak and robustness terms, and an optional fixed-fleet term. Its coverage, conflict, and peak expansions follow the stated binary-QUBO algebra.
- Conversion to Ising, Qiskit `SparsePauliOp`, `qiskit-optimization` `QuadraticProgram`, and dimod BQM. Tests compare the converted representations to the QUBO energy; optional integrations are skipped if their packages aren’t installed.
- Exact brute force, up to 26 variables, and tests comparing tiny/small E1 and E2 ground states against the MILP on the same duty pool. That covers the document’s `n ≤ 20` target for those tests.

### Gaps and differences from the specification

1. **Peak scope defaults to per-station, while the markdown specifies a grid-wide peak.** The default is `"station"` in both `Weights` and the CLI. In this mode, the builder squares each station’s load separately, so it omits cross-station terms. `"total"` implements the grid-wide sum before squaring; use it to match the markdown’s Hamiltonian.

2. **The energy term differs from the stated cost definition.** The markdown defines cost using energy in kWh, multiplied by an energy-price coefficient. The builder instead multiplies `w_en` by each duty’s `energy_cost`—a tariff-based monetary estimate that includes the overnight refill. That may be a deliberate operating-cost model, but it should be aligned with or explicitly distinguished from the markdown’s formula.

3. **Cost normalization and automatic weight-grid tuning are absent.** The plan calls for normalizing costs to `[0,1]` and tuning `P`, `Q`, and `R` on tiny by grid search. The current code uses raw costs and a penalty heuristic, while the CLI accepts a manual `R` and penalty factor; it has no grid-search tuner.

4. **The CLI reports, but does not enforce, a feasible brute-force ground state.** It writes the QUBO JSON before brute-force checking, then prints whether the ground state is feasible. It does not fail the command or withhold the artifact if it isn’t. The tests do assert feasibility for their chosen cases, but the script can still export a bad parameterization.

5. **Optional export dependencies aren’t listed in `requirements.txt`.** `qiskit-optimization` and `dimod` are imported only when their conversion methods are called, and their tests use `importorskip`. So those exports aren’t guaranteed to work in a standard project install.

**Assessment:** the main builder and algebra appear substantially implemented, with useful conversion and ground-state tests. I’d treat C as partially complete against this document: set the intended peak scope explicitly, resolve the energy-cost definition, and add the missing weight-tuning and export-gating behavior before calling the work package complete. I reviewed the code and tests without running them.

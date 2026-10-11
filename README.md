# E-bus scheduling with quantum optimization (E-VSP)

Hybrid quantum-classical scheduling of battery-electric buses on real King County Metro
data (route 150, Seattle). The project covers:

- the daily schedule as a set-partitioning QUBO (variants E1–E4) solved with QAOA,
- repair of the schedule after a bus failure or a charger outage with an XY-mixer QAOA
  that never leaves the valid assignments,
- classical baselines on the same problems (min-cost flow, MILP, greedy dispatcher,
  exact repair, simulated annealing, tabu search),
- a best-of-N solution-quality benchmark (QAOA vs simulated annealing),
- a Streamlit demo and figures.

Qollab × IonQ hackathon project. *This effort is supported by Qollab & IonQ.* MIT license.

## What is in git and what you generate

| folder | in git | content |
|---|---|---|
| `evsp/` | yes | library: instances, columns, QUBO, QAOA, XY-QAOA, repair, baselines, benchmark, viz |
| `scripts/`, `app/`, `tools/`, `tests/` | yes | command-line steps, Streamlit demo, Qollab packaging, tests |
| `instances/` | yes | the 4 bundled instances (tiny, small, medium, large) and their validation report |
| `columns/` | yes | the duty pools used by QAOA, the repair and the demo. Small and reproduced byte for byte by step 3 |
| `results/` | **no** | every experiment output (JSON, CSV, PNG). Run the steps below to create it |
| `qubo/` | **no** | QUBO exports for external solvers (step 4). No script reads them: `run_qaoa.py` builds the QUBO from the columns |
| `figures/`, `dist/`, `repair.png` | no | generated figures, the Qollab bundle, the `main.py` chart |
| `external/` | no | optional GTFS source, only needed to rebuild the instances (step 1) |

The experiments are not versioned. Every number in this README can be reproduced with
the commands below. The tests do not need `results/`.

## Quick start

Python 3.11 or newer (tested with 3.13 and 3.14). From the repository root:

```sh
python -m pip install -r requirements.txt
python -m pytest -q          # 178 tests, about 1–2 min
python main.py               # end-to-end repair run (Qollab entry point), about 5 s
```

## Pipeline: how to work with the project

Run the steps in this order. Each step reads only files that are in git or created by an
earlier step. Times were measured on a 2-core cloud machine without a GPU.

| step | command | writes | time |
|---|---|---|---|
| 0 | `python -m pip install -r requirements.txt` then `python -m pytest -q` | – | 1–2 min |
| 1 *(optional)* | `python scripts/build_instances.py --source-revision <commit>` | `instances/*.json` | see [Rebuild from GTFS](#rebuild-from-gtfs) |
| 2 | `python scripts/validate_instances.py` | `instances/baseline_validation.json` | < 1 s |
| 3 *(optional)* | `python scripts/build_columns.py --instance instances/kcm_tiny.json --budget 20`<br>`python scripts/build_columns.py --instance instances/kcm_small.json --budget 20 --battery 200` | `columns/kcm_tiny_b20.json`, `columns/kcm_small_b20_bat200.json` | < 1 s each |
| 4 *(optional)* | `python scripts/build_qubo.py --columns columns/kcm_small_b20_bat200.json --variant E4` | `qubo/<columns>_<variant>.json` | ~1 s |
| 5 | `python scripts/run_classical.py` | `results/baselines.json` | ~1 min |
| 6a | `python scripts/run_qaoa.py --columns columns/kcm_small_b20_bat200.json --variant E4 --classical-only --compare-peak` | `results/*_peak_comparison_classical.{json,csv}` | ~2 s |
| 6b | `python scripts/run_qaoa.py --columns columns/kcm_small_b20_bat200.json --variant E4 --compare-peak` | `results/*_peak_comparison.{json,csv}` | ~6 min |
| 6c | `python scripts/run_qaoa.py --columns columns/kcm_tiny_b20.json --variant E2 --scaling 8,10,12,16` | `results/*_scaling.{json,csv}` | ~10 s |
| 7 | `python scripts/run_repair.py --bus k4 --at 06:00` | `results/repair_bus_k4_0600.json` | ~5 min |
| 8 | `python scripts/run_benchmark.py` | `results/quality_benchmark.{json,csv}` + 3 PNG charts | ~20 s |
| 9 | `python scripts/make_figures.py` | `figures/*.html` | ~5 s |
| 10 | `streamlit run app/demo.py` | – (browser) | – |
| 11 | `python tools/package_qollab.py` | `dist/qollab/`, `dist/qollab.zip` | < 1 s |

Notes on the steps:

- **Steps 1 and 3** regenerate inputs that are already in git. Step 3 reproduces the
  bundled columns byte for byte. Run these steps only after you change the data or the
  duty generator.
- **Step 4** exports a QUBO as JSON, for inspection or an external solver. Nothing in
  the pipeline depends on it.
- **Step 6** runs QAOA on the daily schedule (package E):
  - 6a gives the classical reference on the same QUBO: exact search, simulated annealing
    and tabu search.
  - 6b runs QAOA itself. It is the slowest step, because it simulates 20 qubits for two
    values of R.
  - 6c writes the scaling data that the demo's *Škálovanie* tab shows.
- **Step 7 feeds the demo.** Penalty QAOA on 20 qubits is too slow to run live, so the
  demo reads its numbers from `results/repair_bus_k4_0600.json`. Without the file the demo
  still works, and the penalty-QAOA row says "skipped: … run scripts/run_repair.py". For
  another event, use for example `--bus k585 --at 07:00` or
  `--charger S3 --from 07:00 --to 09:00`.
- **Step 8** draws the charts at the end. `python scripts/plot_benchmark.py` redraws them
  from the saved JSON without recomputing.
- **Steps 9–10** read whatever is in `results/`. Run 6c and 7 first to get all figures
  and tabs.

Run the whole pipeline (about 15 min, most of it steps 6b and 7):

```sh
python scripts/validate_instances.py
python scripts/run_classical.py
python scripts/run_qaoa.py --columns columns/kcm_small_b20_bat200.json --variant E4 --classical-only --compare-peak
python scripts/run_qaoa.py --columns columns/kcm_small_b20_bat200.json --variant E4 --compare-peak
python scripts/run_qaoa.py --columns columns/kcm_tiny_b20.json --variant E2 --scaling 8,10,12,16
python scripts/run_repair.py --bus k4 --at 06:00
python scripts/run_benchmark.py
python scripts/make_figures.py
streamlit run app/demo.py
```

## Qollab entry point (`main.py`)

`main.py` is the file that runs on Qollab. It runs one repair event end to end:

- the exact true-cost optimum, the greedy dispatcher rule and simulated annealing,
- XY-mixer QAOA and penalty QAOA (penalty QAOA up to 18 qubits),
- **one** XY-QAOA circuit transpiled and sampled on the backend that Qollab injects,
  which fits IonQ's limit of one job per run,
- the best-of-N comparison for the event,
- a before/after chart, `repair.png`.

```sh
python main.py                        # bus k585 fails at 07:00, 14 qubits
python main.py --bus k4 --at 06:00    # 20 qubits
python main.py --shots 2000 --p 2
```

Without an injected `backend`, `main.py` falls back to Qiskit's `BasicSimulator`.

To upload, run `python tools/package_qollab.py`. It copies only what `main.py` needs into
`dist/qollab.zip` (19 files, about 233 KB; Qollab allows at most 100 files of up to 2 MB
each). `QOLLAB.md` is the project page text and `requirements-qollab.txt` lists the
runtime dependencies.

## QUBO builder (work package C)

`evsp/qubo.py` builds one binary variable per selected duty. The default peak
scope is grid-wide (`sum_t (sum_k p_kt y_k)^2`); `--peak-scope station` selects
the alternative sum of independent station peaks. Tariff-based energy cost is
the default so E2 remains comparable with the existing MILP objective. Use
`--energy-basis kwh` to price daytime charging plus the overnight refill directly
in grid kWh, as in the project formulation. `--normalize-costs` scales the duty
cost component to at most 1 before automatic penalties are calculated.

```sh
python scripts/build_qubo.py --columns columns/kcm_tiny_b20.json --variant E4
python scripts/build_qubo.py --columns columns/kcm_tiny_b20.json --variant E4 --tune
```

`--tune` sweeps the configured `R` and penalty-factor grids, uses brute force to
check the exact ground state, and selects the feasible setting with the lowest
grid peak (then cost). It is limited to 20 selected duties. The command validates
an exact ground state before exporting and stops if it is infeasible; over 26
variables it warns and exports without an exact check. Qiskit is pinned in
`requirements.txt`; `qiskit-optimization` and `dimod` are optional imports needed
only for their corresponding conversion methods.

QAOA supports `--optimizer cobyla` (default) and `--optimizer spsa`. To compare
the peak-shaving objective on the same selected columns and solver settings, use:

```sh
python scripts/run_qaoa.py --columns columns/kcm_tiny_b20.json --variant E4 --compare-peak
```

This writes separate `R=0` and `R>0` results, including QAOA and simulated
annealing metrics. SPSA optimizes the same CVaR objective as COBYLA; the optional
`--qiskit --noise` mode applies the noise model when sampling the optimized circuit.

## Column export safety (work package B)

`scripts/build_columns.py` stops before export when replay or schedule validation
fails. Replay checks the charging station, location, power, duration, energy,
battery limits and depot departure/arrival. Aggregated stations must have identical
connector locations and powers; heterogeneous stations require a connector-level model.

Pairwise QUBO conflicts cannot represent a general multi-connector capacity limit.
Export therefore conservatively rejects a selected pool whose combined charging
intervals exceed a multi-connector station's capacity, even if some candidates are
mutually exclusive. The LP/MILP still supports these capacity constraints. Reduce
the pool or use `--no-depot-charging` to generate alternatives without depot charging.
This check prevents unsafe export; it does not add a new QUBO capacity formulation.

## Classical baselines (work package D)

```sh
python scripts/run_classical.py                  # diesel + electric + repair, writes results/baselines.json
python scripts/run_classical.py --part diesel --instances large
python scripts/run_classical.py --part electric --battery instance 200
python scripts/run_classical.py --part repair
python scripts/run_qaoa.py --columns columns/kcm_small_b20_bat200.json --variant E4 --classical-only --compare-peak
```

`run_qaoa.py --classical-only` benchmarks exact enumeration, simulated annealing
and tabu search on the same selected columns and QUBO used by QAOA, without
optimizing angles or constructing a circuit. The existing SA implementation is
NumPy Metropolis annealing; tabu is multi-start single-bit search with aspiration.
Exact enumeration is limited to 24 variables. `--sa-reads`, `--sa-sweeps`,
`--tabu-reads`, `--tabu-steps` and `--seed` control the heuristic runs.

Every run writes JSON plus a CSV table with instance, column file, variant, R,
qubits, method, QUBO energy, feasibility, cost, vehicles, deadhead km, grid kWh
(daytime plus overnight refill), peak kW and solver time. Classical filenames
end in `_classical.json` / `_classical.csv`; `--compare-peak` includes both R
settings in one table. Classical runs preserve existing QAOA result files.
Heuristic rows describe the lowest-energy raw solution, including when it is
infeasible. Normal QAOA runs also export one row per depth for the best raw
sample, with cumulative optimization time through that depth (INTERP uses the
preceding layers). These rows are ideal simulations; optional noisy Qiskit
sampling and classical repair remain separately labelled in JSON. Solver times
exclude QUBO construction; exact landscape construction is timed separately and
is shared reference data for SA/QAOA metrics. It is not included in their times.

The greedy dispatcher and MILP in `run_classical.py` are separate operational
references: they use their own generated duties and the MILP omits the peak
term, so they are not claimed as results on the selected QAOA QUBO.

`evsp/classical_baselines.py` contains these operational reference methods:

- **Diesel scenarios S1–S3 of the assignment** (no battery). These are polynomial and
  are solved as one minimum-cost flow model over the trip connection graph.
  - S1 minimises buses.
  - S2 minimises buses plus deadhead km with the instance weights (500 per bus, 1.5 per km).
  - S3 minimises deadhead km with at most `v_max` buses and returns a trade-off curve.

  The LP constraint matrix is totally unimodular, and `solve_flow` checks that the
  HiGHS solution is integral. S1 is cross-checked by Hopcroft–Karp bipartite
  matching (minimum path cover = trips − maximum matching).
- **Real GTFS blocks** are measured with the same deadhead estimates. They are not a
  fair optimum to beat: real blocks also serve routes outside the instance, and real
  operations have constraints (drivers, garages) that are not modelled here.
- **Electric baselines:**
  - `battery_check` replays the diesel optimum with a battery.
  - `greedy_electric` is a first-fit dispatcher: cheapest feasible insertion, greedy
    charging, and charger capacity respected. Its duties are valid columns of package B.
  - For tiny and small, the exact reference is the MILP over all enumerated duties.
    For larger instances, the diesel S1 fleet is a lower bound.
- **Repair baselines:**
  - `greedy_repair` gives each affected trip its cheapest option that still replays,
    in time order.
  - `exact_repair` replays every one-hot assignment, so it counts reserve activation
    once and finds the true-cost optimum.

Results of `python scripts/run_classical.py` (battery 200 kWh for the electric part):

| | tiny | small | medium | large |
|---|---:|---:|---:|---:|
| S1 / S2 buses (flow) | 2 | 5 | 12 | 118 |
| S2 deadhead km | 71 | 176 | 454 | 5107 |
| real GTFS blocks | 2 | 5 | 32 | 204 |
| diesel optimum still battery-feasible | 0/2 | 3/5 | 0/12 | – |
| greedy electric buses / exact MILP buses | 4 / 3 | 7 / 5 | 21 / ≥12 | – |

Repair, 38 bus failures of the 5-bus demo schedule:
- Greedy repair always matches the replayed QUBO optimum, which is what QAOA targets, and takes about 1 ms.
- Both match the true-cost optimum in 26 of 38 events.
- The remaining gap comes from the pairwise QUBO model: it counts reserve activation per trip and treats insertion costs as additive.

On this problem size, classical methods are therefore fast and exact. The quantum part has to
be judged by how it scales and by how well the circuit design works, not by solution quality on these instances.

## Solution-quality benchmark: QAOA vs simulated annealing (best of N samples)

```sh
python scripts/run_benchmark.py      # 5 repair events, ~20 s; results/quality_benchmark.{json,csv} + charts
python scripts/plot_benchmark.py     # redraw the 3 charts from the saved JSON
python scripts/run_benchmark.py --events k4@06:00 S3@07:00-09:00 --budgets 1 10 100 --p-max 3
```

Events are written `bus@HH:MM` (bus failure) or `station@HH:MM-HH:MM` (charger outage).
`--no-plot` skips the charts.

QAOA and simulated annealing are both samplers: one shot or one annealing run gives
one candidate repair. A dispatcher takes N candidates, replays them and keeps the
cheapest. `evsp/benchmark.py` works this out as follows:

1. It replays every valid assignment once to get its **true** extra cost. Reserve
   activation is counted once per bus. A sample that is not a valid repair counts as
   the cancel-all fallback.
2. For each method it builds the cost distribution of one sample:
   - XY-mixer QAOA and penalty QAOA from the exact statevector,
   - simulated annealing from independent runs,
   - two random baselines (uniform over valid assignments, uniform over all bitstrings).
3. It computes exactly, without Monte Carlo noise:
   - P(the best of N samples is the true optimum) = 1 − (1 − p_opt)^N,
   - the expected cost of the best of N samples, from P(min ≥ v) = P(X ≥ v)^N.

The script writes three charts to `results/`:

- `quality_benchmark.png`: P(best of N = optimum) vs N, one panel per event.
- `quality_benchmark_gap.png`: expected extra cost above the optimum.
- `quality_benchmark_samples.png`: samples needed for 95 % confidence.

Results with the default settings, P(the best of N samples is the true optimum),
XY-QAOA (p = 2) vs simulated annealing (500 sweeps):

| event | qubits | N = 10 | N = 30 | N = 100 |
|---|---:|---:|---:|---:|
| k585 fails 07:00 | 14 | 0.54 vs 0.40 | 0.90 vs 0.78 | 1.00 vs 0.99 |
| k190 fails 05:00 | 15 | 0.39 vs 0.57 | 0.78 vs 0.92 | 0.99 vs 1.00 |
| k4 fails 06:00 | 20 | 0.58 vs 0.27 | 0.93 vs 0.61 | 1.00 vs 0.96 |
| S3 out 07:00–09:00 | 21 | 0.35 vs 0.16 | 0.73 vs 0.41 | 0.99 vs 0.83 |
| k585 fails 05:00 | 22 | 0.22 vs 0.04 | 0.52 vs 0.11 | 0.91 vs 0.33 |

- XY-QAOA needs fewer samples in 4 of 5 events, and the gap grows with the problem size.
- Penalty QAOA (n ≤ 18) is far behind both.
- How to read this:
  - One QAOA sample is one circuit shot. One SA sample is a full annealing run, which
    costs much more classically.
  - QAOA is simulated without noise, and its angle optimization is not counted.
  - On these sizes, the exact optimum by full replay takes under a second.

## Visualization and demo (work package G)

```sh
streamlit run app/demo.py                     # interactive demo in the browser
python scripts/make_figures.py                # figures/*.html for the pitch
python scripts/make_figures.py --bus k585 --at 07:00 --png
python scripts/make_figures.py --charger S3 --from 07:00 --to 09:00
```

The demo loads the 5-bus optimum of `columns/kcm_small_b20_bat200.json` on
`instances/kcm_small.json`:

- **Rozvrh**: Gantt chart (trips, deadhead transfers, charging with station),
  battery SoC of every bus with the minimum-SoC line, grid power per 15 min
  stacked by station, an exact E4 grid-power comparison for R=0 and R=1,
  and a duty table. The included 20-column pool has only one feasible schedule,
  so both R curves coincide on this instance.
- **Výpadok**: choose a failed bus and time (or a charger and outage window) and
  press *Pokaz autobus*. `evsp.demo.solve_disruption` builds the repair problem
  from package F, finds the exact optimum, and runs simulated annealing,
  XY-mixer QAOA and penalty QAOA with the X mixer. The page shows the
  before/after schedules (moved trips orange, cancelled trips red with hatching
  and a label), the cost change, P(optimum) and P(feasible) per method, shots for
  95 % confidence and the XY circuit size (all-to-all and heavy-hex).
- **Škálovanie**: penalty QAOA vs SA vs random guessing from
  `results/*_scaling.json` (pipeline step 6c). Without that file, the tab shows the
  command that creates it.

Penalty QAOA uses a 2^n statevector, so it is run live only up to 16 qubits.
Larger events use a matching `results/repair_<event>.json` from
`scripts/run_repair.py` (pipeline step 7). For example, the default event, bus `k4`
failing at 06:00, has 20 qubits and takes about 5 minutes. The demo uses the file only
if its event, trips, options and conflicts match the current problem. Otherwise it
shows the penalty-QAOA row as skipped. The live demo refuses
events with more than 100 000 one-hot states. For example, S3 out of service
06:00–08:00 has 60 qubits and 21 million states. Lower the number of reserves or
options per trip, or use `scripts/run_repair.py`.

PNG export needs `kaleido` and a Chrome or Chromium binary. If Chrome is not on
PATH, set `BROWSER_PATH`. Without a browser, the script writes HTML only.
Colors follow a palette checked for color-vision deficiency. Every status, such as
a failure, move, cancellation or outage, also has a text label.

## Setup and checks

Use Python 3.11 or newer; the pinned dependencies were tested with Python 3.13 and 3.14.
From the repository root, using the project's configured interpreter:

```sh
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/validate_instances.py
```

The tests use only files in git. A test that needs a results file writes its own copy
in a temporary folder, so a fresh clone passes all 178 tests.

Validation of the bundled instances does not need the original GTFS download.
The validation script writes `instances/baseline_validation.json`. Schema errors
stop execution with the file and JSON path; time-infeasible original block pairs
are reported as diagnostics and are not silently removed or made feasible.

## Rebuild from GTFS

The parser is specific to the February 2020 King County Metro feed distributed
in `dan-mccabe/beb-chargers-pactrans`, weekday service ID `86021`. In this feed,
`shape_dist_traveled` is in feet, not metres. It is not a general GTFS importer.

```sh
git clone https://github.com/dan-mccabe/beb-chargers-pactrans.git external/beb-chargers-pactrans
git -C external/beb-chargers-pactrans checkout <source-commit>
python scripts/build_instances.py --source-revision <source-commit>
```

Replace `<source-commit>` with the actual chosen commit. Every new build records
the supplied revision (or null), SHA256 hashes of all five input files, selection
parameters and assumptions. Pin the commit and retain those inputs to reproduce
an experiment; `generated_utc` will naturally differ on each run. The source
revision of the previously bundled JSON files is unknown and is explicitly null.
They have been migrated to schema 1.0 without regenerating their numerical data.
No claim of exact source reproducibility is made for those legacy files.

`--data-dir` overrides the input directory. `--depot 47.49,-122.27` supplies a depot;
otherwise its coordinates are the terminal centroid, explicitly marked synthetic.
`--preset tiny small` builds only those instances. Defaults currently create:

| Preset | Bundled trips | Selection |
| --- | ---: | --- |
| tiny | 8 | Route 150, at most two blocks and eight trips |
| small | 20 | Route 150, at most five blocks and twenty trips |
| medium | 133 | Route 150, full service day |
| large | 601 | All trips of blocks touching the study routes |

The medium preset exceeds the document's suggested 50–100 trips. An energy-stress
tiny and a fixed-seed synthetic generator remain follow-up work. The current tiny is
not evidence that charging constraints work.

## Contract: schema 1.0

`evsp/instance.schema.json` is the JSON Schema contract. `validate_instance(raw)`
adds cross-reference, uniqueness, finite-number and semantic checks before IDs
are converted to dictionary keys. `load_instance(path)` always validates;
`build_instance` and `save_instance` also validate their output. Unsupported or
missing schema versions fail explicitly, rather than being silently inferred.

```python
from evsp import load_instance

inst = load_instance("instances/kcm_tiny.json")
arcs = inst.arcs()
empty_leg_energy = inst.dh_energy_kwh(inst.depot, inst.trips[0].start)
diagnostics = inst.check_baseline()
```

- `trips`: unique ID, route, original GTFS block, start/end location, departure and
  arrival in minutes after service-day midnight, distance in km, energy in kWh.
  Times greater than 1440 are valid. Distances and energy are positive.
- `locations`: unique IDs and coordinates, with exactly one depot.
- `chargers`: each ID represents one connector serving at most one bus; multiple
  connectors are separate resources. `availability` distinguishes `assumed` from
  `confirmed`. All bundled connectors are assumed, including the candidate study
  sites. They are available in this hypothetical scenario, not verified real assets.
- `deadhead`: complete directed location-to-location matrix with km and minutes,
  nonnegative entries and zero diagonal. Symmetry is not required, so future road
  routing can use asymmetric travel times.
- `tariff`: contiguous ordered slots from midnight, covering all trip arrivals;
  `price` is currency units per grid kWh. `slot_min` is in minutes.
- `baseline_blocks`: every trip exactly once, ordered by departure. These are
  original GTFS groupings restricted to the selected data, not an optimized fleet
  or a guarantee of electric-bus feasibility.
- `assumptions`: battery, energy, charging, transfer, tariff and cost parameters.
  `peak_scope="global"` follows the document's global grid-power square. The
  charger `station` field is an informational grid-connection identifier; it does
  not change the objective into a sum of independent station peaks.
- `meta`: extensible provenance and notes. Other schema objects reject unknown
  fields so misspelled or incompatible contract changes are detected.

## Model assumptions and boundaries

The current case-study configuration uses 419.4 kWh usable capacity, 1.8641 kWh/km,
SoC 10–100%, initial SoC 100%, charging efficiency 90%, site connectors of 300 kW,
and ten assumed 150 kW depot connectors. These differ from the document's suggested
200–300 kWh, 1–1.5 kWh/km and 20–90% SoC. They are explicitly retained for the KCM
case study, not presented as the document's numerical defaults. The electric
experiments override the battery to 200 kWh (`--battery 200`) so that buses must
charge during the day.

Terminals use shape endpoints because the source lacks `stops.txt`. Deadhead
distance is great-circle distance times 1.3, at 30 km/h (the document suggests
approximately 25 km/h). Missing endpoint cumulative distances trigger a labelled
code fallback to endpoint distance times 1.3; no intermediate-stop distance is
substituted. Invalid coordinates, missing endpoints and nonpositive durations fail.

Energy of every empty leg is `km * kwh_per_km`, including depot and charging-site
detours. B must subtract that energy and trip energy, enforce the SoC limits, and
apply linear charging `efficiency * power_kw * minutes / 60`, capped by remaining
battery capacity. `can_follow()` checks time only; it does not certify energy or
charger availability. Physical energy validation is B's responsibility.

`arcs()` and `check_baseline()` both default to the configured five-minute layover.
Explicit `check_baseline(min_layover=0)` is useful to isolate transfer estimates.
The bundled large instance has 49/397 failing pairs at five minutes, versus 34/397
at zero. The report includes all failures, location/trip IDs, available time,
estimated travel time, layover, required time and positive shortfall. These are
known limitations of the data approximation, not proof that the operator's real
schedule was invalid. OSRM calibration can follow once basic optimization works.
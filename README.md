# E-VSP: data and instances (work package A)

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

This PR prepares versioned input data for duty generation (B), QUBO construction
(C), and quantum solvers (E). It does not generate battery-feasible duties or
solve a schedule. `main.py` is an independent Qiskit Bell-state smoke test.

## Classical baselines (work package D)

```sh
python scripts/run_baselines.py                  # diesel + electric + repair, writes results/baselines.json
python scripts/run_baselines.py --part diesel --instances large
python scripts/run_baselines.py --part electric --battery instance 200
python scripts/run_baselines.py --part repair
```

`evsp/baselines.py` contains the classical reference methods:

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

Results on the bundled data (battery 200 kWh for the electric part):

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
  `results/*_scaling.json` (`scripts/run_qaoa.py --scaling 8,10,12,16`).

Penalty QAOA uses a 2^n statevector, so it is run live only up to 16 qubits.
Larger events use a matching `results/repair_<event>.json` from
`scripts/run_repair.py`. The repository includes the result for bus `k4` at
06:00, which has 20 qubits and takes about 6 minutes. The live demo refuses
events with more than 100 000 one-hot states. For example, S3 out of service
06:00–08:00 has 60 qubits and 21 million states. Lower the number of reserves or
options per trip, or use `scripts/run_repair.py`.

PNG export needs `kaleido` and a Chrome or Chromium binary. If Chrome is not on
PATH, set `BROWSER_PATH`. Without a browser, the script writes HTML only.
Colors follow a palette checked for color-vision deficiency. Every status, such as
a failure, move, cancellation or outage, also has a text label.

## Setup and checks

Use Python 3.11 or newer; the pinned dependencies were tested with Python 3.14.3.
From the repository root, using the project's configured interpreter:

```sh
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/validate_instances.py
```

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
tiny, a disruption event format, and a fixed-seed synthetic generator remain
follow-up work. The current tiny is not evidence that charging constraints work.

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
case study, not presented as the document's numerical defaults.

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

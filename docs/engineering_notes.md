# Engineering Notes

## Performance characteristics

| Stage | Input | Samples | Time (approx) |
|-------|-------|---------|--------------|
| load_csv | 1 lap CSV, 100 Hz, 90 s | 9,000 | < 50 ms |
| clean | 9,000 samples, 12 signals | — | < 100 ms |
| segment | 9,000 samples | — | < 150 ms |
| delta (two laps) | 18,000 samples | 5,000 grid | < 80 ms |
| **Total** | | | **< 400 ms** |

Measured on synthetic data, single-core, Python 3.11, MacBook M2.
Real CSVs with 30+ signals will take proportionally longer in the clean stage.

---

## Known limitations and open issues

### Lap boundary detection is heuristic

The current detection relies on `lap_time` or `distance` resets. Some logging
platforms do not reset either signal — they export continuous time and distance
across the session. In that case, all laps will be merged into one.

**Resolution**: Add optional `lap_boundary_times` argument to `load_csv()`
to allow explicit lap boundary specification.

### Corner segmentation detects kinematic corners only

The segmentation detects braking and speed loss events. It does not use circuit
map geometry. This means:

- Chicanes may be merged or split inconsistently.
- Low-speed hairpins with engine braking only may be missed.
- Flat-out kinks will not appear unless a speed reduction is detectable.

**Resolution**: Add FastF1 circuit geometry as an optional constraint source
(corner detection guided by known corner positions).

### Smoothing introduces event timing error

The centred rolling mean shifts brake onset and throttle pickup by approximately
half the window width divided by the sample rate. At 100 Hz and window 5, this
is 25 ms = approximately 1.4 m at 200 km/h.

**Resolution**: For brake point analysis, use the raw (despiked, unsmoothed)
signal. The `clean()` function should optionally return both smoothed and
unsmoothed data.

### Delta narrative is rule-based, not causal

The narrative engine assigns a single cause per corner. In reality multiple
causes may overlap (e.g. a driver can brake later AND carry more apex speed
simultaneously). The current rules attribute priority in a fixed order.

**Resolution**: Implement a weighted scoring system that quantifies the
contribution of each variable to the total time delta.

---

## Dependency notes

| Package | Version tested | Purpose |
|---------|---------------|---------|
| pandas  | 2.2.x | DataFrame operations |
| numpy   | 1.26.x | Numerical computation |
| matplotlib | 3.8.x | Plotting (optional) |
| fastf1  | 3.3.x | Official F1 data (optional, future) |
| scipy   | 1.12.x | Signal processing (future use) |

No dependencies beyond pandas and numpy are required for the core pipeline.

---

## Commit history — telemetry module

| Commit message | Scope |
|----------------|-------|
| `feat(loader): add CSV ingestion with alias resolution` | loader.py |
| `feat(loader): add physical bounds validation` | loader.py |
| `feat(loader): add TelemetryLap dataclass` | loader.py |
| `feat(cleaner): implement clip and NaN interpolation` | cleaner.py |
| `feat(cleaner): add rolling z-score spike detection` | cleaner.py |
| `feat(cleaner): add signal smoothing with documented windows` | cleaner.py |
| `feat(cleaner): add CleaningReport audit trail` | cleaner.py |
| `feat(segmentation): implement state machine corner detector` | segmentation.py |
| `feat(segmentation): add corner classification and DRS detection` | segmentation.py |
| `feat(delta): implement distance-aligned cumulative delta` | delta.py |
| `feat(delta): add corner matching and per-corner analysis` | delta.py |
| `feat(delta): implement engineering narrative engine` | delta.py |
| `docs: add assumptions.md` | docs/ |
| `docs: add methodology.md` | docs/ |
| `docs: add engineering_notes.md` | docs/ |

---

## Planned improvements

### Telemetry (current priority)

- [ ] FastF1 adapter (`loader_fastf1.py`)
- [ ] Mini-sector analysis (track divided into 3×3 = 9 sectors)
- [ ] Straight-line performance analysis (DRS efficiency, ERS contribution)
- [ ] Brake balance estimation from brake temperature differential
- [ ] Tyre degradation model from lap-over-lap speed comparison

### Strategy (next phase — do not start until telemetry is stable)

- [ ] Pit window optimiser
- [ ] Undercut/overcut simulation
- [ ] Safety car scenario model

### Aero (later phase)

- [ ] Drag estimation from coast-down sections
- [ ] Downforce index from lateral G vs speed relationship

### Setup (later phase)

- [ ] Spring rate proxy from suspension travel
- [ ] Brake bias recommendation from temperature differential

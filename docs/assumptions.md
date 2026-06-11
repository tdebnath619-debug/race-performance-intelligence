# Assumptions

## §1 — Telemetry Ingestion

### §1.1 Time column
The time column is assumed to represent elapsed time since logging start,
in seconds or milliseconds. If the maximum value exceeds 10,000 the column
is treated as milliseconds and divided by 1,000.

Limitation: If a session runs for more than 2.7 hours and uses millisecond
timestamps, this heuristic will misclassify the unit. In that case, pass
the pre-normalised file or add an explicit unit argument to `load_csv()`.

### §1.2 Throttle and brake normalisation
Throttle and brake are assumed to be either 0–1 or 0–100. If the maximum
value of either exceeds 1.5, the column is divided by 100.

Limitation: Some platforms output values in the range 0–4096 (12-bit ADC).
These will not be caught by this heuristic. Add an explicit normalisation
step in the platform-specific adapter.

### §1.3 Distance from speed integration
When a distance column is absent, cumulative distance is computed as:

```
d(t) = ∫ v(t) dt
```

where v is in km/h, converted to m/s before integration.

Accumulated error from this method is typically < 1% over a 5 km lap
at a 50 Hz sample rate, assuming speed measurement error is zero-mean.
GPS-derived speed carries additional error at low speeds and under
acceleration. Use a known lap length to normalise if precision is required.

### §1.4 Lap boundary detection
Lap boundaries are detected from one of two signals:

1. `lap_time` decreasing by more than 5 seconds — indicates a new lap.
2. `distance` decreasing by more than 50 metres — indicates a distance reset.

Assumption: The logging system resets either lap_time or distance at each
lap boundary. If neither signal is present, the entire file is treated as
a single lap.

---

## §2 — Signal Processing

### §2.1 Physical bounds
Physical bounds are hard limits derived from engineering knowledge of the
vehicle category (Formula 1 / high-downforce racing car). Values outside
these bounds are treated as sensor errors.

Assumption: A speed of 380 km/h is the absolute upper bound. Any value above
this is a sensor artefact.

Limitation: For different vehicle categories (GT, LMP, Formula E), bounds
must be reviewed and updated in `cleaner.py → _CLIP_BOUNDS`.

### §2.2 Spike detection
Spikes are detected using a rolling z-score. A sample is flagged if:

```
|x - μ_rolling| / σ_rolling > threshold
```

where μ and σ are computed over a 30-sample rolling window.

Assumption: The rolling window of 30 samples is shorter than any legitimate
signal event. At 100 Hz this is 300 ms — shorter than the shortest braking
zone on any circuit.

Limitation: Multi-sample anomalies (sustained noise bursts) will not be
detected. The z-score method is designed for single-sample glitches only.

### §2.3 Smoothing window selection
Windows are chosen to attenuate noise frequencies above the signal bandwidth
of each physical process.

| Signal   | Window | Rationale |
|----------|--------|-----------|
| speed    | 5      | 50 ms at 100 Hz — noise below tyre response bandwidth |
| steering | 7      | Driver steering input is band-limited to ~2 Hz |
| tyre_temp| 11     | Thermal time constant >> 100 ms |

---

## §3 — Corner Segmentation

### §3.1 Apex definition
The apex is defined as the sample of minimum speed within the braking phase
window. This is a kinematic definition, not a geometric one (i.e. it does
not correspond to the geometric apex of the circuit).

Assumption: The minimum speed sample occurs after the driver has reached
maximum braking and before throttle pickup. At very low-speed hairpins this
may be violated if the driver is simultaneously braking and turning.

### §3.2 Chicane treatment
Two consecutive corners within 50 metres are merged into a single event.
This is intentional — the analysis treats chicanes as one braking zone.

Limitation: If the two elements of a chicane have very different minimum
speeds (e.g. Monza Lesmo 1 and 2), the merged minimum will reflect only
the slower element.

### §3.3 Minimum speed threshold
A corner must produce a speed reduction of at least 10 km/h to be accepted.
This rejects road surface undulations and sensor noise that produce apparent
speed minima without a genuine braking event.

---

## §4 — Delta Analysis

### §4.1 Distance-based alignment
Laps are aligned on a common distance grid. This is preferred over time
alignment because it preserves the physical correspondence between track
position and driver inputs.

Assumption: Both laps were recorded on the same circuit in the same direction
and with similar lap lengths (within 5%). Significantly different lap lengths
indicate a track limit violation or a timing error.

### §4.2 Time delta sign convention
Positive time delta → Lap A is faster at that distance.
Negative time delta → Lap B is faster at that distance.

This convention is consistent with a stopwatch: a lower cumulative time
corresponds to a faster passage through that section.

### §4.3 Narrative attribution
The engineering narrative is rule-based, not model-based. It applies a fixed
priority order to attribute each corner delta to the most likely cause.
This is an approximation — multiple causes can contribute simultaneously.

Limitation: The narrative cannot distinguish between a setup change and a
driving style change without additional context (e.g. fuel load, tyre age).

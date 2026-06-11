# Methodology

## §1 — Architecture

The system is structured as a linear processing pipeline:

```
RAW CSV
  │
  ▼
loader.py          — ingestion, validation, unit normalisation
  │
  ▼
cleaner.py         — signal processing (clip, interpolate, despike, smooth)
  │
  ▼
segmentation.py    — corner detection, track partitioning
  │
  ▼
delta.py           — distance-aligned delta analysis, engineering narrative
  │
  ▼
ComparisonReport   — structured output for further use
```

Each stage has a defined input contract (TelemetryLap) and a defined output.
Stages do not share state. The pipeline can be interrupted and inspected at
any stage for debugging.

---

## §2 — Signal Processing

### §2.1 Overview

Five sequential steps are applied:

1. **Clip** — hard-limit to physical bounds (removes ADC overflow, saturation)
2. **Interpolate NaN** — linear gap fill for logging dropouts
3. **Despike** — rolling z-score spike rejection
4. **Smooth** — centred rolling mean per signal
5. **Differentiate** — first-order time derivatives for analysis

### §2.2 Rationale for processing order

Clipping before interpolation ensures that saturated values do not contaminate
the interpolated region. Despiking after interpolation reduces the risk of
introducing new spikes through interpolation across large NaN gaps. Smoothing
after despiking ensures the filter operates on clean data.

### §2.3 Centred vs causal filters

All smoothing filters are centred (symmetric window). This introduces zero
phase shift, which is essential for correct brake point timing analysis.

A causal filter (e.g. exponential moving average) would introduce a lag
equal to half the window length. At 100 Hz and a window of 7, this would
misplace events by 35 ms — equivalent to approximately 2 metres at 200 km/h.

### §2.4 What is not smoothed

The derivative signals (`d_speed`, `d_throttle`, `d_brake`) are not smoothed.
They are intentionally left at their raw resolution so that brake onset and
throttle pickup events remain as sharp as possible.

---

## §3 — Corner Segmentation

### §3.1 State machine

The detector operates as a simple two-state machine:

```
SCANNING ──(braking onset detected)──▶ IN_CORNER
IN_CORNER ──(apex found + exit found + validity passed)──▶ SCANNING
IN_CORNER ──(max_samples exceeded)──▶ SCANNING (event abandoned)
```

### §3.2 Apex search

After braking onset, the algorithm searches forward up to `APEX_LOOKAHEAD`
samples for the speed minimum. The minimum must not be at the edges of the
search window — this ensures the full apex event was captured.

If no valid apex is found within the window, the search continues until the
maximum sample limit is reached.

### §3.3 Exit search

After the apex, the algorithm searches forward up to `EXIT_LOOKAHEAD` samples
for the first sample where throttle exceeds `THROTTLE_THRESHOLD` (0.10).

If no throttle pickup is found (e.g. for a double-apex corner), the exit is
placed 80 samples after the apex.

### §3.4 Corner classification

| Type   | Apex speed |
|--------|-----------|
| slow   | < 100 km/h |
| medium | 100–180 km/h |
| fast   | 180–240 km/h |
| kink   | > 240 km/h |

---

## §4 — Delta Analysis

### §4.1 Common distance grid

Both laps are interpolated onto a common distance grid with 5,000 points
across the overlapping distance range. Linear interpolation is used.

5,000 points over a 5,000 m lap = 1 m resolution, sufficient for all
motorsport analysis applications at current data rates.

### §4.2 Cumulative time delta derivation

The time taken to travel distance increment Δd at speed v is:

```
Δt = Δd / v
```

The cumulative time delta between laps A and B at distance d is:

```
ΔT(d) = ∫₀ᵈ (1/v_B - 1/v_A) dx
```

This gives the time by which Lap A is ahead of (positive) or behind (negative)
Lap B at any track position.

### §4.3 Corner matching tolerance

Corners are matched by apex distance. The maximum allowable mismatch is
200 metres. This is large enough to accommodate lap-to-lap variation in
braking points and small differences in the apex detection result, but
small enough to prevent incorrect matching on circuits with closely spaced
corners.

### §4.4 Narrative priority

The narrative engine applies the following priority order to classify each
corner delta:

| Priority | Condition | Classification |
|----------|-----------|----------------|
| 1 | ΔvEntry ≥ 5 km/h | braking_zone |
| 2 | Δv_apex ≥ 3 km/h | corner_speed |
| 3 | Δthrottle_dist ≥ 10 m | throttle_application |
| 4 | ΔvExit ≥ 5 km/h | corner_exit |
| — | None met | unclassified |

Thresholds are derived from typical lap-to-lap variability in professional
motorsport. Smaller deltas are unlikely to be reproducible and are not
attributed to a specific cause.

---

## §5 — FastF1 Integration (Planned)

FastF1 provides official F1 telemetry at 240 Hz. The integration plan:

1. Add `telemetry/loader_fastf1.py` as a separate adapter.
2. The adapter converts FastF1 channel names to the internal signal names.
3. The returned `TelemetryLap` object is identical to the CSV-loaded version.
4. No changes required in cleaner, segmentation, or delta modules.

Target sessions for initial validation:
- Bahrain 2024 Qualifying (clean lap, no traffic)
- Monza 2023 Qualifying (DRS comparison, slipstream effect)

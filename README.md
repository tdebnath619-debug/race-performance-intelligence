# race-performance-intelligence

Motorsport engineering analysis system. Built to demonstrate skills relevant to Junior Performance Engineer, Motorsport Data Engineer, and Simulation Engineer roles.

---

## Current status

| Module | Status | Notes |
|--------|--------|-------|
| Telemetry ingestion | ✅ Complete | TelemetryLap dataclass, 22+ channels |
| Signal processing | ✅ Complete | 5-step documented pipeline |
| Corner segmentation | ✅ Complete | State-machine detection |
| Delta analysis | ✅ Complete | Distance-aligned, engineering narrative |
| FastF1 integration | 🔄 Planned | See docs/methodology.md §5 |
| Race strategy | ⏸ Stable | Not under active development |
| Aero modelling | ⏸ Stable | Not under active development |
| Setup optimisation | ⏸ Stable | Not under active development |

---

## Quick start

```bash
pip install -r requirements.txt
<<<<<<< HEAD
python run_analysis.py                        # synthetic Bahrain circuit
python run_analysis.py data/raw/my_lap.csv    # your telemetry file
=======
python run_f1_analysis.py            # runs on synthetic Bahrain circuit data
python run_f1_analysis.py my.csv     # run on  telemetry CSV
>>>>>>> 7e8d7e5e76bd9a14cfc8821f84ba13b7e1f00306
```

---

## Telemetry pipeline

```
CSV file
  │
  ▼
loader.py       — ingestion, validation, unit normalisation → TelemetryLap
  │
  ▼
cleaner.py      — clip → interpolate → despike → smooth → differentiate
  │
  ▼
segmentation.py — corner detection → SegmentationResult
  │
  ▼
delta.py        — distance-aligned delta → ComparisonReport
  │
  ▼
reports/        — JSON, CSV, PNG chart
```

---

## Supported telemetry channels

| Category | Channels |
|----------|---------|
| Core | time, distance, speed, throttle, brake, gear, rpm, steering |
| Power unit | ers_deployment_kw, ers_harvesting_kw, fuel_kg, drs_state |
| Dynamics | lateral_g, longitudinal_g |
| Temperatures | brake_temp_fl/fr/rl/rr, tyre_temp_fl/fr/rl/rr |

---

## Documentation

- `docs/assumptions.md` — all signal and algorithmic assumptions
- `docs/methodology.md` — pipeline derivation and parameter rationale
- `docs/engineering_notes.md` — known limitations and planned improvements
- `CHANGELOG.md` — engineering commit progression

---

## Output example

```
DELTA: VER (A) vs LEC (B)   Session: Bahrain_2024_Q

KEY FINDINGS
  • Largest delta: T3 (slow) — 0.071 s — braking zone
  • VER brakes later at 3 corners: [3, 7, 11]
  • LEC picks up throttle earlier at 2 corners: [5, 9]

CORNER-BY-CORNER
  T    Type        Δt(s)   ΔvEntry    ΔvApex    ΔvExit
  T3   slow       +0.071     +8.2      +1.1      +3.4
       → VER carries 8.2 km/h more entry speed (brakes 12m later).
  T7   slow       -0.043     -2.1      +3.8      +2.2
       → LEC achieves 3.8 km/h higher minimum speed.
```

---

## Repository structure

```
race-performance-intelligence/
├── telemetry/
│   ├── loader.py        — ingestion
│   ├── cleaner.py       — signal processing
│   ├── segmentation.py  — corner detection
│   ├── delta.py         — delta analysis ← core output
│   ├── metrics.py       — per-lap/corner metrics
│   ├── plot.py          — comparison charts
│   └── report.py        — JSON/CSV export
├── strategy/            — pit window, undercut, safety car (stable)
├── aero/                — drag, downforce (stable)
├── setup/               — brake bias, tyre, diff (stable)
├── docs/
│   ├── assumptions.md
│   ├── methodology.md
│   └── engineering_notes.md
├── data/raw/
│   └── generate_telemetry.py
├── CHANGELOG.md
├── run_analysis.py
└── requirements.txt
```

---

## Requirements

```
pandas>=2.0
numpy>=1.25
matplotlib>=3.7
scipy>=1.11
fastf1>=3.3    # optional — real F1 data
```

# race-performance-intelligence

Motorsport engineering analysis system. Built to demonstrate skills relevant to Junior Performance Engineer, Motorsport Data Engineer, and Simulation Engineer roles.

---

## Live dashboard

**[View live →](https://tdebnath619-debug.github.io/race-performance-intelligence)**

---

## Quick start

```bash
pip install -r requirements.txt

# Synthetic data
python run_analysis.py

# Real F1 data
python run_analysis.py --fastf1 --year 2024 --gp Bahrain --session Q --driver_a VER --driver_b LEC

# Your own CSV
python run_analysis.py data/raw/my_lap.csv

# Run tests
pytest tests/ -v
```

---

## Current status

| Module | Status | Notes |
|--------|--------|-------|
| Telemetry ingestion | ✅ Complete | TelemetryLap dataclass, 25+ channels |
| Signal processing | ✅ Complete | 5-step documented pipeline |
| Corner segmentation | ✅ Complete | v2 multi-signal detection |
| Delta analysis | ✅ Complete | Distance-aligned, engineering narrative |
| FastF1 integration | ✅ Complete | Real F1 data — qualifying and race sessions |
| Test suite | ✅ Complete | 8 passing unit tests |
| CI/CD pipeline | ✅ Complete | GitHub Actions on every push |
| Live dashboard | ✅ Complete | Auto-generated from pipeline output |
| Race strategy | ⏸ Stable | Not under active development |
| Aero modelling | ⏸ Stable | Not under active development |
| Setup optimisation | ⏸ Stable | Not under active development |

---

## Repository structure

```
race-performance-intelligence/
├── telemetry/
│   ├── loader.py           — ingestion
│   ├── loader_fastf1.py    — real F1 data via FastF1
│   ├── cleaner.py          — signal processing
│   ├── segmentation.py     — corner detection
│   ├── delta.py            — delta analysis
│   ├── metrics.py          — per-lap/corner metrics
│   ├── dashboard.py        — HTML dashboard generator
│   ├── plot.py             — comparison charts
│   └── report.py           — JSON/CSV export
├── tests/                  — 8 passing unit tests
├── docs/
│   ├── assumptions.md
│   ├── methodology.md
│   └── engineering_notes.md
├── data/raw/
│   └── generate_telemetry.py
├── .github/workflows/
│   └── pipeline.yml
├── CHANGELOG.md
├── run_analysis.py
└── requirements.txt
```

---

## Telemetry pipeline

```
CSV / FastF1
     │
     ▼
loader.py          ingestion, validation, unit normalisation
     │
     ▼
cleaner.py         clip, interpolate, despike, smooth, differentiate
     │
     ▼
segmentation.py    corner detection, SegmentationResult
     │
     ▼
delta.py           distance-aligned delta, engineering narrative
     │
     ▼
reports/           JSON, CSV, PNG chart, index.html dashboard
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

## Example output

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

## Documentation

| File | Content |
|------|---------|
| `docs/assumptions.md` | All signal and algorithmic assumptions |
| `docs/methodology.md` | Pipeline derivation and parameter rationale |
| `docs/engineering_notes.md` | Known limitations and planned improvements |
| `CHANGELOG.md` | Engineering commit progression |

---

## Requirements

```
pandas>=2.0
numpy>=1.25
matplotlib>=3.7
scipy>=1.11
pytest>=7.0
fastf1>=3.3
```

---

## Target roles

- Junior Performance Engineer
- Motorsport Data Engineer
- Simulation Engineer
- Motorsport Software Engineer

# F1 Performance Intelligence System

> Production-grade F1 telemetry analysis platform covering telemetry, aerodynamics, race strategy, and car setup optimisation.

---

## Quick Start

```bash
pip install -r requirements.txt
python run_f1_analysis.py            # runs on synthetic Bahrain circuit data
python run_f1_analysis.py my.csv     # run on your own telemetry CSV
```

---

## What It Does (F1-Job Relevant Features)

| Module | Capability | Real F1 Equivalent |
|--------|-----------|-------------------|
| `telemetry/loader` | 30+ channel ingestion, FastF1 API support | Data engineer / MoTeC import |
| `telemetry/cleaner` | ERS budget validation, DRS latching, physical bounds | Signal processing / QA |
| `telemetry/segmentation` | 15-corner detection, mini-sectors, DRS zones | Track engineer segmentation |
| `telemetry/metrics` | Entry/apex/exit speed, ERS per corner, downforce index, fuel-corrected lap time | Performance analyst |
| `telemetry/comparison` | Distance-aligned delta, corner coaching narrative, ERS delta | Race engineer debrief tool |
| `aero/aero_analysis` | Cd proxy from coast-down, downforce index, DRS quantification, balance estimate | Aero performance engineer |
| `strategy/race_strategy` | Strategy optimiser, undercut/overcut sim, safety car model | Strategy engineer |
| `setup/setup_optimizer` | Brake bias, tyre pressure, diff, wing recommendations from telemetry | Setup/vehicle engineer |

---

## Project Structure

```
f1-performance-system/
│
├── run_f1_analysis.py          ← full pipeline (run this)
├── requirements.txt
│
├── telemetry/
│   ├── loader.py               ← CSV + FastF1 ingestion
│   ├── cleaner.py              ← F1-aware signal cleaning
│   ├── segmentation.py         ← corners, mini-sectors, DRS zones
│   ├── metrics.py              ← per-corner + lap metrics incl. ERS
│   ├── comparison.py           ← lap delta + coaching narrative
│   └── report.py               ← JSON / CSV export
│
├── aero/
│   └── aero_analysis.py        ← Cd, downforce index, DRS gain
│
├── strategy/
│   └── race_strategy.py        ← pit windows, undercut, SC model
│
├── setup/
│   └── setup_optimizer.py      ← brake bias, tyres, diff, wing recs
│
├── data/
│   └── raw/
│       └── generate_f1_telemetry.py   ← synthetic Bahrain circuit data
│
└── reports/                    ← auto-generated outputs
    ├── f1_analysis.json
    ├── f1_analysis_corners.csv
    ├── f1_analysis_strategies.csv
    └── f1_comparison_chart.png
```

---

## Telemetry Channels (30+)

| Category | Channels |
|----------|----------|
| Core | time, distance, speed, throttle, brake, gear, rpm, steering |
| Power Unit | ers_deployment_kw, ers_harvesting_kw, mguk_kw, mguh_kw, fuel_flow_kgh, fuel_kg |
| Chassis | drs_state, g_lateral, g_longitudinal, g_vertical |
| Brakes | brake_temp_fl/fr/rl/rr |
| Tyres | tyre_surf_fl/fr/rl/rr |
| Suspension | susp_travel_fl/fr/rl/rr |

---

## FastF1 Integration (live F1 data)

```python
from telemetry.loader import load_fastf1

df = load_fastf1(session_year=2024, gp="Bahrain", session="R", driver="VER")
```

---

## Future Modules (ready for integration)

- `lap_time_simulator` — physics-based lap time from car parameters
- `weather_model` — tyre compound selection under changing conditions
- `rival_tracker` — multi-driver strategy tracker during live race
- `setup_ml` — ML-based setup prediction from historical data

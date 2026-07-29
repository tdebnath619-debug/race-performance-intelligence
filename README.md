# race-performance-intelligence

Motorsport engineering telemetry analysis system. Built to demonstrate skills relevant to Junior Performance Engineer, Motorsport Data Engineer, and Simulation Engineer roles in professional motorsport.

---

## Live dashboard

**[View live →](https://tdebnath619-debug.github.io/race-performance-intelligence)**

---

## Quick start

```bash
# Synthetic data (no internet required)
pip install -r requirements.txt
python run_analysis.py

# Real F1 data — Verstappen vs Leclerc, Bahrain 2024 Qualifying
python run_analysis.py --fastf1 --year 2024 --gp Bahrain --session Q --driver_a VER --driver_b LEC

# Your own CSV
python run_analysis.py data/raw/my_telemetry.csv

# Run tests
pytest tests/ -v
```

---

## System status

| Module | Status | Description |
|--------|--------|-------------|
| Telemetry ingestion | ✅ Complete | 25+ channels, alias resolution, validation |
| Signal processing | ✅ Complete | 5-step documented pipeline, audit trail |
| Corner segmentation | ✅ v2 | Multi-signal detection, all corner types |
| Delta analysis | ✅ Complete | Distance-aligned, engineering narrative |
| FastF1 integration | ✅ Complete | Real F1 data, qualifying & race |
| Test suite | ✅ Complete | 25 unit + integration tests |
| CI/CD | ✅ Complete | GitHub Actions on every push |
| Live dashboard | ✅ Complete | Auto-generated from pipeline output |
| Race strategy | ⏸ Stable | Pit window, undercut, safety car models |
| Aero modelling | ⏸ Stable | Cd proxy, downforce index |
| Setup optimisation | ⏸ Stable | Brake bias, tyre, differential |

---

## Pipeline architecture

```
CSV / FastF1
     │
     ▼
loader.py          ingestion · validation · unit normalisation · TelemetryLap
     │
     ▼
cleaner.py         clip → interpolate → despike → smooth → differentiate
                   CleaningReport audit trail per lap
     │
     ▼
segmentation.py    multi-signal corner detection (v2)
                   brake signal + d_speed + steering angle
                   SegmentationResult with corner/straight timing split
     │
     ▼
delta.py           distance-aligned cumulative time delta
                   ΔT(d) = ∫(1/v_B − 1/v_A) dx
                   engineering narrative: braking_zone · corner_speed ·
                   throttle_application · corner_exit
     │
     ▼
reports/           JSON · CSV · PNG chart · index.html dashboard
```

---

## FastF1 real data

```python
from loader_fastf1 import load_fastf1_comparison, session_info

# See all drivers in session
print(session_info(2024, "Bahrain", "Q"))

# Load and compare
lap_ver, lap_lec = load_fastf1_comparison(2024, "Bahrain", "Q", "VER", "LEC")
```

---

## Example output

```
DELTA: VER (A) vs LEC (B) | Session: Bahrain_2024_Q

KEY FINDINGS
  • Largest delta: T3 (slow) — 0.071 s — braking zone
  • VER brakes later at 3 corners: [3, 7, 11]
  • LEC picks up throttle earlier at 2 corners: [5, 9]

CORNER-BY-CORNER
  T    Type        Δt(s)   ΔvEntry    ΔvApex    ΔvExit
  T3   slow       +0.071     +8.2      +1.1      +3.4
       → VER carries 8.2 km/h more entry speed (brakes 12m later). VER gains 0.071 s.
  T8   slow       -0.043     -2.1      +3.8      +2.2
       → LEC achieves 3.8 km/h higher minimum speed. LEC gains 0.043 s.
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

## Telemetry channels (25+)

| Category | Channels |
|----------|---------|
| Core | time, distance, speed, throttle, brake, gear, rpm, steering |
| Power unit | ers_deployment_kw, ers_harvesting_kw, fuel_kg, drs_state |
| Dynamics | lateral_g, longitudinal_g |
| Temperatures | brake_temp_fl/fr/rl/rr, tyre_temp_fl/fr/rl/rr |

---

## Target roles

- Junior Performance Engineer
- Motorsport Data Engineer  
- Simulation Engineer
- Motorsport Software Engineer

---

## Requirements

```
pandas>=2.0 · numpy>=1.25 · matplotlib>=3.7 · scipy>=1.11 · pytest>=7.0 · fastf1>=3.3
```

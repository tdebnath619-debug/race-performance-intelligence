# Changelog

## [v2.0] — Telemetry Module Complete

### New features
- `feat(segmentation): v2 multi-signal corner detector — brake + d_speed + steering`
- `feat(segmentation): adaptive apex search with steering fallback for hairpins`
- `feat(loader_fastf1): FastF1 adapter for real F1 data — qualifying and race sessions`
- `feat(loader_fastf1): session_info() — driver comparison table for any session`
- `feat(run_analysis): CLI with --fastf1 --year --gp --session --driver_a --driver_b`
- `feat(tests): 25 unit and integration tests across all pipeline stages`
- `fix(dashboard): rewrite using string concatenation — no f-string JS escaping`
- `fix(dashboard): addEventListener tab switching — all 4 tabs functional`
- `ci: GitHub Actions runs tests then pipeline on every push`
- `ci: reports and dashboard uploaded as build artifacts`

### Engineering improvements
- Corner detection now catches all 15 corners on Bahrain-length circuit
- Minimum speed drop threshold reduced 10→5 km/h (catches chicane elements)
- Minimum corner gap reduced 50→30 m (handles tight chicane pairs)
- Apex lookahead increased 100→150 samples (handles long braking zones)
- Steering signal used as apex fallback for engine-braking hairpins

## [v1.0] — Initial system

- Telemetry ingestion (CSV, 22+ channels)
- Signal cleaning pipeline (5 documented steps)
- Corner segmentation (state machine)
- Delta analysis (distance-aligned)
- Lap metrics (per-corner and per-lap)
- Race strategy engine (pit window, undercut, safety car)
- Aero analysis (Cd proxy, downforce index, DRS gain)
- Setup optimiser (brake bias, tyre pressure, differential, wing)
- GitHub Pages dashboard

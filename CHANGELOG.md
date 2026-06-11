# Changelog

## [Unreleased] — Telemetry Module v2

### Engineering progression — commit by commit

```
feat(loader): add TelemetryLap dataclass with provenance metadata
feat(loader): add alias resolution for MoTeC, Pi Toolbox, ACC, iRacing
feat(loader): add physical bounds validation on ingestion
feat(loader): add automatic unit normalisation (time, throttle, steering)
feat(loader): add distance synthesis from speed integration
feat(cleaner): implement CleaningReport audit trail
feat(cleaner): add physical bounds clipping (Step 1)
feat(cleaner): add NaN interpolation with gap logging (Step 2)
feat(cleaner): add rolling z-score spike detection (Step 3)
feat(cleaner): add centred rolling mean smoothing (Step 4)
feat(cleaner): add time-derivative computation (Step 5)
feat(segmentation): implement two-state corner detection machine
feat(segmentation): add apex search with edge rejection
feat(segmentation): add throttle-pickup exit detection
feat(segmentation): add SegmentationResult with corner/straight timing
feat(delta): implement distance-aligned cumulative time delta
feat(delta): add corner matching with apex distance tolerance
feat(delta): add per-corner entry/apex/exit speed analysis
feat(delta): add engineering narrative attribution engine
feat(delta): add ComparisonReport with JSON export
feat(metrics): add per-corner CornerMetrics dataclass
feat(metrics): add LapMetrics with ERS and G-force summary
feat(plot): add dark-theme comparison chart (4-panel)
docs: add assumptions.md covering all ingestion assumptions
docs: add methodology.md with pipeline derivation
docs: add engineering_notes.md with known limitations
```

## [v1] — Initial system

- Basic telemetry ingestion (CSV)
- Signal cleaning (smoothing, spike removal)
- Corner segmentation
- Lap comparison
- Race strategy engine (pit window, undercut, safety car)
- Aero analysis (Cd proxy, downforce index)
- Setup optimizer (brake bias, tyre pressure, differential)
- GitHub Pages dashboard

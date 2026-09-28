# Publication Manifest

## Public-safe source modules

The `src/` directory contains selected, previously validated modules chosen because they are understandable in isolation and representative of the engineering work:

- `forza_packet_decoder.py` — fixed-size FH6 binary packet decoding and plausibility support
- `forza_udp_receiver.py` — UDP transport layer
- `forza_session_logger.py` — native CSV capture/logging
- `forza_adapter.py` — native-to-normalized session adaptation and distance-source handling
- `analyze_forza_session.py` — session metrics and event analysis
- `audit_forza_lap_readiness.py` — formal-lap readiness auditing
- `extract_forza_complete_laps.py` — complete-lap extraction
- `normalize_forza_laps.py` — common-distance lap normalization
- `select_forza_reference_lap.py` — automatic reference selection
- `compare_forza_normalized_laps.py` — real-lap comparison and delta analysis
- `detect_forza_corner_segments.py` — telemetry-derived semantic segmentation
- `build_forza_corner_analysis.py` — corner-aware comparative analysis
- `plot_forza_lap_comparison.py` — comparative visualization
- `plot_forza_corner_analysis.py` — semantic-analysis visualization

## Representative figures

- `real-session-overview.png`
- `lap-speed-comparison.png`
- `lap-time-delta.png`
- `driver-input-comparison.png`
- `reference-corner-detection.png`
- `semantic-segment-delta.png`

## Public sample data

- `reference-lap-03-normalized.csv` — selected engineering channels from the 400-point validated real reference lap
- `lap-02-vs-lap-03-comparison.csv` — selected channels and time-delta fields from a validated real-lap comparison
- `semantic-segment-boundaries.csv` — public-safe V1.1 semantic segment boundaries/metrics
- `validation-anchors.csv` — compact protected numerical anchors

## Deliberately excluded

- raw/native telemetry archives containing network metadata
- internal machine paths
- private development and recovery records
- repository history from the canonical engineering clone
- unused/experimental scripts
- full generated-output trees
- caches, binaries, ZIP backups, and local configuration

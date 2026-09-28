# Development Status and Limitations

## Current state

**In Development — currently paused between validated milestones.**

The latest protected technical state is V1.1 Corner-Aware Semantic Analysis. The next analytical milestone has **not** been selected. The private roadmap lists bounded future directions such as corner-phase modeling (braking / turn-in / apex / exit) and position-based track-map reconstruction, but neither is part of the current validated baseline.

## Interpretation boundaries

- FH6 telemetry is game telemetry; acceleration/g channels are not presented as calibrated real-vehicle measurements.
- Controlled free-roam captures are engineering test captures, not laboratory-grade tests.
- The formal-lap dataset is valid for the implemented pipeline but is not presented as a professional race-engineering dataset.
- Ten fixed sectors are mathematical controls, not semantic track features.
- V1.1 corner boundaries are telemetry-derived semantic driving regions, not official geometric track corners.
- Historical synthetic V0.3 testing established comparison mechanics but contains a documented integrated-delta consistency limitation (approximately -0.817 s integrated vs approximately -0.600 s raw end-timestamp difference). Synthetic validation is not used as proof of FH6 physical fidelity.

## Publication boundary

This repository is a curated public interface, not the full development repository. Internal continuity/recovery documents, raw capture archives, machine-specific reports, abandoned experiments, and development debris are intentionally excluded.

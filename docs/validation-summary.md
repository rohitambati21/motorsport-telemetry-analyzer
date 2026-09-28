# Validation Summary

This public summary is derived from the frozen validation reports in the authoritative development repository. Internal machine paths and recovery-only metadata are intentionally omitted here.

## FH6 Session-Level Telemetry Branch V1.0

**Status: FROZEN / VALIDATED**

The verify-only milestone report records **17/17 checks passing**, with six cataloged sessions, four controlled-purpose sessions, ten comparison figures, six controlled figures, full required artifact coverage, and zero missing required outputs.

Validated scope includes real FH6 UDP capture, packet decoding/native logging, native-to-normalized adaptation, robust session-distance recovery, session plotting, metrics, context-filtered event detection, multi-session comparison, dashboard reporting, controlled-session reporting, and milestone validation.

## FH6 Formal-Lap Analysis Pipeline V1.0

**Status: FROZEN / VALIDATED**

The protected real-data formal-lap capture produced three complete laps. Validation records:

- reference: **Lap 03 — 62.584166 s**
- Lap 02: **63.905411 s** — **+1.321244489 s** vs reference
- Lap 01: **64.156156 s** — **+1.571989647 s** vs reference
- normalized comparison grid: **400 points**
- common normalized distance: **5953.017578 m**
- real comparisons: **2**
- comparison figures: **12**
- session-level V1.0 regression: **PASS**

## V1.1 Reference + Sector Foundation

**Status: VALIDATED BASELINE**

Automatic reference selection reproduced Lap 03. Ten equal-distance sectors cover the full normalized lap and reconstruct both comparison deltas exactly relative to the frozen V1.0 baseline.

## V1.1 Corner-Aware Semantic Analysis

**Status: RECOVERED / VALIDATED BASELINE**

The reference lap is partitioned into **13 semantic segments: 6 corners and 7 straights**. Cross-baseline time-delta reconstruction matches both V1.0 and the fixed-sector control to the preserved numerical precision. Four corner-analysis figures are present.

Automated validation establishes partition integrity, timing reconstruction, cross-baseline agreement, output existence, and classification consistency. A preserved reference-detection figure was also reviewed for semantic plausibility during recovery.

## Acquisition implementation facts

Representative source code defines:

- UDP port **53000**
- fixed packet size **324 bytes**
- **88 decoded native fields**

Validated session data includes multiple approximately 60-second / 1,800-row captures; a representative pipeline record reports 1,800 normalized, plotted, and analyzed rows across approximately **60.031 s**, corresponding to roughly 30 samples/s.

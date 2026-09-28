"""
audit_forza_lap_readiness.py

Boundary-aware FH6 formal lap-readiness auditor.

V2 responsibilities:
- Read native FH6 telemetry.
- Resolve lap, timing, distance, race-state, and driver-input fields.
- Build a stable audit timing axis.
- Detect lap boundaries.
- Build complete-lap intervals.
- Classify telemetry discontinuities by location:
    - pre-first-boundary
    - boundary transition
    - inside candidate lap
    - post-final-boundary
- Reject only candidate laps whose own interval contains an integrity failure.
- Keep initialization and teardown discontinuities as contextual evidence.
- Write catalog-wide lap-readiness summaries.

Important V2 correction:
A reset occurring before the first complete lap no longer invalidates every
later complete lap.

When no independent capture timestamp exists, CurrentRaceTime is converted
into a cumulative-positive audit time axis. This prevents one race-time reset
from being counted both as a race-time failure and as an artificial
capture-time failure.

This script remains read-only with respect to native telemetry and the frozen
V1.0 session-level branch.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final, Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

DEFAULT_CATALOG_FILE: Final[Path] = (
    PROJECT_ROOT
    / "data"
    / "forza"
    / "comparisons"
    / "forza_session_catalog.csv"
)

DEFAULT_NATIVE_DIRECTORY: Final[Path] = (
    PROJECT_ROOT
    / "data"
    / "forza"
    / "native"
)

DEFAULT_SUMMARY_FILE: Final[Path] = (
    PROJECT_ROOT
    / "data"
    / "forza"
    / "comparisons"
    / "forza_lap_readiness_summary.csv"
)

DEFAULT_REPORT_FILE: Final[Path] = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "forza_lap_readiness_audit.txt"
)

FIELD_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "capture_time": (
        "capture_elapsed_s",
        "capture_time_s",
        "elapsed_time_s",
        "timestamp_s",
        "packet_timestamp_s",
        "timestamp",
        "capture_timestamp",
        "capture_time",
        "time_s",
        "time",
    ),
    "race_on": (
        "is_race_on",
        "race_on",
        "israceon",
        "IsRaceOn",
    ),
    "lap_number": (
        "lap_number",
        "lapnumber",
        "LapNumber",
        "current_lap_number",
    ),
    "current_lap_time": (
        "current_lap_time_s",
        "current_lap_time",
        "currentlaptime",
        "CurrentLap",
        "current_lap",
    ),
    "current_race_time": (
        "current_race_time_s",
        "current_race_time",
        "currentracetime",
        "CurrentRaceTime",
        "race_time_s",
    ),
    "distance_traveled": (
        "distance_traveled_m",
        "distance_traveled",
        "distancetraveled",
        "DistanceTraveled",
        "distance_m",
    ),
    "speed": (
        "speed_mps",
        "speed",
        "Speed",
        "vehicle_speed_mps",
    ),
    "throttle": (
        "accel",
        "accelerator",
        "throttle",
        "throttle_raw",
        "accel_raw",
        "Accel",
    ),
    "brake": (
        "brake",
        "brake_raw",
        "Brake",
    ),
    "steer": (
        "steer",
        "steering",
        "steer_raw",
        "Steer",
    ),
    "gear": (
        "gear",
        "Gear",
        "current_gear",
    ),
}

REQUIRED_CONTROL_FIELDS: Final[tuple[str, ...]] = (
    "speed",
    "throttle",
    "brake",
    "steer",
)

TRUE_TEXT_VALUES: Final[frozenset[str]] = frozenset(
    {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
)

FALSE_TEXT_VALUES: Final[frozenset[str]] = frozenset(
    {
        "0",
        "false",
        "no",
        "n",
        "off",
    }
)


@dataclass(frozen=True, slots=True)
class AuditConfig:
    """
    Thresholds used by one lap-readiness audit.
    """

    catalog_file: Path
    native_directory: Path
    summary_file: Path
    report_file: Path

    minimum_active_duration_s: float = 60.0
    minimum_field_coverage_pct: float = 90.0
    minimum_race_on_coverage_pct: float = 80.0
    minimum_timestamp_monotonic_pct: float = 99.0

    minimum_complete_laps: int = 2
    minimum_candidate_lap_duration_s: float = 15.0
    minimum_candidate_lap_distance_m: float = 100.0
    minimum_candidate_rows: int = 100

    capture_time_jump_threshold_s: float = 0.05
    race_time_jump_threshold_s: float = 0.50
    distance_jump_threshold_m: float = 10.0


@dataclass(frozen=True, slots=True)
class TimeAxisResult:
    """
    Timing axis selected for audit calculations.
    """

    elapsed_seconds: pd.Series
    source: str
    independent_capture_time: bool


@dataclass(frozen=True, slots=True)
class LapIntervalAudit:
    """
    Boundary-to-boundary integrity result for one possible complete lap.
    """

    interval_index: int
    lap_number: str

    start_position: int
    end_position: int
    start_row: int
    end_row: int

    duration_s: float
    distance_m: float
    row_count: int
    speed_coverage_pct: float

    base_quality_pass: bool

    capture_time_jumps: int
    race_time_jumps: int
    distance_jumps: int

    integrity_pass: bool
    qualified: bool

    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SessionAudit:
    """
    Machine-readable summary for one native FH6 session.
    """

    session_id: str
    short_label: str
    display_name: str
    native_csv: str
    rows: int

    capture_time_column: str
    capture_time_source: str
    capture_time_independent: bool

    race_on_column: str
    lap_number_column: str
    current_lap_time_column: str
    current_race_time_column: str
    distance_column: str
    speed_column: str
    throttle_column: str
    brake_column: str
    steer_column: str
    gear_column: str

    capture_duration_s: float
    active_driving_duration_s: float
    median_sample_period_s: float
    estimated_sample_rate_hz: float

    capture_time_coverage_pct: float
    capture_time_monotonic_pct: float
    capture_time_negative_jumps: int

    race_on_available: bool
    race_on_coverage_pct: float
    driving_row_coverage_pct: float

    lap_number_available: bool
    lap_numbers_observed: str
    lap_number_transitions: int

    current_lap_time_available: bool
    current_lap_time_coverage_pct: float
    current_lap_time_resets: int

    current_race_time_available: bool
    current_race_time_coverage_pct: float
    current_race_time_monotonic_pct: float
    current_race_time_negative_jumps: int

    distance_available: bool
    distance_coverage_pct: float
    distance_monotonic_pct: float
    distance_reset_count: int

    speed_coverage_pct: float
    throttle_coverage_pct: float
    brake_coverage_pct: float
    steer_coverage_pct: float
    gear_coverage_pct: float
    required_control_fields_present: int

    detected_lap_boundaries: int
    possible_complete_lap_intervals: int
    base_qualified_lap_candidates: int
    qualified_complete_lap_candidates: int
    qualified_laps_with_integrity_failures: int

    candidate_lap_durations_s: str
    candidate_lap_distances_m: str
    interval_integrity_summary: str

    pre_boundary_capture_time_jumps: int
    pre_boundary_race_time_jumps: int
    pre_boundary_distance_jumps: int

    boundary_capture_time_jumps: int
    boundary_race_time_jumps: int
    boundary_distance_jumps: int

    in_lap_capture_time_jumps: int
    in_lap_race_time_jumps: int
    in_lap_distance_jumps: int

    post_boundary_capture_time_jumps: int
    post_boundary_race_time_jumps: int
    post_boundary_distance_jumps: int

    formal_lap_ready: bool
    readiness_status: str
    readiness_reason: str


@dataclass(frozen=True, slots=True)
class DetailedSessionAudit:
    """
    Session summary plus individual candidate-lap interval results.
    """

    summary: SessionAudit
    intervals: tuple[LapIntervalAudit, ...]


def clean_text(value: object) -> str:
    """
    Convert a mixed value to stripped text.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass

    return str(value).strip()


def normalize_name(value: object) -> str:
    """
    Normalize field names for alias matching.
    """

    return re.sub(
        pattern=r"[^a-z0-9]+",
        repl="",
        string=clean_text(value).lower(),
    )


def finite_float(value: object) -> float:
    """
    Return a finite float or NaN.
    """

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan

    return parsed if math.isfinite(parsed) else math.nan


def round_or_nan(
    value: float,
    decimals: int = 4,
) -> float:
    """
    Round a finite value while preserving NaN.
    """

    if not math.isfinite(value):
        return math.nan

    return round(value, decimals)


def resolve_column(
    dataframe: pd.DataFrame,
    semantic_field: str,
) -> str:
    """
    Resolve one semantic field against supported aliases.
    """

    normalized_columns = {
        normalize_name(column): str(column)
        for column in dataframe.columns
    }

    for alias in FIELD_ALIASES[semantic_field]:
        resolved = normalized_columns.get(
            normalize_name(alias)
        )

        if resolved:
            return resolved

    return ""


def numeric_series(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Return one column as floating-point numeric data.
    """

    if not column:
        return pd.Series(
            np.nan,
            index=dataframe.index,
            dtype=float,
        )

    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).astype(float)


def boolean_series(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Parse one field into nullable booleans.
    """

    result = pd.Series(
        pd.NA,
        index=dataframe.index,
        dtype="boolean",
    )

    if not column:
        return result

    source = dataframe[column]

    numeric = pd.to_numeric(
        source,
        errors="coerce",
    )

    numeric_mask = numeric.notna()

    result.loc[numeric_mask] = (
        numeric.loc[numeric_mask] != 0
    )

    unresolved_mask = result.isna()

    for index, value in source.loc[unresolved_mask].items():
        normalized = clean_text(value).lower()

        if normalized in TRUE_TEXT_VALUES:
            result.loc[index] = True

        elif normalized in FALSE_TEXT_VALUES:
            result.loc[index] = False

    return result


def coverage_pct(series: pd.Series) -> float:
    """
    Calculate non-null field coverage.
    """

    if len(series) == 0:
        return 0.0

    return float(
        series.notna().mean() * 100.0
    )


def useful_numeric_signal(
    series: pd.Series,
    minimum_range: float,
) -> bool:
    """
    Confirm that a numeric field contains changing usable data.
    """

    valid = series.dropna()

    if len(valid) < 2:
        return False

    signal_range = float(
        valid.max() - valid.min()
    )

    return (
        math.isfinite(signal_range)
        and signal_range >= minimum_range
    )


def scale_numeric_time(
    numeric: pd.Series,
) -> tuple[pd.Series, float]:
    """
    Convert a numeric timing signal to seconds.

    The FH6 timing fields used by this project are already seconds, but the
    scaler keeps the audit robust to millisecond, microsecond, or nanosecond
    logger timestamps.
    """

    valid = numeric.dropna()

    if len(valid) < 2:
        return numeric.astype(float), 1.0

    differences = np.diff(
        valid.to_numpy(dtype=float)
    )

    positive = differences[
        differences > 0
    ]

    scale = 1.0

    if len(positive) > 0:
        median_difference = float(
            np.median(positive)
        )

        if median_difference >= 1_000_000.0:
            scale = 1_000_000_000.0

        elif median_difference >= 1_000.0:
            scale = 1_000_000.0

        elif median_difference > 2.0:
            scale = 1_000.0

    return numeric.astype(float) / scale, scale


def independent_capture_time_axis(
    dataframe: pd.DataFrame,
    capture_time_column: str,
) -> TimeAxisResult | None:
    """
    Resolve an independent logger/capture timestamp when present.
    """

    if not capture_time_column:
        return None

    raw = dataframe[capture_time_column]

    numeric = pd.to_numeric(
        raw,
        errors="coerce",
    ).astype(float)

    if numeric.notna().mean() >= 0.50:
        numeric_seconds, _ = scale_numeric_time(
            numeric
        )

        valid = numeric_seconds.dropna()

        if len(valid) >= 2:
            first_value = float(
                valid.iloc[0]
            )

            elapsed = (
                numeric_seconds - first_value
            )

            return TimeAxisResult(
                elapsed_seconds=elapsed,
                source="capture_time_numeric",
                independent_capture_time=True,
            )

    datetimes = pd.to_datetime(
        raw,
        errors="coerce",
        utc=True,
    )

    if datetimes.notna().mean() >= 0.50:
        first_datetime = datetimes.dropna().iloc[0]

        elapsed = (
            datetimes - first_datetime
        ).dt.total_seconds().astype(float)

        return TimeAxisResult(
            elapsed_seconds=elapsed,
            source="capture_time_datetime",
            independent_capture_time=True,
        )

    return None


def cumulative_positive_time_axis(
    race_time: pd.Series,
) -> pd.Series:
    """
    Convert resetting CurrentRaceTime into a monotonic audit time axis.

    Only positive progress is accumulated. Negative resets and zero-value
    initialization periods contribute zero time rather than making the
    derived timing axis move backward.

    This axis is for capture/interval timing calculations only. The raw race
    timer is still audited independently for reset locations.
    """

    race_seconds, _ = scale_numeric_time(
        race_time
    )

    differences = race_seconds.diff()

    positive_progress = differences.where(
        differences > 0.0,
        0.0,
    )

    positive_progress = positive_progress.fillna(
        0.0
    )

    elapsed = positive_progress.cumsum()

    return elapsed.astype(float)


def build_audit_time_axis(
    dataframe: pd.DataFrame,
    capture_time_column: str,
    current_race_time_column: str,
) -> TimeAxisResult:
    """
    Build the best timing axis for interval calculations.

    Priority:
    1. Independent capture/logger timestamp.
    2. Cumulative-positive CurrentRaceTime.
    3. Unavailable.

    The fallback deliberately does not masquerade as an independent capture
    clock, preventing duplicate failure reporting.
    """

    independent = independent_capture_time_axis(
        dataframe=dataframe,
        capture_time_column=capture_time_column,
    )

    if independent is not None:
        return independent

    if current_race_time_column:
        race_time = numeric_series(
            dataframe,
            current_race_time_column,
        )

        if (
            coverage_pct(race_time) >= 50.0
            and useful_numeric_signal(
                race_time,
                minimum_range=1.0,
            )
        ):
            elapsed = cumulative_positive_time_axis(
                race_time
            )

            return TimeAxisResult(
                elapsed_seconds=elapsed,
                source="current_race_time_cumulative_positive",
                independent_capture_time=False,
            )

    return TimeAxisResult(
        elapsed_seconds=pd.Series(
            np.nan,
            index=dataframe.index,
            dtype=float,
        ),
        source="unavailable",
        independent_capture_time=False,
    )


def monotonic_pct(
    series: pd.Series,
    tolerance: float = 1e-9,
) -> float:
    """
    Calculate non-decreasing consecutive-difference coverage.
    """

    valid = series.dropna().to_numpy(
        dtype=float
    )

    if len(valid) < 2:
        return 0.0

    differences = np.diff(valid)

    return float(
        np.mean(
            differences >= -abs(tolerance)
        )
        * 100.0
    )


def negative_jump_positions(
    series: pd.Series,
    threshold: float,
) -> list[int]:
    """
    Return zero-based positions at which a negative jump arrives.

    Example:
    position 528 means the transition 527 -> 528 caused the jump.
    """

    values = series.to_numpy(
        dtype=float
    )

    positions: list[int] = []

    previous_value: float | None = None

    for position, value in enumerate(values):
        if not math.isfinite(value):
            continue

        if (
            previous_value is not None
            and value
            < previous_value - abs(threshold)
        ):
            positions.append(position)

        previous_value = value

    return positions


def median_positive_step(
    series: pd.Series,
) -> float:
    """
    Return the median positive sample interval.
    """

    valid = series.dropna().to_numpy(
        dtype=float
    )

    if len(valid) < 2:
        return math.nan

    differences = np.diff(valid)

    positive = differences[
        differences > 0
    ]

    if len(positive) == 0:
        return math.nan

    return float(
        np.median(positive)
    )


def capture_duration(
    elapsed_seconds: pd.Series,
) -> float:
    """
    Return duration represented by a monotonic audit timing axis.
    """

    valid = elapsed_seconds.dropna()

    if len(valid) < 2:
        return math.nan

    duration = float(
        valid.iloc[-1] - valid.iloc[0]
    )

    return (
        duration
        if duration >= 0
        else math.nan
    )


def active_duration(
    elapsed_seconds: pd.Series,
    active_mask: pd.Series,
) -> float:
    """
    Sum positive audit-time intervals whose destination row is active.

    Because the audit axis is monotonic, active duration cannot exceed total
    represented capture duration.
    """

    times = elapsed_seconds.to_numpy(
        dtype=float
    )

    active = (
        active_mask.fillna(False)
        .to_numpy(dtype=bool)
    )

    if len(times) < 2:
        return math.nan

    differences = np.diff(times)

    valid_intervals = (
        np.isfinite(differences)
        & (differences > 0.0)
        & active[1:]
    )

    if not np.any(valid_intervals):
        return 0.0

    return float(
        np.sum(
            differences[valid_intervals]
        )
    )


def lap_number_boundaries(
    lap_number: pd.Series,
) -> tuple[list[int], list[int]]:
    """
    Detect rows where the lap number changes.
    """

    boundaries: list[int] = []
    observed: list[int] = []

    previous_value: int | None = None

    for position, raw_value in enumerate(
        lap_number.to_numpy(dtype=float)
    ):
        if not math.isfinite(raw_value):
            continue

        value = int(
            round(raw_value)
        )

        if value not in observed:
            observed.append(value)

        if (
            previous_value is not None
            and value != previous_value
        ):
            boundaries.append(position)

        previous_value = value

    return boundaries, observed


def reset_boundaries(
    series: pd.Series,
    threshold: float,
) -> list[int]:
    """
    Detect downward resets in a field such as CurrentLap.
    """

    return negative_jump_positions(
        series=series,
        threshold=threshold,
    )


def deduplicate_boundaries(
    boundaries: Iterable[int],
    tolerance_rows: int = 5,
) -> list[int]:
    """
    Merge independent boundary detections occurring within a few rows.
    """

    ordered = sorted(
        set(boundaries)
    )

    if not ordered:
        return []

    deduplicated = [
        ordered[0]
    ]

    for boundary in ordered[1:]:
        previous = deduplicated[-1]

        if boundary - previous <= tolerance_rows:
            deduplicated[-1] = int(
                round(
                    (
                        previous
                        + boundary
                    )
                    / 2.0
                )
            )

        else:
            deduplicated.append(
                boundary
            )

    return deduplicated


def interval_distance(
    distance: pd.Series,
    start_position: int,
    end_position: int,
) -> float:
    """
    Calculate distance span inside one candidate interval.
    """

    segment = distance.iloc[
        start_position:end_position + 1
    ].dropna()

    if len(segment) < 2:
        return math.nan

    return float(
        segment.max()
        - segment.min()
    )


def lap_number_at_position(
    lap_number: pd.Series,
    position: int,
) -> str:
    """
    Return the lap number at one boundary position.
    """

    if (
        position < 0
        or position >= len(lap_number)
    ):
        return ""

    value = finite_float(
        lap_number.iloc[position]
    )

    if not math.isfinite(value):
        return ""

    return str(
        int(round(value))
    )


def build_base_intervals(
    *,
    boundaries: Sequence[int],
    elapsed_seconds: pd.Series,
    distance: pd.Series,
    speed: pd.Series,
    lap_number: pd.Series,
    config: AuditConfig,
) -> tuple[LapIntervalAudit, ...]:
    """
    Build candidate complete-lap intervals before integrity classification.
    """

    intervals: list[LapIntervalAudit] = []

    for interval_index, (
        start_position,
        end_position,
    ) in enumerate(
        zip(
            boundaries[:-1],
            boundaries[1:],
        ),
        start=1,
    ):
        start_time = finite_float(
            elapsed_seconds.iloc[start_position]
        )

        end_time = finite_float(
            elapsed_seconds.iloc[end_position]
        )

        duration_s = (
            end_time - start_time
            if (
                math.isfinite(start_time)
                and math.isfinite(end_time)
            )
            else math.nan
        )

        row_count = (
            end_position
            - start_position
            + 1
        )

        distance_m = interval_distance(
            distance=distance,
            start_position=start_position,
            end_position=end_position,
        )

        speed_coverage = coverage_pct(
            speed.iloc[
                start_position:end_position + 1
            ]
        )

        rejection_reasons: list[str] = []

        if (
            not math.isfinite(duration_s)
            or duration_s
            < config.minimum_candidate_lap_duration_s
        ):
            rejection_reasons.append(
                "lap duration below minimum"
            )

        if row_count < config.minimum_candidate_rows:
            rejection_reasons.append(
                "telemetry row count below minimum"
            )

        if (
            not math.isfinite(distance_m)
            or distance_m
            < config.minimum_candidate_lap_distance_m
        ):
            rejection_reasons.append(
                "lap distance below minimum"
            )

        if (
            speed_coverage
            < config.minimum_field_coverage_pct
        ):
            rejection_reasons.append(
                "speed coverage below minimum"
            )

        base_quality_pass = (
            not rejection_reasons
        )

        intervals.append(
            LapIntervalAudit(
                interval_index=interval_index,
                lap_number=lap_number_at_position(
                    lap_number,
                    start_position,
                ),
                start_position=start_position,
                end_position=end_position,
                start_row=start_position + 1,
                end_row=end_position + 1,
                duration_s=duration_s,
                distance_m=distance_m,
                row_count=row_count,
                speed_coverage_pct=speed_coverage,
                base_quality_pass=base_quality_pass,
                capture_time_jumps=0,
                race_time_jumps=0,
                distance_jumps=0,
                integrity_pass=True,
                qualified=base_quality_pass,
                rejection_reasons=tuple(
                    rejection_reasons
                ),
            )
        )

    return tuple(intervals)


def jump_region(
    position: int,
    boundaries: Sequence[int],
    intervals: Sequence[LapIntervalAudit],
) -> str:
    """
    Classify one discontinuity relative to lap boundaries.

    A jump exactly on a boundary is treated as a boundary-transition event.
    Only jumps strictly inside an interval count as in-lap integrity faults.
    """

    if not boundaries:
        return "outside"

    first_boundary = boundaries[0]
    final_boundary = boundaries[-1]

    if position < first_boundary:
        return "pre"

    if position > final_boundary:
        return "post"

    if position in boundaries:
        return "boundary"

    for interval in intervals:
        if (
            interval.start_position
            < position
            < interval.end_position
        ):
            return "in_lap"

    return "boundary"


def count_jump_regions(
    positions: Sequence[int],
    boundaries: Sequence[int],
    intervals: Sequence[LapIntervalAudit],
) -> dict[str, int]:
    """
    Count discontinuities by boundary-aware region.
    """

    counts = {
        "pre": 0,
        "boundary": 0,
        "in_lap": 0,
        "post": 0,
        "outside": 0,
    }

    for position in positions:
        region = jump_region(
            position=position,
            boundaries=boundaries,
            intervals=intervals,
        )

        counts[region] += 1

    return counts


def jumps_inside_interval(
    positions: Sequence[int],
    interval: LapIntervalAudit,
) -> int:
    """
    Count jumps strictly inside one candidate lap.
    """

    return sum(
        interval.start_position
        < position
        < interval.end_position
        for position in positions
    )


def apply_interval_integrity(
    *,
    intervals: Sequence[LapIntervalAudit],
    capture_time_jump_positions: Sequence[int],
    race_time_jump_positions: Sequence[int],
    distance_jump_positions: Sequence[int],
) -> tuple[LapIntervalAudit, ...]:
    """
    Add interval-local discontinuity results.

    A candidate lap fails integrity only when a discontinuity is located
    strictly inside that interval.
    """

    updated: list[LapIntervalAudit] = []

    for interval in intervals:
        capture_jumps = jumps_inside_interval(
            capture_time_jump_positions,
            interval,
        )

        race_jumps = jumps_inside_interval(
            race_time_jump_positions,
            interval,
        )

        distance_jumps = jumps_inside_interval(
            distance_jump_positions,
            interval,
        )

        integrity_reasons: list[str] = []

        if capture_jumps > 0:
            integrity_reasons.append(
                "capture-time discontinuity inside lap"
            )

        if race_jumps > 0:
            integrity_reasons.append(
                "race-time discontinuity inside lap"
            )

        if distance_jumps > 0:
            integrity_reasons.append(
                "distance discontinuity inside lap"
            )

        integrity_pass = (
            not integrity_reasons
        )

        qualified = (
            interval.base_quality_pass
            and integrity_pass
        )

        rejection_reasons = (
            *interval.rejection_reasons,
            *integrity_reasons,
        )

        updated.append(
            replace(
                interval,
                capture_time_jumps=capture_jumps,
                race_time_jumps=race_jumps,
                distance_jumps=distance_jumps,
                integrity_pass=integrity_pass,
                qualified=qualified,
                rejection_reasons=tuple(
                    rejection_reasons
                ),
            )
        )

    return tuple(updated)


def join_numbers(
    values: Sequence[float],
    decimals: int,
) -> str:
    """
    Join numeric values into one pipe-delimited CSV cell.
    """

    return " | ".join(
        f"{value:.{decimals}f}"
        for value in values
        if math.isfinite(value)
    )


def interval_integrity_summary(
    intervals: Sequence[LapIntervalAudit],
) -> str:
    """
    Build a compact lap-by-lap status string.
    """

    parts: list[str] = []

    for interval in intervals:
        label = (
            f"Lap {interval.lap_number}"
            if interval.lap_number
            else f"Interval {interval.interval_index}"
        )

        if interval.qualified:
            status = "PASS"

        elif (
            interval.base_quality_pass
            and not interval.integrity_pass
        ):
            status = "INTEGRITY_FAIL"

        else:
            status = "QUALITY_FAIL"

        parts.append(
            f"{label}: {status}"
        )

    return " | ".join(parts)


def catalog_value(
    row: pd.Series,
    candidates: Sequence[str],
) -> str:
    """
    Read the first non-empty matching catalog value.
    """

    normalized_columns = {
        normalize_name(column): column
        for column in row.index
    }

    for candidate in candidates:
        column = normalized_columns.get(
            normalize_name(candidate)
        )

        if column is None:
            continue

        value = clean_text(
            row[column]
        )

        if value:
            return value

    return ""


def resolve_native_csv(
    catalog_row: pd.Series,
    session_id: str,
    native_directory: Path,
) -> Path:
    """
    Resolve one native CSV.
    """

    catalog_path_text = catalog_value(
        catalog_row,
        (
            "native_csv",
            "native_csv_path",
            "native_file",
            "native_path",
            "source_csv",
        ),
    )

    if catalog_path_text:
        catalog_path = Path(
            catalog_path_text
        )

        if not catalog_path.is_absolute():
            catalog_path = (
                PROJECT_ROOT
                / catalog_path
            )

        if catalog_path.exists():
            return catalog_path.resolve()

    return (
        native_directory
        / f"{session_id}.csv"
    ).resolve()


def build_readiness_reasons(
    *,
    time_axis: TimeAxisResult,
    capture_duration_s: float,
    active_duration_s: float,
    capture_time_monotonic_pct: float,
    race_on_available: bool,
    race_on_coverage_pct: float,
    current_race_time_available: bool,
    current_race_time_coverage_pct: float,
    distance_available: bool,
    distance_coverage_pct: float,
    required_control_fields_present: int,
    qualified_complete_laps: int,
    config: AuditConfig,
) -> list[str]:
    """
    Build formal-lap readiness failures.

    Global pre-race or post-race resets are intentionally not included here.
    Interval-local faults have already removed affected laps from the
    qualified-lap count.
    """

    reasons: list[str] = []

    if time_axis.source == "unavailable":
        reasons.append(
            "no usable timing axis"
        )

    if (
        not math.isfinite(capture_duration_s)
        or capture_duration_s
        < config.minimum_active_duration_s
    ):
        reasons.append(
            "represented capture duration below minimum"
        )

    if (
        not math.isfinite(active_duration_s)
        or active_duration_s
        < config.minimum_active_duration_s
    ):
        reasons.append(
            "active-driving duration below minimum"
        )

    if (
        capture_time_monotonic_pct
        < config.minimum_timestamp_monotonic_pct
    ):
        reasons.append(
            "audit timing axis is not sufficiently monotonic"
        )

    if not race_on_available:
        reasons.append(
            "race-on signal unavailable"
        )

    elif (
        race_on_coverage_pct
        < config.minimum_race_on_coverage_pct
    ):
        reasons.append(
            "race-on coverage below threshold"
        )

    if not current_race_time_available:
        reasons.append(
            "current race time unavailable or static"
        )

    elif (
        current_race_time_coverage_pct
        < config.minimum_field_coverage_pct
    ):
        reasons.append(
            "current race-time coverage below threshold"
        )

    if not distance_available:
        reasons.append(
            "distance-traveled signal unavailable or static"
        )

    elif (
        distance_coverage_pct
        < config.minimum_field_coverage_pct
    ):
        reasons.append(
            "distance-traveled coverage below threshold"
        )

    if (
        required_control_fields_present
        < len(REQUIRED_CONTROL_FIELDS)
    ):
        reasons.append(
            "one or more required control fields are missing"
        )

    if (
        qualified_complete_laps
        < config.minimum_complete_laps
    ):
        reasons.append(
            "fewer than the required number of clean complete laps"
        )

    return reasons


def audit_one_session_detailed(
    catalog_row: pd.Series,
    config: AuditConfig,
) -> DetailedSessionAudit:
    """
    Audit one native session and retain interval-level evidence.
    """

    session_id = catalog_value(
        catalog_row,
        ("session_id",),
    )

    if not session_id:
        raise ValueError(
            "Catalog row is missing session_id."
        )

    short_label = catalog_value(
        catalog_row,
        (
            "short_label",
            "label",
            "session_label",
        ),
    )

    display_name = catalog_value(
        catalog_row,
        (
            "display_name",
            "session_name",
            "name",
        ),
    )

    native_csv = resolve_native_csv(
        catalog_row=catalog_row,
        session_id=session_id,
        native_directory=config.native_directory,
    )

    if not native_csv.exists():
        raise FileNotFoundError(
            f"Native CSV does not exist for {session_id}: "
            f"{native_csv}"
        )

    dataframe = pd.read_csv(
        native_csv
    )

    if dataframe.empty:
        raise ValueError(
            f"Native CSV is empty for {session_id}: "
            f"{native_csv}"
        )

    columns = {
        semantic_field: resolve_column(
            dataframe,
            semantic_field,
        )
        for semantic_field in FIELD_ALIASES
    }

    race_on = boolean_series(
        dataframe,
        columns["race_on"],
    )

    lap_number = numeric_series(
        dataframe,
        columns["lap_number"],
    )

    current_lap_time = numeric_series(
        dataframe,
        columns["current_lap_time"],
    )

    current_race_time = numeric_series(
        dataframe,
        columns["current_race_time"],
    )

    distance = numeric_series(
        dataframe,
        columns["distance_traveled"],
    )

    speed = numeric_series(
        dataframe,
        columns["speed"],
    )

    throttle = numeric_series(
        dataframe,
        columns["throttle"],
    )

    brake = numeric_series(
        dataframe,
        columns["brake"],
    )

    steer = numeric_series(
        dataframe,
        columns["steer"],
    )

    gear = numeric_series(
        dataframe,
        columns["gear"],
    )

    time_axis = build_audit_time_axis(
        dataframe=dataframe,
        capture_time_column=columns["capture_time"],
        current_race_time_column=columns[
            "current_race_time"
        ],
    )

    elapsed_seconds = (
        time_axis.elapsed_seconds
    )

    race_on_available = (
        bool(columns["race_on"])
        and race_on.notna().any()
    )

    race_on_coverage = (
        float(
            race_on.fillna(False).mean()
            * 100.0
        )
        if race_on_available
        else 0.0
    )

    driving_mask = (
        speed.fillna(0.0) > 1.0
    )

    driving_row_coverage = float(
        driving_mask.mean() * 100.0
    )

    active_mask = (
        race_on.fillna(False)
        if race_on_available
        else driving_mask
    )

    lap_boundaries, observed_lap_numbers = (
        lap_number_boundaries(
            lap_number
        )
    )

    current_lap_resets = reset_boundaries(
        current_lap_time,
        threshold=1.0,
    )

    combined_boundaries = deduplicate_boundaries(
        [
            *lap_boundaries,
            *current_lap_resets,
        ]
    )

    base_intervals = build_base_intervals(
        boundaries=combined_boundaries,
        elapsed_seconds=elapsed_seconds,
        distance=distance,
        speed=speed,
        lap_number=lap_number,
        config=config,
    )

    # An independent capture clock may legitimately contain its own jumps.
    # A derived CurrentRaceTime audit axis is monotonic by construction, so
    # raw CurrentRaceTime resets are audited only in the race-time channel.
    capture_time_jump_positions = (
        negative_jump_positions(
            elapsed_seconds,
            threshold=(
                config.capture_time_jump_threshold_s
            ),
        )
        if time_axis.independent_capture_time
        else []
    )

    race_time_jump_positions = (
        negative_jump_positions(
            current_race_time,
            threshold=(
                config.race_time_jump_threshold_s
            ),
        )
    )

    distance_jump_positions = (
        negative_jump_positions(
            distance,
            threshold=(
                config.distance_jump_threshold_m
            ),
        )
    )

    intervals = apply_interval_integrity(
        intervals=base_intervals,
        capture_time_jump_positions=(
            capture_time_jump_positions
        ),
        race_time_jump_positions=(
            race_time_jump_positions
        ),
        distance_jump_positions=(
            distance_jump_positions
        ),
    )

    capture_regions = count_jump_regions(
        positions=capture_time_jump_positions,
        boundaries=combined_boundaries,
        intervals=intervals,
    )

    race_regions = count_jump_regions(
        positions=race_time_jump_positions,
        boundaries=combined_boundaries,
        intervals=intervals,
    )

    distance_regions = count_jump_regions(
        positions=distance_jump_positions,
        boundaries=combined_boundaries,
        intervals=intervals,
    )

    qualified_intervals = [
        interval
        for interval in intervals
        if interval.qualified
    ]

    base_qualified_intervals = [
        interval
        for interval in intervals
        if interval.base_quality_pass
    ]

    integrity_failed_intervals = [
        interval
        for interval in intervals
        if (
            interval.base_quality_pass
            and not interval.integrity_pass
        )
    ]

    candidate_durations = [
        interval.duration_s
        for interval in qualified_intervals
    ]

    candidate_distances = [
        interval.distance_m
        for interval in qualified_intervals
    ]

    capture_duration_s = capture_duration(
        elapsed_seconds
    )

    active_duration_s = active_duration(
        elapsed_seconds=elapsed_seconds,
        active_mask=active_mask,
    )

    median_period_s = median_positive_step(
        elapsed_seconds
    )

    estimated_sample_rate_hz = (
        1.0 / median_period_s
        if (
            math.isfinite(median_period_s)
            and median_period_s > 0
        )
        else math.nan
    )

    capture_time_coverage = coverage_pct(
        elapsed_seconds
    )

    capture_time_monotonic = monotonic_pct(
        elapsed_seconds
    )

    current_lap_time_available = (
        useful_numeric_signal(
            current_lap_time,
            minimum_range=1.0,
        )
    )

    current_race_time_available = (
        useful_numeric_signal(
            current_race_time,
            minimum_range=1.0,
        )
    )

    distance_available = useful_numeric_signal(
        distance,
        minimum_range=10.0,
    )

    control_coverages = {
        "speed": coverage_pct(speed),
        "throttle": coverage_pct(throttle),
        "brake": coverage_pct(brake),
        "steer": coverage_pct(steer),
    }

    required_control_fields_present = sum(
        bool(columns[field])
        and control_coverages[field]
        >= config.minimum_field_coverage_pct
        for field in REQUIRED_CONTROL_FIELDS
    )

    readiness_reasons = build_readiness_reasons(
        time_axis=time_axis,
        capture_duration_s=capture_duration_s,
        active_duration_s=active_duration_s,
        capture_time_monotonic_pct=(
            capture_time_monotonic
        ),
        race_on_available=race_on_available,
        race_on_coverage_pct=race_on_coverage,
        current_race_time_available=(
            current_race_time_available
        ),
        current_race_time_coverage_pct=(
            coverage_pct(current_race_time)
        ),
        distance_available=distance_available,
        distance_coverage_pct=coverage_pct(
            distance
        ),
        required_control_fields_present=(
            required_control_fields_present
        ),
        qualified_complete_laps=len(
            qualified_intervals
        ),
        config=config,
    )

    formal_lap_ready = (
        not readiness_reasons
    )

    if formal_lap_ready:
        readiness_status = (
            "formal_lap_ready"
        )

    elif (
        len(combined_boundaries) >= 2
        or len(qualified_intervals) >= 1
    ):
        readiness_status = (
            "partial_lap_evidence"
        )

    elif (
        required_control_fields_present
        == len(REQUIRED_CONTROL_FIELDS)
        and time_axis.source != "unavailable"
    ):
        readiness_status = (
            "session_level_only"
        )

    else:
        readiness_status = (
            "insufficient_telemetry"
        )

    readiness_reason = (
        "All formal lap-readiness gates passed."
        if formal_lap_ready
        else "; ".join(readiness_reasons)
    )

    summary = SessionAudit(
        session_id=session_id,
        short_label=short_label,
        display_name=display_name,
        native_csv=str(native_csv),
        rows=len(dataframe),

        capture_time_column=columns[
            "capture_time"
        ],
        capture_time_source=time_axis.source,
        capture_time_independent=(
            time_axis.independent_capture_time
        ),

        race_on_column=columns["race_on"],
        lap_number_column=columns[
            "lap_number"
        ],
        current_lap_time_column=columns[
            "current_lap_time"
        ],
        current_race_time_column=columns[
            "current_race_time"
        ],
        distance_column=columns[
            "distance_traveled"
        ],
        speed_column=columns["speed"],
        throttle_column=columns[
            "throttle"
        ],
        brake_column=columns["brake"],
        steer_column=columns["steer"],
        gear_column=columns["gear"],

        capture_duration_s=round_or_nan(
            capture_duration_s
        ),
        active_driving_duration_s=(
            round_or_nan(active_duration_s)
        ),
        median_sample_period_s=(
            round_or_nan(
                median_period_s,
                decimals=6,
            )
        ),
        estimated_sample_rate_hz=(
            round_or_nan(
                estimated_sample_rate_hz
            )
        ),

        capture_time_coverage_pct=(
            round_or_nan(
                capture_time_coverage
            )
        ),
        capture_time_monotonic_pct=(
            round_or_nan(
                capture_time_monotonic
            )
        ),
        capture_time_negative_jumps=len(
            capture_time_jump_positions
        ),

        race_on_available=race_on_available,
        race_on_coverage_pct=round_or_nan(
            race_on_coverage
        ),
        driving_row_coverage_pct=(
            round_or_nan(
                driving_row_coverage
            )
        ),

        lap_number_available=(
            bool(columns["lap_number"])
            and lap_number.notna().any()
        ),
        lap_numbers_observed=" | ".join(
            str(value)
            for value in observed_lap_numbers
        ),
        lap_number_transitions=len(
            lap_boundaries
        ),

        current_lap_time_available=(
            current_lap_time_available
        ),
        current_lap_time_coverage_pct=(
            round_or_nan(
                coverage_pct(
                    current_lap_time
                )
            )
        ),
        current_lap_time_resets=len(
            current_lap_resets
        ),

        current_race_time_available=(
            current_race_time_available
        ),
        current_race_time_coverage_pct=(
            round_or_nan(
                coverage_pct(
                    current_race_time
                )
            )
        ),
        current_race_time_monotonic_pct=(
            round_or_nan(
                monotonic_pct(
                    current_race_time
                )
            )
        ),
        current_race_time_negative_jumps=len(
            race_time_jump_positions
        ),

        distance_available=distance_available,
        distance_coverage_pct=(
            round_or_nan(
                coverage_pct(distance)
            )
        ),
        distance_monotonic_pct=(
            round_or_nan(
                monotonic_pct(distance)
            )
        ),
        distance_reset_count=len(
            distance_jump_positions
        ),

        speed_coverage_pct=round_or_nan(
            coverage_pct(speed)
        ),
        throttle_coverage_pct=round_or_nan(
            coverage_pct(throttle)
        ),
        brake_coverage_pct=round_or_nan(
            coverage_pct(brake)
        ),
        steer_coverage_pct=round_or_nan(
            coverage_pct(steer)
        ),
        gear_coverage_pct=round_or_nan(
            coverage_pct(gear)
        ),
        required_control_fields_present=(
            required_control_fields_present
        ),

        detected_lap_boundaries=len(
            combined_boundaries
        ),
        possible_complete_lap_intervals=len(
            intervals
        ),
        base_qualified_lap_candidates=len(
            base_qualified_intervals
        ),
        qualified_complete_lap_candidates=len(
            qualified_intervals
        ),
        qualified_laps_with_integrity_failures=len(
            integrity_failed_intervals
        ),

        candidate_lap_durations_s=(
            join_numbers(
                candidate_durations,
                decimals=3,
            )
        ),
        candidate_lap_distances_m=(
            join_numbers(
                candidate_distances,
                decimals=2,
            )
        ),
        interval_integrity_summary=(
            interval_integrity_summary(
                intervals
            )
        ),

        pre_boundary_capture_time_jumps=(
            capture_regions["pre"]
        ),
        pre_boundary_race_time_jumps=(
            race_regions["pre"]
        ),
        pre_boundary_distance_jumps=(
            distance_regions["pre"]
        ),

        boundary_capture_time_jumps=(
            capture_regions["boundary"]
        ),
        boundary_race_time_jumps=(
            race_regions["boundary"]
        ),
        boundary_distance_jumps=(
            distance_regions["boundary"]
        ),

        in_lap_capture_time_jumps=(
            capture_regions["in_lap"]
        ),
        in_lap_race_time_jumps=(
            race_regions["in_lap"]
        ),
        in_lap_distance_jumps=(
            distance_regions["in_lap"]
        ),

        post_boundary_capture_time_jumps=(
            capture_regions["post"]
        ),
        post_boundary_race_time_jumps=(
            race_regions["post"]
        ),
        post_boundary_distance_jumps=(
            distance_regions["post"]
        ),

        formal_lap_ready=formal_lap_ready,
        readiness_status=readiness_status,
        readiness_reason=readiness_reason,
    )

    return DetailedSessionAudit(
        summary=summary,
        intervals=intervals,
    )


def audit_one_session(
    catalog_row: pd.Series,
    config: AuditConfig,
) -> SessionAudit:
    """
    Backward-compatible summary-only audit entry point.
    """

    return audit_one_session_detailed(
        catalog_row=catalog_row,
        config=config,
    ).summary


def read_catalog(
    catalog_file: Path,
) -> pd.DataFrame:
    """
    Read and validate the authoritative session catalog.
    """

    if not catalog_file.exists():
        raise FileNotFoundError(
            f"Session catalog does not exist: "
            f"{catalog_file}"
        )

    dataframe = pd.read_csv(
        catalog_file
    )

    if dataframe.empty:
        raise ValueError(
            f"Session catalog is empty: "
            f"{catalog_file}"
        )

    normalized_columns = {
        normalize_name(column)
        for column in dataframe.columns
    }

    if (
        normalize_name("session_id")
        not in normalized_columns
    ):
        raise ValueError(
            "Session catalog is missing required "
            "column: session_id"
        )

    return dataframe


def run_audit(
    config: AuditConfig,
) -> tuple[SessionAudit, ...]:
    """
    Audit all cataloged V1.0 sessions.
    """

    catalog = read_catalog(
        config.catalog_file
    )

    return tuple(
        audit_one_session(
            catalog_row=row,
            config=config,
        )
        for _, row in catalog.iterrows()
    )


def write_summary_csv(
    audits: Sequence[SessionAudit],
    output_file: Path,
) -> None:
    """
    Write the machine-readable summary.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(
        asdict(audit)
        for audit in audits
    )

    dataframe.to_csv(
        output_file,
        index=False,
    )


def format_float(
    value: float,
    decimals: int = 3,
    suffix: str = "",
) -> str:
    """
    Format one finite value for a text report.
    """

    if not math.isfinite(value):
        return "n/a"

    return (
        f"{value:.{decimals}f}"
        f"{suffix}"
    )


def build_report(
    audits: Sequence[SessionAudit],
    config: AuditConfig,
) -> str:
    """
    Build the catalog-wide lap-readiness report.
    """

    formal_ready = [
        audit
        for audit in audits
        if audit.readiness_status
        == "formal_lap_ready"
    ]

    partial = [
        audit
        for audit in audits
        if audit.readiness_status
        == "partial_lap_evidence"
    ]

    session_only = [
        audit
        for audit in audits
        if audit.readiness_status
        == "session_level_only"
    ]

    insufficient = [
        audit
        for audit in audits
        if audit.readiness_status
        == "insufficient_telemetry"
    ]

    lines: list[str] = [
        "FH6 FORMAL LAP-READINESS AUDIT — V2",
        "=" * 72,
        "",
        "PURPOSE",
        "-" * 72,
        "Evaluate native FH6 telemetry for complete-lap analysis while",
        "classifying resets relative to validated lap boundaries.",
        "",
        "V2 INTEGRITY POLICY",
        "-" * 72,
        "Pre-first-boundary resets: contextual warning only.",
        "Boundary-transition resets: contextual warning unless they affect",
        "the actual interior of a complete-lap interval.",
        "Inside-lap resets: invalidate that candidate lap.",
        "Post-final-boundary resets: contextual warning only.",
        "",
        "SUMMARY",
        "-" * 72,
        f"Catalog sessions audited: {len(audits)}",
        f"Formal lap-ready sessions: {len(formal_ready)}",
        f"Partial lap-evidence sessions: {len(partial)}",
        f"Session-level-only sessions: {len(session_only)}",
        f"Insufficient-telemetry sessions: {len(insufficient)}",
        "",
        "SESSION RESULTS",
        "-" * 72,
    ]

    for audit in audits:
        label = (
            audit.short_label
            or audit.session_id
        )

        lines.extend(
            [
                "",
                label,
                f"  Session ID: {audit.session_id}",
                (
                    "  Readiness status: "
                    f"{audit.readiness_status}"
                ),
                (
                    "  Formal lap ready: "
                    f"{audit.formal_lap_ready}"
                ),
                (
                    "  Timing source: "
                    f"{audit.capture_time_source}"
                ),
                (
                    "  Capture timing independent: "
                    f"{audit.capture_time_independent}"
                ),
                (
                    "  Represented capture duration: "
                    f"{format_float(audit.capture_duration_s, 3, ' s')}"
                ),
                (
                    "  Active-driving duration: "
                    f"{format_float(audit.active_driving_duration_s, 3, ' s')}"
                ),
                (
                    "  Lap boundaries: "
                    f"{audit.detected_lap_boundaries}"
                ),
                (
                    "  Clean qualified complete laps: "
                    f"{audit.qualified_complete_lap_candidates}"
                ),
                (
                    "  Otherwise-qualified laps with integrity failures: "
                    f"{audit.qualified_laps_with_integrity_failures}"
                ),
                (
                    "  Interval status: "
                    f"{audit.interval_integrity_summary or 'none'}"
                ),
                (
                    "  Pre-boundary race-time jumps: "
                    f"{audit.pre_boundary_race_time_jumps}"
                ),
                (
                    "  Pre-boundary distance jumps: "
                    f"{audit.pre_boundary_distance_jumps}"
                ),
                (
                    "  In-lap race-time jumps: "
                    f"{audit.in_lap_race_time_jumps}"
                ),
                (
                    "  In-lap distance jumps: "
                    f"{audit.in_lap_distance_jumps}"
                ),
                (
                    "  Reason: "
                    f"{audit.readiness_reason}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "OUTPUTS",
            "-" * 72,
            f"Summary CSV: {config.summary_file}",
            f"Audit report: {config.report_file}",
            "",
            "Report generated automatically by",
            "audit_forza_lap_readiness.py.",
        ]
    )

    return "\n".join(lines)


def write_text_file(
    text: str,
    output_file: Path,
) -> None:
    """
    Write one UTF-8 text report.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
        text,
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Audit cataloged native FH6 telemetry using "
            "boundary-aware formal-lap integrity logic."
        )
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG_FILE,
    )

    parser.add_argument(
        "--native-dir",
        type=Path,
        default=DEFAULT_NATIVE_DIRECTORY,
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_FILE,
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_FILE,
    )

    return parser.parse_args()


def print_terminal_summary(
    audits: Sequence[SessionAudit],
    config: AuditConfig,
) -> None:
    """
    Print a concise audit summary.
    """

    formal_ready = sum(
        audit.readiness_status
        == "formal_lap_ready"
        for audit in audits
    )

    partial = sum(
        audit.readiness_status
        == "partial_lap_evidence"
        for audit in audits
    )

    session_only = sum(
        audit.readiness_status
        == "session_level_only"
        for audit in audits
    )

    insufficient = sum(
        audit.readiness_status
        == "insufficient_telemetry"
        for audit in audits
    )

    print(
        "\nFH6 boundary-aware lap-readiness "
        "audit complete."
    )

    print(
        f"\nCatalog sessions audited: "
        f"{len(audits)}"
    )

    print(
        f"Formal lap-ready sessions: "
        f"{formal_ready}"
    )

    print(
        f"Partial lap-evidence sessions: "
        f"{partial}"
    )

    print(
        f"Session-level-only sessions: "
        f"{session_only}"
    )

    print(
        f"Insufficient-telemetry sessions: "
        f"{insufficient}"
    )

    print(
        f"Summary rows written: "
        f"{len(audits)}"
    )

    print(
        f"\nSummary CSV:\n"
        f"{config.summary_file}"
    )

    print(
        f"\nAudit report:\n"
        f"{config.report_file}"
    )


def main() -> None:
    """
    Command-line entry point.
    """

    args = parse_arguments()

    config = AuditConfig(
        catalog_file=args.catalog,
        native_directory=args.native_dir,
        summary_file=args.summary,
        report_file=args.report,
    )

    print("=" * 72)
    print("FH6 Boundary-Aware Formal Lap-Readiness Audit V2")
    print("=" * 72)

    audits = run_audit(
        config
    )

    write_summary_csv(
        audits=audits,
        output_file=config.summary_file,
    )

    report_text = build_report(
        audits=audits,
        config=config,
    )

    write_text_file(
        text=report_text,
        output_file=config.report_file,
    )

    print_terminal_summary(
        audits=audits,
        config=config,
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
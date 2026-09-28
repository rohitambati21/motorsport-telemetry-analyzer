"""
analyze_forza_session.py

FH6 free-roam drive-session metrics and event analysis.

Responsibilities:
- Load a normalized FH6 session CSV created by forza_adapter.py.
- Validate the normalized session schema.
- Compute session-level drive metrics.
- Detect raw threshold-based event candidates.
- Apply context filters to create a cleaner validated event table.
- Save a metrics CSV.
- Save a raw event-candidates CSV.
- Save a validated events CSV.
- Save a human-readable analysis report.

This module intentionally does not:
- run UDP capture
- decode packets
- normalize native FH6 data
- generate plots
- create fake laps
- run lap comparison
- calculate lap time delta
- calculate gain/loss regions
- interpret special FH6 gear values

Development role:
This is the V2 quantitative analysis layer for real FH6 free-roam telemetry.

FH6 native CSV
→ forza_adapter.py
→ normalized FH6 session CSV
→ plot_forza_session.py
→ analyze_forza_session.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

DEFAULT_NORMALIZED_DIRECTORY: Final[Path] = (
    PROJECT_ROOT / "data" / "forza" / "normalized"
)

DEFAULT_ANALYSIS_DIRECTORY: Final[Path] = (
    PROJECT_ROOT / "data" / "forza" / "analysis"
)

DEFAULT_REPORT_DIRECTORY: Final[Path] = (
    PROJECT_ROOT / "outputs" / "reports"
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "source",
    "session_id",
    "sample_index",
    "time_s",
    "session_distance_m",
    "distance_source",
    "speed_kph",
    "throttle_pct",
    "brake_pct",
    "steering_input_norm",
    "rpm",
    "gear_raw",
    "lat_g",
    "lon_g",
    "power_kw",
    "torque_nm",
)

OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (
    "speed_mps",
    "packet_length",
    "boost_psi",
    "fuel_fraction",
    "session_distance_native_m",
    "session_distance_position_m",
    "session_distance_speed_integrated_m",
    "position_x_m",
    "position_y_m",
    "position_z_m",
    "lap_number_native",
    "race_position_native",
    "is_race_on",
)

NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "sample_index",
    "time_s",
    "session_distance_m",
    "speed_kph",
    "throttle_pct",
    "brake_pct",
    "steering_input_norm",
    "rpm",
    "gear_raw",
    "lat_g",
    "lon_g",
    "power_kw",
    "torque_nm",
    *OPTIONAL_COLUMNS,
)

NON_NUMERIC_COLUMNS: Final[set[str]] = {
    "source",
    "session_id",
    "distance_source",
}

MOVING_SPEED_THRESHOLD_KPH: Final[float] = 1.0
FULL_THROTTLE_THRESHOLD_PCT: Final[float] = 99.0
HIGH_THROTTLE_THRESHOLD_PCT: Final[float] = 80.0
BRAKING_THRESHOLD_PCT: Final[float] = 1.0
HARD_BRAKING_THRESHOLD_PCT: Final[float] = 80.0
FULL_BRAKE_THRESHOLD_PCT: Final[float] = 99.0
INPUT_OVERLAP_THRESHOLD_PCT: Final[float] = 5.0

HIGH_SPEED_THRESHOLD_KPH: Final[float] = 100.0
VERY_HIGH_SPEED_THRESHOLD_KPH: Final[float] = 150.0

STRONG_ACCELERATION_G_THRESHOLD: Final[float] = 0.50
STRONG_DECELERATION_G_THRESHOLD: Final[float] = -0.80
HIGH_LATERAL_G_THRESHOLD: Final[float] = 0.50
MODERATE_STEERING_THRESHOLD: Final[float] = 0.30


@dataclass(frozen=True, slots=True)
class EventSpec:
    """
    Definition for one event detector.

    Threshold detection creates a raw candidate.
    Context filters decide whether that candidate becomes a validated event.
    """

    event_type: str
    value_column: str
    comparator: str
    threshold: float
    peak_mode: str
    peak_unit: str
    minimum_duration_s: float
    minimum_mean_speed_kph: float
    minimum_distance_span_m: float
    description: str


@dataclass(frozen=True, slots=True)
class AnalysisOutputs:
    """
    Paths and summary values from one analysis run.
    """

    input_file: Path
    metrics_file: Path
    event_candidates_file: Path
    events_file: Path
    report_file: Path
    session_id: str
    row_count: int
    duration_s: float
    session_distance_m: float
    distance_source: str
    metric_count: int
    candidate_count: int
    event_count: int
    rejected_candidate_count: int


EVENT_SPECS: Final[tuple[EventSpec, ...]] = (
    EventSpec(
        event_type="full_throttle",
        value_column="throttle_pct",
        comparator=">=",
        threshold=FULL_THROTTLE_THRESHOLD_PCT,
        peak_mode="max",
        peak_unit="%",
        minimum_duration_s=0.10,
        minimum_mean_speed_kph=0.0,
        minimum_distance_span_m=0.0,
        description="Throttle input at or above 99%.",
    ),
    EventSpec(
        event_type="high_throttle",
        value_column="throttle_pct",
        comparator=">=",
        threshold=HIGH_THROTTLE_THRESHOLD_PCT,
        peak_mode="max",
        peak_unit="%",
        minimum_duration_s=0.25,
        minimum_mean_speed_kph=0.0,
        minimum_distance_span_m=0.0,
        description="Throttle input at or above 80%.",
    ),
    EventSpec(
        event_type="hard_braking",
        value_column="brake_pct",
        comparator=">=",
        threshold=HARD_BRAKING_THRESHOLD_PCT,
        peak_mode="max",
        peak_unit="%",
        minimum_duration_s=0.25,
        minimum_mean_speed_kph=0.0,
        minimum_distance_span_m=0.0,
        description="Brake input at or above 80%.",
    ),
    EventSpec(
        event_type="full_brake",
        value_column="brake_pct",
        comparator=">=",
        threshold=FULL_BRAKE_THRESHOLD_PCT,
        peak_mode="max",
        peak_unit="%",
        minimum_duration_s=0.25,
        minimum_mean_speed_kph=0.0,
        minimum_distance_span_m=0.0,
        description="Brake input at or above 99%.",
    ),
    EventSpec(
        event_type="strong_acceleration",
        value_column="lon_g",
        comparator=">=",
        threshold=STRONG_ACCELERATION_G_THRESHOLD,
        peak_mode="max",
        peak_unit="g estimate",
        minimum_duration_s=0.25,
        minimum_mean_speed_kph=5.0,
        minimum_distance_span_m=1.0,
        description=(
            "Positive longitudinal acceleration estimate at or above 0.50 g."
        ),
    ),
    EventSpec(
        event_type="strong_deceleration",
        value_column="lon_g",
        comparator="<=",
        threshold=STRONG_DECELERATION_G_THRESHOLD,
        peak_mode="min",
        peak_unit="g estimate",
        minimum_duration_s=0.25,
        minimum_mean_speed_kph=5.0,
        minimum_distance_span_m=1.0,
        description=(
            "Negative longitudinal acceleration estimate at or below -0.80 g."
        ),
    ),
    EventSpec(
        event_type="high_lateral_g",
        value_column="lat_g",
        comparator="abs>=",
        threshold=HIGH_LATERAL_G_THRESHOLD,
        peak_mode="absmax",
        peak_unit="g estimate",
        minimum_duration_s=0.15,
        minimum_mean_speed_kph=5.0,
        minimum_distance_span_m=0.5,
        description=(
            "Absolute lateral acceleration estimate at or above 0.50 g."
        ),
    ),
    EventSpec(
        event_type="high_speed",
        value_column="speed_kph",
        comparator=">=",
        threshold=HIGH_SPEED_THRESHOLD_KPH,
        peak_mode="max",
        peak_unit="km/h",
        minimum_duration_s=0.50,
        minimum_mean_speed_kph=0.0,
        minimum_distance_span_m=0.0,
        description="Speed at or above 100 km/h.",
    ),
)


EVENT_CANDIDATE_COLUMNS: Final[tuple[str, ...]] = (
    "candidate_event_id",
    "event_type",
    "event_description",
    "context_valid",
    "rejection_reason",
    "start_sample",
    "end_sample",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "start_distance_m",
    "end_distance_m",
    "distance_span_m",
    "value_column",
    "threshold",
    "peak_value",
    "peak_value_unit",
    "mean_speed_kph",
    "max_speed_kph",
    "minimum_duration_s",
    "minimum_mean_speed_kph",
    "minimum_distance_span_m",
)

VALIDATED_EVENT_COLUMNS: Final[tuple[str, ...]] = (
    "event_id",
    "candidate_event_id",
    "event_type",
    "event_description",
    "start_sample",
    "end_sample",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "start_distance_m",
    "end_distance_m",
    "distance_span_m",
    "value_column",
    "threshold",
    "peak_value",
    "peak_value_unit",
    "mean_speed_kph",
    "max_speed_kph",
)


def find_latest_normalized_session(
    normalized_directory: Path,
) -> Path:
    """
    Find the most recently modified normalized FH6 session CSV.
    """

    if not normalized_directory.exists():
        raise FileNotFoundError(
            f"Normalized telemetry directory does not exist: {normalized_directory}"
        )

    candidates = sorted(
        normalized_directory.glob("*_normalized.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            "No normalized FH6 session CSV files were found in "
            f"{normalized_directory}"
        )

    return candidates[0]


def load_normalized_session(
    input_file: Path,
) -> pd.DataFrame:
    """
    Load and validate a normalized FH6 session CSV.
    """

    if not input_file.exists():
        raise FileNotFoundError(
            f"Normalized session CSV does not exist: {input_file}"
        )

    session_df = pd.read_csv(
        input_file
    )

    if session_df.empty:
        raise ValueError(
            f"Normalized session CSV is empty: {input_file}"
        )

    missing_columns = sorted(
        set(REQUIRED_COLUMNS) - set(session_df.columns)
    )

    if missing_columns:
        missing_text = "\n".join(
            f"  - {column}"
            for column in missing_columns
        )

        raise ValueError(
            "Normalized session CSV is missing required analysis columns:\n"
            f"{missing_text}"
        )

    return session_df


def coerce_numeric_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """
    Convert expected numeric columns to numeric dtype.
    """

    cleaned_df = df.copy()

    for column in columns:
        if (
            column in cleaned_df.columns
            and column not in NON_NUMERIC_COLUMNS
        ):
            cleaned_df[column] = pd.to_numeric(
                cleaned_df[column],
                errors="coerce",
            )

    return cleaned_df


def prepare_session_dataframe(
    session_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare normalized FH6 data for analysis.
    """

    cleaned_df = coerce_numeric_columns(
        session_df,
        NUMERIC_COLUMNS,
    )

    cleaned_df = cleaned_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if "sample_index" in cleaned_df.columns:
        cleaned_df = cleaned_df.sort_values(
            by="sample_index",
            kind="mergesort",
        )

    cleaned_df = cleaned_df.reset_index(
        drop=True
    )

    cleaned_df["sample_duration_s"] = calculate_sample_durations(
        cleaned_df["time_s"]
    )

    return cleaned_df


def get_session_id(
    session_df: pd.DataFrame,
    input_file: Path,
) -> str:
    """
    Resolve a stable session ID from the dataframe or filename.
    """

    if "session_id" in session_df.columns:
        valid_values = (
            session_df["session_id"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        unique_values = valid_values[
            valid_values != ""
        ].unique()

        if len(unique_values) == 1:
            return str(
                unique_values[0]
            )

    stem = input_file.stem

    if stem.endswith("_normalized"):
        stem = stem.removesuffix(
            "_normalized"
        )

    return stem


def get_primary_distance_source(
    session_df: pd.DataFrame,
) -> str:
    """
    Return the dominant distance source value.
    """

    if "distance_source" not in session_df.columns:
        return "unknown"

    valid_values = (
        session_df["distance_source"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    if valid_values.empty:
        return "unknown"

    return str(
        valid_values.mode().iloc[0]
    )


def finite_numeric_series(
    series: pd.Series,
) -> pd.Series:
    """
    Return finite numeric values only.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return numeric[
        np.isfinite(numeric)
    ]


def finite_min(
    series: pd.Series,
) -> float:
    """
    Return finite minimum or NaN.
    """

    numeric = finite_numeric_series(
        series
    )

    if numeric.empty:
        return math.nan

    return float(
        numeric.min()
    )


def finite_max(
    series: pd.Series,
) -> float:
    """
    Return finite maximum or NaN.
    """

    numeric = finite_numeric_series(
        series
    )

    if numeric.empty:
        return math.nan

    return float(
        numeric.max()
    )


def finite_mean(
    series: pd.Series,
) -> float:
    """
    Return finite arithmetic mean or NaN.
    """

    numeric = finite_numeric_series(
        series
    )

    if numeric.empty:
        return math.nan

    return float(
        numeric.mean()
    )


def finite_median(
    series: pd.Series,
) -> float:
    """
    Return finite median or NaN.
    """

    numeric = finite_numeric_series(
        series
    )

    if numeric.empty:
        return math.nan

    return float(
        numeric.median()
    )


def finite_std(
    series: pd.Series,
) -> float:
    """
    Return finite standard deviation or zero for a single finite value.
    """

    numeric = finite_numeric_series(
        series
    )

    if len(numeric) < 2:
        return 0.0

    return float(
        numeric.std()
    )


def finite_range(
    series: pd.Series,
) -> float:
    """
    Return finite max-min range or NaN.
    """

    minimum = finite_min(
        series
    )

    maximum = finite_max(
        series
    )

    if (
        math.isnan(minimum)
        or math.isnan(maximum)
    ):
        return math.nan

    return maximum - minimum


def finite_range_text(
    series: pd.Series,
    precision: int = 3,
) -> str:
    """
    Format a finite numeric range for reports.
    """

    minimum = finite_min(
        series
    )

    maximum = finite_max(
        series
    )

    if (
        math.isnan(minimum)
        or math.isnan(maximum)
    ):
        return "not available"

    return (
        f"{minimum:.{precision}f} to "
        f"{maximum:.{precision}f}"
    )


def calculate_sample_durations(
    time_s: pd.Series,
) -> pd.Series:
    """
    Calculate per-sample durations using the interval to the next sample.

    The final sample receives zero duration so total duration equals the
    actual time range of the session.
    """

    time_values = pd.to_numeric(
        time_s,
        errors="coerce",
    ).to_numpy(dtype="float64")

    durations = np.zeros_like(
        time_values,
        dtype="float64",
    )

    if len(time_values) < 2:
        return pd.Series(
            durations,
            index=time_s.index,
            name="sample_duration_s",
        )

    dt = np.diff(
        time_values
    )

    valid_dt = (
        np.isfinite(dt)
        & (dt >= 0.0)
    )

    step_durations = np.zeros_like(
        dt,
        dtype="float64",
    )

    step_durations[valid_dt] = dt[valid_dt]

    durations[:-1] = step_durations

    return pd.Series(
        durations,
        index=time_s.index,
        name="sample_duration_s",
    )


def calculate_duration_s(
    df: pd.DataFrame,
) -> float:
    """
    Calculate session duration from time_s.
    """

    return finite_range(
        df["time_s"]
    )


def calculate_session_distance_m(
    df: pd.DataFrame,
) -> float:
    """
    Calculate selected session distance from session_distance_m.
    """

    return finite_range(
        df["session_distance_m"]
    )


def weighted_mean(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    """
    Calculate a finite weighted mean.
    """

    value_array = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(dtype="float64")

    weight_array = pd.to_numeric(
        weights,
        errors="coerce",
    ).to_numpy(dtype="float64")

    valid = (
        np.isfinite(value_array)
        & np.isfinite(weight_array)
        & (weight_array > 0.0)
    )

    if not valid.any():
        return math.nan

    return float(
        np.average(
            value_array[valid],
            weights=weight_array[valid],
        )
    )


def time_where(
    mask: pd.Series | np.ndarray,
    sample_duration_s: pd.Series,
) -> float:
    """
    Sum sample durations where a boolean condition is true.
    """

    mask_array = np.asarray(
        mask,
        dtype=bool,
    )

    duration_array = pd.to_numeric(
        sample_duration_s,
        errors="coerce",
    ).to_numpy(dtype="float64")

    valid = (
        mask_array
        & np.isfinite(duration_array)
        & (duration_array > 0.0)
    )

    return float(
        duration_array[valid].sum()
    )


def percent_of_duration(
    partial_duration_s: float,
    total_duration_s: float,
) -> float:
    """
    Convert a partial duration into percent of total duration.
    """

    if total_duration_s <= 0.0:
        return 0.0

    return (
        partial_duration_s
        / total_duration_s
        * 100.0
    )


def add_metric(
    metrics: list[dict[str, object]],
    metric: str,
    value: object,
    unit: str,
    notes: str = "",
) -> None:
    """
    Append one metric row.
    """

    metrics.append(
        {
            "metric": metric,
            "value": value,
            "unit": unit,
            "notes": notes,
        }
    )


def calculate_steering_activity_per_second(
    steering_input: pd.Series,
    duration_s: float,
) -> float:
    """
    Calculate normalized steering activity per second.
    """

    if duration_s <= 0.0:
        return 0.0

    steering_values = pd.to_numeric(
        steering_input,
        errors="coerce",
    ).to_numpy(dtype="float64")

    valid = np.isfinite(
        steering_values
    )

    if valid.sum() < 2:
        return 0.0

    valid_values = steering_values[
        valid
    ]

    activity = np.abs(
        np.diff(
            valid_values
        )
    ).sum()

    return float(
        activity / duration_s
    )


def append_gear_usage_metrics(
    metrics: list[dict[str, object]],
    df: pd.DataFrame,
    duration_s: float,
) -> None:
    """
    Append raw gear usage time and percent metrics.

    gear_raw is intentionally not interpreted.
    """

    gear_series = pd.to_numeric(
        df["gear_raw"],
        errors="coerce",
    )

    valid_gears = sorted(
        gear_series.dropna().astype(int).unique().tolist()
    )

    for gear_value in valid_gears:
        gear_mask = (
            gear_series == gear_value
        )

        gear_time_s = time_where(
            gear_mask,
            df["sample_duration_s"],
        )

        add_metric(
            metrics,
            f"gear_raw_{gear_value}_time_s",
            gear_time_s,
            "s",
            f"Time spent with raw gear value {gear_value}.",
        )

        add_metric(
            metrics,
            f"gear_raw_{gear_value}_time_pct",
            percent_of_duration(
                gear_time_s,
                duration_s,
            ),
            "%",
            f"Percent of session with raw gear value {gear_value}.",
        )


def build_session_metrics(
    df: pd.DataFrame,
    session_id: str,
) -> pd.DataFrame:
    """
    Build a tidy metrics table for one free-roam session.
    """

    metrics: list[dict[str, object]] = []

    duration_s = calculate_duration_s(
        df
    )

    session_distance_m = calculate_session_distance_m(
        df
    )

    distance_source = get_primary_distance_source(
        df
    )

    sample_duration_s = df["sample_duration_s"]

    moving_mask = (
        df["speed_kph"] > MOVING_SPEED_THRESHOLD_KPH
    )

    stationary_mask = ~moving_mask

    moving_time_s = time_where(
        moving_mask,
        sample_duration_s,
    )

    stationary_time_s = time_where(
        stationary_mask,
        sample_duration_s,
    )

    distance_average_speed_kph = (
        (session_distance_m / 1000.0)
        / (duration_s / 3600.0)
    ) if duration_s > 0.0 else math.nan

    add_metric(
        metrics,
        "session_id",
        session_id,
        "",
        "Identifier carried from the normalized FH6 session file.",
    )

    add_metric(
        metrics,
        "row_count",
        int(len(df)),
        "rows",
        "Number of normalized samples analyzed.",
    )

    add_metric(
        metrics,
        "duration_s",
        duration_s,
        "s",
        "Session duration from time_s.",
    )

    add_metric(
        metrics,
        "session_distance_m",
        session_distance_m,
        "m",
        "Selected session distance from adapter-selected session_distance_m.",
    )

    add_metric(
        metrics,
        "session_distance_km",
        session_distance_m / 1000.0,
        "km",
        "Selected session distance in kilometers.",
    )

    add_metric(
        metrics,
        "distance_source",
        distance_source,
        "",
        "Dominant adapter-selected distance source.",
    )

    add_metric(
        metrics,
        "average_speed_kph_time_weighted",
        weighted_mean(
            df["speed_kph"],
            sample_duration_s,
        ),
        "km/h",
        "Time-weighted average speed.",
    )

    add_metric(
        metrics,
        "average_speed_kph_distance_based",
        distance_average_speed_kph,
        "km/h",
        "Distance divided by session duration.",
    )

    add_metric(
        metrics,
        "median_speed_kph",
        finite_median(
            df["speed_kph"]
        ),
        "km/h",
        "Median speed across samples.",
    )

    add_metric(
        metrics,
        "max_speed_kph",
        finite_max(
            df["speed_kph"]
        ),
        "km/h",
        "Maximum speed observed.",
    )

    add_metric(
        metrics,
        "speed_std_kph",
        finite_std(
            df["speed_kph"]
        ),
        "km/h",
        "Sample standard deviation of speed.",
    )

    add_metric(
        metrics,
        "moving_time_s",
        moving_time_s,
        "s",
        f"Time above {MOVING_SPEED_THRESHOLD_KPH:.1f} km/h.",
    )

    add_metric(
        metrics,
        "stationary_time_s",
        stationary_time_s,
        "s",
        f"Time at or below {MOVING_SPEED_THRESHOLD_KPH:.1f} km/h.",
    )

    add_metric(
        metrics,
        "moving_time_pct",
        percent_of_duration(
            moving_time_s,
            duration_s,
        ),
        "%",
        "Moving time as percent of session duration.",
    )

    for speed_threshold in (
        50.0,
        100.0,
        150.0,
    ):
        speed_time_s = time_where(
            df["speed_kph"] >= speed_threshold,
            sample_duration_s,
        )

        add_metric(
            metrics,
            f"time_above_{int(speed_threshold)}_kph_s",
            speed_time_s,
            "s",
            f"Time at or above {speed_threshold:.0f} km/h.",
        )

        add_metric(
            metrics,
            f"time_above_{int(speed_threshold)}_kph_pct",
            percent_of_duration(
                speed_time_s,
                duration_s,
            ),
            "%",
            f"Percent of session at or above {speed_threshold:.0f} km/h.",
        )

    add_metric(
        metrics,
        "mean_throttle_pct_time_weighted",
        weighted_mean(
            df["throttle_pct"],
            sample_duration_s,
        ),
        "%",
        "Time-weighted average throttle input.",
    )

    add_metric(
        metrics,
        "max_throttle_pct",
        finite_max(
            df["throttle_pct"]
        ),
        "%",
        "Maximum throttle input.",
    )

    full_throttle_time_s = time_where(
        df["throttle_pct"] >= FULL_THROTTLE_THRESHOLD_PCT,
        sample_duration_s,
    )

    high_throttle_time_s = time_where(
        df["throttle_pct"] >= HIGH_THROTTLE_THRESHOLD_PCT,
        sample_duration_s,
    )

    add_metric(
        metrics,
        "full_throttle_time_s",
        full_throttle_time_s,
        "s",
        f"Time at or above {FULL_THROTTLE_THRESHOLD_PCT:.0f}% throttle.",
    )

    add_metric(
        metrics,
        "full_throttle_time_pct",
        percent_of_duration(
            full_throttle_time_s,
            duration_s,
        ),
        "%",
        "Full-throttle time as percent of session duration.",
    )

    add_metric(
        metrics,
        "high_throttle_time_s",
        high_throttle_time_s,
        "s",
        f"Time at or above {HIGH_THROTTLE_THRESHOLD_PCT:.0f}% throttle.",
    )

    add_metric(
        metrics,
        "mean_brake_pct_time_weighted",
        weighted_mean(
            df["brake_pct"],
            sample_duration_s,
        ),
        "%",
        "Time-weighted average brake input.",
    )

    add_metric(
        metrics,
        "max_brake_pct",
        finite_max(
            df["brake_pct"]
        ),
        "%",
        "Maximum brake input.",
    )

    braking_time_s = time_where(
        df["brake_pct"] > BRAKING_THRESHOLD_PCT,
        sample_duration_s,
    )

    hard_braking_time_s = time_where(
        df["brake_pct"] >= HARD_BRAKING_THRESHOLD_PCT,
        sample_duration_s,
    )

    full_brake_time_s = time_where(
        df["brake_pct"] >= FULL_BRAKE_THRESHOLD_PCT,
        sample_duration_s,
    )

    add_metric(
        metrics,
        "braking_time_s",
        braking_time_s,
        "s",
        f"Time above {BRAKING_THRESHOLD_PCT:.0f}% brake input.",
    )

    add_metric(
        metrics,
        "braking_time_pct",
        percent_of_duration(
            braking_time_s,
            duration_s,
        ),
        "%",
        "Brake usage time as percent of session duration.",
    )

    add_metric(
        metrics,
        "hard_braking_time_s",
        hard_braking_time_s,
        "s",
        f"Time at or above {HARD_BRAKING_THRESHOLD_PCT:.0f}% brake input.",
    )

    add_metric(
        metrics,
        "full_brake_time_s",
        full_brake_time_s,
        "s",
        f"Time at or above {FULL_BRAKE_THRESHOLD_PCT:.0f}% brake input.",
    )

    overlap_time_s = time_where(
        (
            (df["throttle_pct"] > INPUT_OVERLAP_THRESHOLD_PCT)
            & (df["brake_pct"] > INPUT_OVERLAP_THRESHOLD_PCT)
        ),
        sample_duration_s,
    )

    add_metric(
        metrics,
        "throttle_brake_overlap_time_s",
        overlap_time_s,
        "s",
        (
            "Time with throttle and brake both above "
            f"{INPUT_OVERLAP_THRESHOLD_PCT:.0f}%."
        ),
    )

    add_metric(
        metrics,
        "peak_lateral_g_abs",
        finite_max(
            df["lat_g"].abs()
        ),
        "g estimate",
        "Peak absolute lateral acceleration estimate.",
    )

    add_metric(
        metrics,
        "peak_positive_longitudinal_g",
        finite_max(
            df["lon_g"]
        ),
        "g estimate",
        "Peak positive longitudinal acceleration estimate.",
    )

    add_metric(
        metrics,
        "peak_negative_longitudinal_g",
        finite_min(
            df["lon_g"]
        ),
        "g estimate",
        "Peak negative longitudinal acceleration estimate.",
    )

    add_metric(
        metrics,
        "mean_abs_lateral_g_time_weighted",
        weighted_mean(
            df["lat_g"].abs(),
            sample_duration_s,
        ),
        "g estimate",
        "Time-weighted mean absolute lateral acceleration estimate.",
    )

    add_metric(
        metrics,
        "mean_abs_longitudinal_g_time_weighted",
        weighted_mean(
            df["lon_g"].abs(),
            sample_duration_s,
        ),
        "g estimate",
        "Time-weighted mean absolute longitudinal acceleration estimate.",
    )

    add_metric(
        metrics,
        "max_abs_steering_input",
        finite_max(
            df["steering_input_norm"].abs()
        ),
        "normalized input",
        "Maximum absolute normalized steering input.",
    )

    add_metric(
        metrics,
        "mean_abs_steering_input_time_weighted",
        weighted_mean(
            df["steering_input_norm"].abs(),
            sample_duration_s,
        ),
        "normalized input",
        "Time-weighted mean absolute steering input.",
    )

    add_metric(
        metrics,
        "steering_activity_per_s",
        calculate_steering_activity_per_second(
            df["steering_input_norm"],
            duration_s,
        ),
        "normalized input change / s",
        "Sum of absolute steering-input changes divided by session duration.",
    )

    moderate_steering_time_s = time_where(
        df["steering_input_norm"].abs() >= MODERATE_STEERING_THRESHOLD,
        sample_duration_s,
    )

    add_metric(
        metrics,
        "moderate_steering_time_s",
        moderate_steering_time_s,
        "s",
        (
            "Time with absolute steering input at or above "
            f"{MODERATE_STEERING_THRESHOLD:.2f}."
        ),
    )

    add_metric(
        metrics,
        "min_rpm",
        finite_min(
            df["rpm"]
        ),
        "rpm",
        "Minimum engine speed.",
    )

    add_metric(
        metrics,
        "mean_rpm_time_weighted",
        weighted_mean(
            df["rpm"],
            sample_duration_s,
        ),
        "rpm",
        "Time-weighted mean engine speed.",
    )

    add_metric(
        metrics,
        "max_rpm",
        finite_max(
            df["rpm"]
        ),
        "rpm",
        "Maximum engine speed.",
    )

    add_metric(
        metrics,
        "min_power_kw",
        finite_min(
            df["power_kw"]
        ),
        "kW",
        "Minimum power value observed.",
    )

    add_metric(
        metrics,
        "max_power_kw",
        finite_max(
            df["power_kw"]
        ),
        "kW",
        "Maximum power value observed.",
    )

    add_metric(
        metrics,
        "min_torque_nm",
        finite_min(
            df["torque_nm"]
        ),
        "Nm",
        "Minimum torque value observed.",
    )

    add_metric(
        metrics,
        "max_torque_nm",
        finite_max(
            df["torque_nm"]
        ),
        "Nm",
        "Maximum torque value observed.",
    )

    append_gear_usage_metrics(
        metrics,
        df,
        duration_s,
    )

    return pd.DataFrame(
        metrics,
        columns=(
            "metric",
            "value",
            "unit",
            "notes",
        ),
    )


def build_event_mask(
    df: pd.DataFrame,
    event_spec: EventSpec,
) -> pd.Series:
    """
    Build a boolean event mask from an event specification.
    """

    values = pd.to_numeric(
        df[event_spec.value_column],
        errors="coerce",
    )

    if event_spec.comparator == ">=":
        return values >= event_spec.threshold

    if event_spec.comparator == "<=":
        return values <= event_spec.threshold

    if event_spec.comparator == "abs>=":
        return values.abs() >= event_spec.threshold

    raise ValueError(
        f"Unsupported comparator: {event_spec.comparator}"
    )


def contiguous_true_regions(
    mask: pd.Series | np.ndarray,
) -> list[tuple[int, int]]:
    """
    Return inclusive start/end index pairs for contiguous True regions.
    """

    mask_array = np.asarray(
        mask,
        dtype=bool,
    )

    regions: list[tuple[int, int]] = []
    start_index: int | None = None

    for index, value in enumerate(
        mask_array
    ):
        if value and start_index is None:
            start_index = index

        elif not value and start_index is not None:
            regions.append(
                (
                    start_index,
                    index - 1,
                )
            )

            start_index = None

    if start_index is not None:
        regions.append(
            (
                start_index,
                len(mask_array) - 1,
            )
        )

    return regions


def calculate_peak_value(
    values: pd.Series,
    peak_mode: str,
) -> float:
    """
    Calculate the peak value for an event region.
    """

    numeric = finite_numeric_series(
        values
    )

    if numeric.empty:
        return math.nan

    if peak_mode == "max":
        return float(
            numeric.max()
        )

    if peak_mode == "min":
        return float(
            numeric.min()
        )

    if peak_mode == "absmax":
        absolute_values = numeric.abs()

        peak_index = absolute_values.idxmax()

        return float(
            numeric.loc[peak_index]
        )

    raise ValueError(
        f"Unsupported peak_mode: {peak_mode}"
    )


def validate_event_context(
    duration_s: float,
    mean_speed_kph: float,
    distance_span_m: float,
    event_spec: EventSpec,
) -> tuple[bool, str]:
    """
    Apply event-specific context filters.

    This separates raw threshold detection from event qualification.
    """

    rejection_reasons: list[str] = []

    if duration_s < event_spec.minimum_duration_s:
        rejection_reasons.append(
            (
                f"duration {duration_s:.3f} s below minimum "
                f"{event_spec.minimum_duration_s:.3f} s"
            )
        )

    if event_spec.minimum_mean_speed_kph > 0.0:
        if (
            math.isnan(mean_speed_kph)
            or mean_speed_kph < event_spec.minimum_mean_speed_kph
        ):
            rejection_reasons.append(
                (
                    f"mean speed {mean_speed_kph:.3f} km/h below minimum "
                    f"{event_spec.minimum_mean_speed_kph:.3f} km/h"
                )
            )

    if event_spec.minimum_distance_span_m > 0.0:
        if (
            math.isnan(distance_span_m)
            or distance_span_m < event_spec.minimum_distance_span_m
        ):
            rejection_reasons.append(
                (
                    f"distance span {distance_span_m:.3f} m below minimum "
                    f"{event_spec.minimum_distance_span_m:.3f} m"
                )
            )

    if rejection_reasons:
        return (
            False,
            "; ".join(
                rejection_reasons
            ),
        )

    return (
        True,
        "accepted",
    )


def detect_event_candidates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect raw threshold-based event candidates and apply context validation.
    """

    candidate_rows: list[dict[str, object]] = []
    candidate_event_id = 1

    for event_spec in EVENT_SPECS:
        mask = build_event_mask(
            df,
            event_spec,
        )

        regions = contiguous_true_regions(
            mask
        )

        for start_index, end_index in regions:
            region_df = df.iloc[
                start_index : end_index + 1
            ]

            duration_s = float(
                region_df["sample_duration_s"].sum()
            )

            start_time_s = float(
                region_df["time_s"].iloc[0]
            )

            end_time_s = (
                start_time_s
                + duration_s
            )

            start_distance_m = float(
                region_df["session_distance_m"].iloc[0]
            )

            end_distance_m = float(
                region_df["session_distance_m"].iloc[-1]
            )

            distance_span_m = (
                end_distance_m
                - start_distance_m
            )

            peak_value = calculate_peak_value(
                region_df[event_spec.value_column],
                event_spec.peak_mode,
            )

            mean_speed_kph = weighted_mean(
                region_df["speed_kph"],
                region_df["sample_duration_s"],
            )

            max_speed_kph = finite_max(
                region_df["speed_kph"]
            )

            context_valid, rejection_reason = validate_event_context(
                duration_s=duration_s,
                mean_speed_kph=mean_speed_kph,
                distance_span_m=distance_span_m,
                event_spec=event_spec,
            )

            candidate_rows.append(
                {
                    "candidate_event_id": candidate_event_id,
                    "event_type": event_spec.event_type,
                    "event_description": event_spec.description,
                    "context_valid": context_valid,
                    "rejection_reason": rejection_reason,
                    "start_sample": int(start_index),
                    "end_sample": int(end_index),
                    "start_time_s": start_time_s,
                    "end_time_s": end_time_s,
                    "duration_s": duration_s,
                    "start_distance_m": start_distance_m,
                    "end_distance_m": end_distance_m,
                    "distance_span_m": distance_span_m,
                    "value_column": event_spec.value_column,
                    "threshold": event_spec.threshold,
                    "peak_value": peak_value,
                    "peak_value_unit": event_spec.peak_unit,
                    "mean_speed_kph": mean_speed_kph,
                    "max_speed_kph": max_speed_kph,
                    "minimum_duration_s": event_spec.minimum_duration_s,
                    "minimum_mean_speed_kph": event_spec.minimum_mean_speed_kph,
                    "minimum_distance_span_m": event_spec.minimum_distance_span_m,
                }
            )

            candidate_event_id += 1

    if not candidate_rows:
        return pd.DataFrame(
            columns=EVENT_CANDIDATE_COLUMNS
        )

    candidates_df = pd.DataFrame(
        candidate_rows
    )

    candidates_df = candidates_df.sort_values(
        by=[
            "start_time_s",
            "event_type",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    candidates_df["candidate_event_id"] = np.arange(
        1,
        len(candidates_df) + 1,
        dtype=int,
    )

    return candidates_df.loc[
        :,
        EVENT_CANDIDATE_COLUMNS,
    ]


def build_validated_events(
    candidates_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the final validated event table from accepted candidates.
    """

    if candidates_df.empty:
        return pd.DataFrame(
            columns=VALIDATED_EVENT_COLUMNS
        )

    accepted_df = candidates_df.loc[
        candidates_df["context_valid"] == True  # noqa: E712
    ].copy()

    if accepted_df.empty:
        return pd.DataFrame(
            columns=VALIDATED_EVENT_COLUMNS
        )

    accepted_df = accepted_df.sort_values(
        by=[
            "start_time_s",
            "event_type",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    accepted_df.insert(
        0,
        "event_id",
        np.arange(
            1,
            len(accepted_df) + 1,
            dtype=int,
        ),
    )

    return accepted_df.loc[
        :,
        VALIDATED_EVENT_COLUMNS,
    ]


def metric_value(
    metrics_df: pd.DataFrame,
    metric_name: str,
    default: object = "not available",
) -> object:
    """
    Fetch one metric value from the metrics table.
    """

    matches = metrics_df.loc[
        metrics_df["metric"] == metric_name,
        "value",
    ]

    if matches.empty:
        return default

    return matches.iloc[0]


def format_float(
    value: object,
    precision: int = 3,
) -> str:
    """
    Format a numeric value for reports.
    """

    try:
        numeric_value = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return str(
            value
        )

    if math.isnan(
        numeric_value
    ):
        return "not available"

    return f"{numeric_value:.{precision}f}"


def event_count_summary(
    events_df: pd.DataFrame,
) -> list[str]:
    """
    Build event count lines by event type.
    """

    if events_df.empty:
        return [
            "  none"
        ]

    counts = (
        events_df["event_type"]
        .value_counts()
        .sort_index()
    )

    return [
        f"  {event_type}: {count}"
        for event_type, count in counts.items()
    ]


def top_event_lines(
    events_df: pd.DataFrame,
    maximum_events: int = 12,
) -> list[str]:
    """
    Build compact validated-event lines for the text report.
    """

    if events_df.empty:
        return [
            "  none"
        ]

    sorted_events = events_df.sort_values(
        by=[
            "start_time_s",
            "event_type",
        ],
        kind="mergesort",
    ).head(
        maximum_events
    )

    lines: list[str] = []

    for _, row in sorted_events.iterrows():
        lines.append(
            "  "
            f"#{int(row['event_id'])} "
            f"{row['event_type']} | "
            f"{float(row['start_time_s']):.3f}–{float(row['end_time_s']):.3f} s | "
            f"{float(row['start_distance_m']):.1f}–{float(row['end_distance_m']):.1f} m | "
            f"duration {float(row['duration_s']):.3f} s | "
            f"peak {float(row['peak_value']):.3f} {row['peak_value_unit']}"
        )

    if len(events_df) > maximum_events:
        lines.append(
            f"  ... {len(events_df) - maximum_events} additional events"
        )

    return lines


def rejected_event_lines(
    candidates_df: pd.DataFrame,
    maximum_events: int = 10,
) -> list[str]:
    """
    Build compact rejected-candidate lines for the text report.
    """

    if candidates_df.empty:
        return [
            "  none"
        ]

    rejected_df = candidates_df.loc[
        candidates_df["context_valid"] == False  # noqa: E712
    ].copy()

    if rejected_df.empty:
        return [
            "  none"
        ]

    rejected_df = rejected_df.sort_values(
        by=[
            "start_time_s",
            "event_type",
        ],
        kind="mergesort",
    ).head(
        maximum_events
    )

    lines: list[str] = []

    for _, row in rejected_df.iterrows():
        lines.append(
            "  "
            f"candidate #{int(row['candidate_event_id'])} "
            f"{row['event_type']} | "
            f"{float(row['start_time_s']):.3f}–{float(row['end_time_s']):.3f} s | "
            f"{float(row['start_distance_m']):.3f}–{float(row['end_distance_m']):.3f} m | "
            f"reason: {row['rejection_reason']}"
        )

    total_rejected = int(
        (candidates_df["context_valid"] == False).sum()  # noqa: E712
    )

    if total_rejected > maximum_events:
        lines.append(
            f"  ... {total_rejected - maximum_events} additional rejected candidates"
        )

    return lines


def rejection_count_summary(
    candidates_df: pd.DataFrame,
) -> list[str]:
    """
    Build rejection-reason count lines.
    """

    if candidates_df.empty:
        return [
            "  none"
        ]

    rejected_df = candidates_df.loc[
        candidates_df["context_valid"] == False  # noqa: E712
    ]

    if rejected_df.empty:
        return [
            "  none"
        ]

    counts = (
        rejected_df["rejection_reason"]
        .value_counts()
        .sort_index()
    )

    return [
        f"  {reason}: {count}"
        for reason, count in counts.items()
    ]


def build_analysis_report(
    input_file: Path,
    metrics_file: Path,
    event_candidates_file: Path,
    events_file: Path,
    report_file: Path,
    session_id: str,
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    event_candidates_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> str:
    """
    Build a human-readable drive-session analysis report.
    """

    duration_s = calculate_duration_s(
        df
    )

    distance_m = calculate_session_distance_m(
        df
    )

    distance_source = get_primary_distance_source(
        df
    )

    candidate_count = len(
        event_candidates_df
    )

    event_count = len(
        events_df
    )

    rejected_count = int(
        candidate_count
        - event_count
    )

    lines: list[str] = []

    lines.extend(
        [
            "FH6 FREE-ROAM DRIVE-SESSION ANALYSIS",
            "=" * 72,
            "",
            "FILES",
            "-" * 72,
            f"Input normalized CSV: {input_file}",
            f"Session metrics CSV: {metrics_file}",
            f"Event candidates CSV: {event_candidates_file}",
            f"Validated events CSV: {events_file}",
            f"Report file: {report_file}",
            "",
            "SESSION",
            "-" * 72,
            f"Session ID: {session_id}",
            f"Rows analyzed: {len(df)}",
            f"Duration: {duration_s:.3f} s",
            f"Selected session distance: {distance_m:.3f} m",
            f"Selected session distance: {distance_m / 1000.0:.3f} km",
            f"Distance source: {distance_source}",
            "",
            "SPEED SUMMARY",
            "-" * 72,
            f"Average speed, time-weighted: {format_float(metric_value(metrics_df, 'average_speed_kph_time_weighted'), 2)} km/h",
            f"Average speed, distance/duration: {format_float(metric_value(metrics_df, 'average_speed_kph_distance_based'), 2)} km/h",
            f"Median speed: {format_float(metric_value(metrics_df, 'median_speed_kph'), 2)} km/h",
            f"Max speed: {format_float(metric_value(metrics_df, 'max_speed_kph'), 2)} km/h",
            f"Moving time: {format_float(metric_value(metrics_df, 'moving_time_s'), 3)} s",
            f"Moving time: {format_float(metric_value(metrics_df, 'moving_time_pct'), 2)} %",
            f"Time above 100 km/h: {format_float(metric_value(metrics_df, 'time_above_100_kph_s'), 3)} s",
            f"Time above 150 km/h: {format_float(metric_value(metrics_df, 'time_above_150_kph_s'), 3)} s",
            "",
            "DRIVER INPUT SUMMARY",
            "-" * 72,
            f"Mean throttle: {format_float(metric_value(metrics_df, 'mean_throttle_pct_time_weighted'), 2)} %",
            f"Max throttle: {format_float(metric_value(metrics_df, 'max_throttle_pct'), 2)} %",
            f"Full-throttle time: {format_float(metric_value(metrics_df, 'full_throttle_time_s'), 3)} s",
            f"High-throttle time: {format_float(metric_value(metrics_df, 'high_throttle_time_s'), 3)} s",
            f"Mean brake: {format_float(metric_value(metrics_df, 'mean_brake_pct_time_weighted'), 2)} %",
            f"Max brake: {format_float(metric_value(metrics_df, 'max_brake_pct'), 2)} %",
            f"Braking time: {format_float(metric_value(metrics_df, 'braking_time_s'), 3)} s",
            f"Hard-braking time: {format_float(metric_value(metrics_df, 'hard_braking_time_s'), 3)} s",
            f"Full-brake time: {format_float(metric_value(metrics_df, 'full_brake_time_s'), 3)} s",
            f"Throttle/brake overlap time: {format_float(metric_value(metrics_df, 'throttle_brake_overlap_time_s'), 3)} s",
            "",
            "VEHICLE RESPONSE SUMMARY",
            "-" * 72,
            f"Peak absolute lateral g estimate: {format_float(metric_value(metrics_df, 'peak_lateral_g_abs'), 4)} g",
            f"Peak positive longitudinal g estimate: {format_float(metric_value(metrics_df, 'peak_positive_longitudinal_g'), 4)} g",
            f"Peak negative longitudinal g estimate: {format_float(metric_value(metrics_df, 'peak_negative_longitudinal_g'), 4)} g",
            f"Mean absolute lateral g estimate: {format_float(metric_value(metrics_df, 'mean_abs_lateral_g_time_weighted'), 4)} g",
            f"Mean absolute longitudinal g estimate: {format_float(metric_value(metrics_df, 'mean_abs_longitudinal_g_time_weighted'), 4)} g",
            "",
            "STEERING SUMMARY",
            "-" * 72,
            f"Max absolute steering input: {format_float(metric_value(metrics_df, 'max_abs_steering_input'), 4)}",
            f"Mean absolute steering input: {format_float(metric_value(metrics_df, 'mean_abs_steering_input_time_weighted'), 4)}",
            f"Steering activity: {format_float(metric_value(metrics_df, 'steering_activity_per_s'), 4)} normalized input change / s",
            f"Moderate steering time: {format_float(metric_value(metrics_df, 'moderate_steering_time_s'), 3)} s",
            "",
            "POWERTRAIN SUMMARY",
            "-" * 72,
            f"RPM range: {finite_range_text(df['rpm'], 0)} rpm",
            f"Mean RPM: {format_float(metric_value(metrics_df, 'mean_rpm_time_weighted'), 0)} rpm",
            f"Power range: {finite_range_text(df['power_kw'], 3)} kW",
            f"Torque range: {finite_range_text(df['torque_nm'], 3)} Nm",
            f"Raw gear values: {sorted(pd.to_numeric(df['gear_raw'], errors='coerce').dropna().astype(int).unique().tolist())}",
            "",
            "EVENT DETECTION SUMMARY",
            "-" * 72,
            f"Raw event candidates detected: {candidate_count}",
            f"Validated events: {event_count}",
            f"Rejected candidates: {rejected_count}",
            "",
            "Validated event counts by type:",
            *event_count_summary(events_df),
            "",
            "Rejected candidate reasons:",
            *rejection_count_summary(event_candidates_df),
            "",
            "First validated events:",
            *top_event_lines(events_df),
            "",
            "Rejected candidates:",
            *rejected_event_lines(event_candidates_df),
            "",
            "DESIGN NOTES",
            "-" * 72,
            "This analysis is session-level free-roam telemetry analysis.",
            "It does not create fake laps or run lap comparison.",
            "Distance uses adapter-selected session_distance_m.",
            "gear_raw remains uninterpreted.",
            "steering_input_norm is normalized FH6 steering input, not physical steering angle.",
            "lat_g and lon_g are acceleration estimates; sign convention still requires controlled validation.",
            "Event detection is threshold-based, then context-filtered.",
            "The event candidates CSV preserves raw threshold detections.",
            "The validated events CSV contains only candidates that passed duration, speed, and distance-span checks.",
            "",
            "NEXT STEP",
            "-" * 72,
            "Validate the event-candidate and validated-event tables against the previously reviewed plots.",
            "After validation, the next layer can refine thresholds further or support multi-session comparison.",
            "",
            "Report generated automatically by analyze_forza_session.py.",
        ]
    )

    return "\n".join(
        lines
    )


def build_output_paths(
    session_id: str,
    analysis_directory: Path,
    report_directory: Path,
) -> tuple[Path, Path, Path, Path]:
    """
    Build metrics, candidates, validated events, and report output paths.
    """

    metrics_file = (
        analysis_directory
        / f"{session_id}_session_metrics.csv"
    )

    event_candidates_file = (
        analysis_directory
        / f"{session_id}_detected_event_candidates.csv"
    )

    events_file = (
        analysis_directory
        / f"{session_id}_detected_events.csv"
    )

    report_file = (
        report_directory
        / f"{session_id}_drive_session_analysis.txt"
    )

    return (
        metrics_file,
        event_candidates_file,
        events_file,
        report_file,
    )


def write_dataframe(
    df: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    Write a dataframe to CSV.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_file,
        index=False,
    )


def write_report(
    report_text: str,
    report_file: Path,
) -> None:
    """
    Write a text report.
    """

    report_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_file.write_text(
        report_text,
        encoding="utf-8",
    )


def run_analysis_pipeline(
    input_file: Path,
    analysis_directory: Path,
    report_directory: Path,
) -> AnalysisOutputs:
    """
    Execute the complete FH6 free-roam analysis pipeline.
    """

    raw_df = load_normalized_session(
        input_file
    )

    df = prepare_session_dataframe(
        raw_df
    )

    session_id = get_session_id(
        df,
        input_file,
    )

    metrics_df = build_session_metrics(
        df,
        session_id,
    )

    event_candidates_df = detect_event_candidates(
        df
    )

    events_df = build_validated_events(
        event_candidates_df
    )

    (
        metrics_file,
        event_candidates_file,
        events_file,
        report_file,
    ) = build_output_paths(
        session_id=session_id,
        analysis_directory=analysis_directory,
        report_directory=report_directory,
    )

    write_dataframe(
        metrics_df,
        metrics_file,
    )

    write_dataframe(
        event_candidates_df,
        event_candidates_file,
    )

    write_dataframe(
        events_df,
        events_file,
    )

    report_text = build_analysis_report(
        input_file=input_file,
        metrics_file=metrics_file,
        event_candidates_file=event_candidates_file,
        events_file=events_file,
        report_file=report_file,
        session_id=session_id,
        df=df,
        metrics_df=metrics_df,
        event_candidates_df=event_candidates_df,
        events_df=events_df,
    )

    write_report(
        report_text,
        report_file,
    )

    return AnalysisOutputs(
        input_file=input_file,
        metrics_file=metrics_file,
        event_candidates_file=event_candidates_file,
        events_file=events_file,
        report_file=report_file,
        session_id=session_id,
        row_count=len(df),
        duration_s=calculate_duration_s(df),
        session_distance_m=calculate_session_distance_m(df),
        distance_source=get_primary_distance_source(df),
        metric_count=len(metrics_df),
        candidate_count=len(event_candidates_df),
        event_count=len(events_df),
        rejected_candidate_count=(
            len(event_candidates_df)
            - len(events_df)
        ),
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Analyze a normalized FH6 free-roam drive session and generate "
            "metrics, raw event candidates, validated events, and a text report."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Path to a normalized FH6 CSV. "
            "If omitted, the latest *_normalized.csv file is used."
        ),
    )

    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=DEFAULT_NORMALIZED_DIRECTORY,
        help=(
            "Directory used when --input is omitted. "
            "Default: data/forza/normalized"
        ),
    )

    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_DIRECTORY,
        help=(
            "Directory for analysis CSV outputs. "
            "Default: data/forza/analysis"
        ),
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIRECTORY,
        help=(
            "Directory for drive-session analysis reports. "
            "Default: outputs/reports"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """

    args = parse_arguments()

    input_file = (
        args.input
        if args.input is not None
        else find_latest_normalized_session(
            args.normalized_dir
        )
    )

    input_file = input_file.resolve()

    print("=" * 72)
    print(
        "FH6 Free-Roam Drive-Session Analyzer V2"
    )
    print("=" * 72)

    print(
        f"\nInput normalized CSV:\n{input_file}"
    )

    outputs = run_analysis_pipeline(
        input_file=input_file,
        analysis_directory=args.analysis_dir,
        report_directory=args.report_dir,
    )

    print("\nAnalysis complete.")

    print(
        f"\nSession ID: "
        f"{outputs.session_id}"
    )

    print(
        f"Rows analyzed: "
        f"{outputs.row_count}"
    )

    print(
        f"Duration: "
        f"{outputs.duration_s:.3f} s"
    )

    print(
        f"Session distance: "
        f"{outputs.session_distance_m:.3f} m"
    )

    print(
        f"Distance source: "
        f"{outputs.distance_source}"
    )

    print(
        f"Metrics written: "
        f"{outputs.metric_count}"
    )

    print(
        f"Event candidates detected: "
        f"{outputs.candidate_count}"
    )

    print(
        f"Validated events: "
        f"{outputs.event_count}"
    )

    print(
        f"Rejected candidates: "
        f"{outputs.rejected_candidate_count}"
    )

    print(
        f"\nMetrics CSV:\n"
        f"{outputs.metrics_file}"
    )

    print(
        f"\nEvent candidates CSV:\n"
        f"{outputs.event_candidates_file}"
    )

    print(
        f"\nValidated events CSV:\n"
        f"{outputs.events_file}"
    )

    print(
        f"\nAnalysis report:\n"
        f"{outputs.report_file}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
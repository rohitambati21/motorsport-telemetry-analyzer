"""
forza_adapter.py

Native FH6 telemetry normalization adapter.

Responsibilities:
- Load a native FH6 telemetry CSV created by forza_session_logger.py.
- Convert source-accurate FH6 fields into a clean normalized session schema.
- Preserve important native values needed for future validation.
- Generate robust session-distance channels.
- Select the best available distance source.
- Generate a normalized CSV for downstream analysis.
- Generate a plain-text normalization report.

This module intentionally does not:
- modify the existing synthetic telemetry pipeline
- force free-roam data into fake laps
- run lap comparison
- generate plots
- interpret special gear values
- infer formal lap transitions

The adapter creates a session-level normalized dataset. Formal lap-aware
normalization should be added only after a controlled race/lap capture
validates FH6 lap-transition behavior.
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

DEFAULT_NATIVE_DIRECTORY: Final[Path] = (
    PROJECT_ROOT / "data" / "forza" / "native"
)

DEFAULT_NORMALIZED_DIRECTORY: Final[Path] = (
    PROJECT_ROOT / "data" / "forza" / "normalized"
)

DEFAULT_REPORT_DIRECTORY: Final[Path] = (
    PROJECT_ROOT / "outputs" / "reports"
)

SOURCE_LABEL: Final[str] = "fh6_udp"

STANDARD_GRAVITY_MPS2: Final[float] = 9.80665

UINT32_MODULUS: Final[int] = 2**32
UINT32_HALF_RANGE: Final[int] = 2**31

DISTANCE_MONOTONIC_TOLERANCE_M: Final[float] = 0.01
TIME_MONOTONIC_TOLERANCE_S: Final[float] = 1e-9

# A source must change by at least this much to be considered useful.
# This prevents a constant zero native-distance field from being selected.
MIN_USEFUL_DISTANCE_RANGE_M: Final[float] = 1.0

# Extremely small speed values are still preserved, but this prevents tiny
# numerical noise from being treated as meaningful distance movement.
MIN_SPEED_INTEGRATION_DISTANCE_RANGE_M: Final[float] = 1.0


REQUIRED_NATIVE_COLUMNS: Final[tuple[str, ...]] = (
    "sequence_number",
    "local_receive_time_ns",
    "sender_ip",
    "sender_port",
    "packet_length",
    "is_race_on",
    "timestamp_ms",
    "distance_traveled_m",
    "speed_mps",
    "accel_raw",
    "brake_raw",
    "steer_raw",
    "current_engine_rpm",
    "gear",
    "acceleration_x_mps2",
    "acceleration_z_mps2",
    "position_x_m",
    "position_y_m",
    "position_z_m",
    "power_w",
    "torque_nm",
    "boost_psi",
    "fuel_fraction",
    "lap_number",
    "race_position",
)

OPTIONAL_NATIVE_COLUMNS: Final[tuple[str, ...]] = (
    "current_lap_s",
    "current_race_time_s",
    "best_lap_s",
    "last_lap_s",
    "acceleration_y_mps2",
    "velocity_x_mps",
    "velocity_y_mps",
    "velocity_z_mps",
    "yaw_rad",
    "pitch_rad",
    "roll_rad",
)


NUMERIC_NATIVE_COLUMNS: Final[tuple[str, ...]] = (
    REQUIRED_NATIVE_COLUMNS
    + OPTIONAL_NATIVE_COLUMNS
)

NON_NUMERIC_NATIVE_COLUMNS: Final[set[str]] = {
    "sender_ip",
}


NORMALIZED_COLUMN_ORDER: Final[tuple[str, ...]] = (
    "source",
    "session_id",
    "sample_index",
    "sequence_number",

    "time_s",
    "timestamp_ms",
    "timestamp_unwrapped_ms",
    "local_receive_time_ns",
    "local_receive_time_s",
    "sender_ip",
    "sender_port",
    "packet_length",

    "session_distance_m",
    "distance_source",
    "session_distance_native_m",
    "session_distance_position_m",
    "session_distance_speed_integrated_m",
    "distance_traveled_total_m",

    "speed_mps",
    "speed_kph",

    "throttle_raw",
    "brake_raw",
    "steer_raw",
    "throttle_pct",
    "brake_pct",
    "steering_input_norm",

    "rpm",
    "gear_raw",

    "acceleration_x_mps2",
    "acceleration_y_mps2",
    "acceleration_z_mps2",
    "lat_g",
    "lon_g",

    "position_x_m",
    "position_y_m",
    "position_z_m",

    "power_w",
    "power_kw",
    "torque_nm",
    "boost_psi",
    "fuel_fraction",

    "lap_number_native",
    "race_position_native",
    "current_lap_s",
    "current_race_time_s",
    "best_lap_s",
    "last_lap_s",
    "is_race_on",
)


@dataclass(frozen=True, slots=True)
class DistanceSelection:
    """
    Summary of all distance channels and the selected distance source.
    """

    selected_source: str
    selection_reason: str

    native_distance_range_m: float
    position_distance_range_m: float
    speed_integrated_distance_range_m: float
    selected_distance_range_m: float

    native_distance_usable: bool
    position_distance_usable: bool
    speed_integrated_distance_usable: bool


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """
    Normalized dataframe plus distance-source metadata.
    """

    normalized_df: pd.DataFrame
    distance_selection: DistanceSelection


@dataclass(frozen=True, slots=True)
class AdapterOutputs:
    """
    Paths and summary values produced by one adapter run.
    """

    input_file: Path
    normalized_file: Path
    report_file: Path
    session_id: str
    row_count: int
    duration_s: float
    session_distance_m: float
    distance_source: str


def find_latest_native_session(
    native_directory: Path,
) -> Path:
    """
    Find the most recently modified FH6 native session CSV.
    """

    if not native_directory.exists():
        raise FileNotFoundError(
            f"Native telemetry directory does not exist: {native_directory}"
        )

    candidates = sorted(
        native_directory.glob("fh6_session_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            "No native FH6 session CSV files were found in "
            f"{native_directory}"
        )

    return candidates[0]


def derive_session_id(
    input_file: Path,
    requested_session_id: str | None,
) -> str:
    """
    Create a stable session ID from either a user value or the input filename.
    """

    if requested_session_id:
        cleaned = requested_session_id.strip()

        if not cleaned:
            raise ValueError(
                "session_id cannot be empty."
            )

        return cleaned

    return input_file.stem


def load_native_session(
    input_file: Path,
) -> pd.DataFrame:
    """
    Load a native FH6 session CSV and validate required columns.
    """

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input native session file does not exist: {input_file}"
        )

    native_df = pd.read_csv(
        input_file
    )

    if native_df.empty:
        raise ValueError(
            f"Input native session file is empty: {input_file}"
        )

    missing_columns = sorted(
        set(REQUIRED_NATIVE_COLUMNS) - set(native_df.columns)
    )

    if missing_columns:
        missing_text = "\n".join(
            f"  - {column}"
            for column in missing_columns
        )

        raise ValueError(
            "Input native session CSV is missing required columns:\n"
            f"{missing_text}"
        )

    return native_df


def coerce_numeric_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """
    Convert expected numeric columns to numeric dtype.

    Invalid values become NaN so the report can expose data-quality issues.
    """

    cleaned_df = df.copy()

    for column in columns:
        if (
            column in cleaned_df.columns
            and column not in NON_NUMERIC_NATIVE_COLUMNS
        ):
            cleaned_df[column] = pd.to_numeric(
                cleaned_df[column],
                errors="coerce",
            )

    return cleaned_df


def unwrap_uint32_timestamp_ms(
    timestamp_ms: pd.Series,
) -> pd.Series:
    """
    Unwrap a uint32 millisecond counter into a monotonic millisecond series.

    FH6 timestamps are stored as an unsigned 32-bit millisecond counter.
    This helper preserves normal sessions and also makes the adapter ready
    for long-running captures where wraparound could eventually occur.
    """

    raw_values = pd.to_numeric(
        timestamp_ms,
        errors="coerce",
    ).to_numpy(dtype="float64")

    unwrapped_values = np.full(
        shape=raw_values.shape,
        fill_value=np.nan,
        dtype="float64",
    )

    offset = 0.0
    previous_raw_value: float | None = None

    for index, raw_value in enumerate(raw_values):
        if math.isnan(raw_value):
            continue

        if previous_raw_value is not None:
            difference = raw_value - previous_raw_value

            if difference < -UINT32_HALF_RANGE:
                offset += UINT32_MODULUS

            elif difference > UINT32_HALF_RANGE:
                offset -= UINT32_MODULUS

        unwrapped_values[index] = raw_value + offset
        previous_raw_value = raw_value

    return pd.Series(
        unwrapped_values,
        index=timestamp_ms.index,
        name="timestamp_unwrapped_ms",
    )


def first_valid_value(
    series: pd.Series,
    field_name: str,
) -> float:
    """
    Return the first non-missing numeric value from a series.
    """

    valid_values = series.dropna()

    if valid_values.empty:
        raise ValueError(
            f"Column {field_name} does not contain any valid numeric values."
        )

    return float(
        valid_values.iloc[0]
    )


def normalize_raw_input_to_percent(
    raw_input: pd.Series,
) -> pd.Series:
    """
    Convert FH6 0-255 raw input scale to 0-100 percent.
    """

    return (
        pd.to_numeric(
            raw_input,
            errors="coerce",
        )
        .clip(lower=0, upper=255)
        / 255.0
        * 100.0
    )


def normalize_steering_input(
    steer_raw: pd.Series,
) -> pd.Series:
    """
    Convert FH6 signed steering input to approximately -1 to +1.

    This is source-accurate input normalization. It is not a conversion to
    steering angle degrees.
    """

    return (
        pd.to_numeric(
            steer_raw,
            errors="coerce",
        )
        .clip(lower=-127, upper=127)
        / 127.0
    )


def copy_column_or_nan(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Return a source column when present; otherwise return a NaN-filled series.
    """

    if column in df.columns:
        return df[column]

    return pd.Series(
        np.nan,
        index=df.index,
        name=column,
    )


def finite_min(
    series: pd.Series,
) -> float:
    """
    Return finite min or NaN.
    """

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric_series = numeric_series[
        np.isfinite(numeric_series)
    ]

    if numeric_series.empty:
        return math.nan

    return float(
        numeric_series.min()
    )


def finite_max(
    series: pd.Series,
) -> float:
    """
    Return finite max or NaN.
    """

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric_series = numeric_series[
        np.isfinite(numeric_series)
    ]

    if numeric_series.empty:
        return math.nan

    return float(
        numeric_series.max()
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
    Format a numeric series range.
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


def is_monotonic_non_decreasing(
    series: pd.Series,
    tolerance: float = 0.0,
) -> bool:
    """
    Return True when a numeric series is monotonic non-decreasing.
    """

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(numeric_series) < 2:
        return True

    differences = numeric_series.diff().dropna()

    return bool(
        (differences >= -tolerance).all()
    )


def compute_native_session_distance(
    native_distance_total_m: pd.Series,
) -> pd.Series:
    """
    Compute session-relative native distance from FH6 distance_traveled_m.

    This preserves the original native distance strategy, but it is no longer
    the only possible source for session_distance_m.
    """

    numeric_distance = pd.to_numeric(
        native_distance_total_m,
        errors="coerce",
    )

    first_distance_m = first_valid_value(
        numeric_distance,
        "distance_traveled_m",
    )

    return (
        numeric_distance
        - first_distance_m
    )


def compute_position_path_distance(
    position_x_m: pd.Series,
    position_z_m: pd.Series,
) -> pd.Series:
    """
    Compute cumulative horizontal path distance from FH6 world position.

    FH6 exposes X/Y/Z world position. For driving distance, horizontal
    ground-path movement is usually more useful than 3D path length, so this
    uses X and Z only. Y is preserved elsewhere as elevation/world-position
    context.
    """

    x_values = pd.to_numeric(
        position_x_m,
        errors="coerce",
    ).to_numpy(dtype="float64")

    z_values = pd.to_numeric(
        position_z_m,
        errors="coerce",
    ).to_numpy(dtype="float64")

    increments = np.zeros_like(
        x_values,
        dtype="float64",
    )

    if len(x_values) < 2:
        return pd.Series(
            increments,
            index=position_x_m.index,
            name="session_distance_position_m",
        )

    previous_x = x_values[:-1]
    current_x = x_values[1:]

    previous_z = z_values[:-1]
    current_z = z_values[1:]

    valid_steps = (
        np.isfinite(previous_x)
        & np.isfinite(current_x)
        & np.isfinite(previous_z)
        & np.isfinite(current_z)
    )

    dx = current_x - previous_x
    dz = current_z - previous_z

    step_distances = np.zeros(
        len(x_values) - 1,
        dtype="float64",
    )

    step_distances[valid_steps] = np.sqrt(
        dx[valid_steps] ** 2
        + dz[valid_steps] ** 2
    )

    increments[1:] = step_distances

    cumulative_distance = np.cumsum(
        increments
    )

    return pd.Series(
        cumulative_distance,
        index=position_x_m.index,
        name="session_distance_position_m",
    )


def compute_speed_integrated_distance(
    time_s: pd.Series,
    speed_mps: pd.Series,
) -> pd.Series:
    """
    Compute cumulative distance by trapezoidal integration of speed over time.

    This is a robust fallback because speed_mps and time_s have already
    validated well. It is useful when native distance is unavailable and
    world-position data is missing or unreliable.
    """

    time_values = pd.to_numeric(
        time_s,
        errors="coerce",
    ).to_numpy(dtype="float64")

    speed_values = pd.to_numeric(
        speed_mps,
        errors="coerce",
    ).to_numpy(dtype="float64")

    increments = np.zeros_like(
        time_values,
        dtype="float64",
    )

    if len(time_values) < 2:
        return pd.Series(
            increments,
            index=time_s.index,
            name="session_distance_speed_integrated_m",
        )

    previous_time = time_values[:-1]
    current_time = time_values[1:]

    previous_speed = speed_values[:-1]
    current_speed = speed_values[1:]

    dt = current_time - previous_time

    valid_steps = (
        np.isfinite(previous_time)
        & np.isfinite(current_time)
        & np.isfinite(previous_speed)
        & np.isfinite(current_speed)
        & (dt >= 0.0)
    )

    average_speed = (
        previous_speed
        + current_speed
    ) / 2.0

    step_distances = np.zeros(
        len(time_values) - 1,
        dtype="float64",
    )

    step_distances[valid_steps] = (
        average_speed[valid_steps]
        * dt[valid_steps]
    )

    # Distance should not move backwards. Negative speed is not expected, but
    # clipping protects the fallback from unusual values.
    step_distances = np.maximum(
        step_distances,
        0.0,
    )

    increments[1:] = step_distances

    cumulative_distance = np.cumsum(
        increments
    )

    return pd.Series(
        cumulative_distance,
        index=time_s.index,
        name="session_distance_speed_integrated_m",
    )


def source_has_useful_distance(
    distance_series: pd.Series,
    minimum_range_m: float,
    require_monotonic: bool = True,
) -> bool:
    """
    Determine whether a candidate distance source is useful enough to select.
    """

    distance_range_m = finite_range(
        distance_series
    )

    if (
        math.isnan(distance_range_m)
        or distance_range_m < minimum_range_m
    ):
        return False

    if require_monotonic and not is_monotonic_non_decreasing(
        distance_series,
        DISTANCE_MONOTONIC_TOLERANCE_M,
    ):
        return False

    return True


def select_session_distance(
    native_distance_m: pd.Series,
    position_distance_m: pd.Series,
    speed_integrated_distance_m: pd.Series,
) -> tuple[pd.Series, DistanceSelection]:
    """
    Select the best available distance source.

    Selection priority:
    1. Native FH6 distance if it changes meaningfully and is monotonic.
    2. Horizontal world-position path distance if it changes meaningfully.
    3. Speed-integrated distance if it changes meaningfully.
    4. Zero-filled unavailable distance if all sources fail.
    """

    native_range_m = finite_range(
        native_distance_m
    )

    position_range_m = finite_range(
        position_distance_m
    )

    speed_integrated_range_m = finite_range(
        speed_integrated_distance_m
    )

    native_usable = source_has_useful_distance(
        native_distance_m,
        MIN_USEFUL_DISTANCE_RANGE_M,
        require_monotonic=True,
    )

    position_usable = source_has_useful_distance(
        position_distance_m,
        MIN_USEFUL_DISTANCE_RANGE_M,
        require_monotonic=True,
    )

    speed_integrated_usable = source_has_useful_distance(
        speed_integrated_distance_m,
        MIN_SPEED_INTEGRATION_DISTANCE_RANGE_M,
        require_monotonic=True,
    )

    if native_usable:
        selected = native_distance_m.copy()
        selected_source = "native_distance"
        selection_reason = (
            "Native FH6 distance_traveled_m changed meaningfully "
            "and was monotonic, so it was selected."
        )

    elif position_usable:
        selected = position_distance_m.copy()
        selected_source = "position_path"
        selection_reason = (
            "Native FH6 distance_traveled_m was not useful, so horizontal "
            "world-position path distance was selected."
        )

    elif speed_integrated_usable:
        selected = speed_integrated_distance_m.copy()
        selected_source = "speed_integration"
        selection_reason = (
            "Native distance and position-path distance were not useful, "
            "so trapezoidal speed integration was selected."
        )

    else:
        selected = pd.Series(
            np.zeros(
                len(native_distance_m),
                dtype="float64",
            ),
            index=native_distance_m.index,
            name="session_distance_m",
        )

        selected_source = "unavailable"
        selection_reason = (
            "No distance source changed meaningfully. session_distance_m "
            "was filled with zeros."
        )

    selected = selected.astype(
        "float64"
    )

    selected.name = "session_distance_m"

    selected_range_m = finite_range(
        selected
    )

    distance_selection = DistanceSelection(
        selected_source=selected_source,
        selection_reason=selection_reason,
        native_distance_range_m=(
            native_range_m
            if not math.isnan(native_range_m)
            else 0.0
        ),
        position_distance_range_m=(
            position_range_m
            if not math.isnan(position_range_m)
            else 0.0
        ),
        speed_integrated_distance_range_m=(
            speed_integrated_range_m
            if not math.isnan(speed_integrated_range_m)
            else 0.0
        ),
        selected_distance_range_m=(
            selected_range_m
            if not math.isnan(selected_range_m)
            else 0.0
        ),
        native_distance_usable=native_usable,
        position_distance_usable=position_usable,
        speed_integrated_distance_usable=speed_integrated_usable,
    )

    return selected, distance_selection


def normalize_native_session(
    native_df: pd.DataFrame,
    session_id: str,
) -> NormalizationResult:
    """
    Convert native FH6 telemetry into the normalized session schema.
    """

    native_df = coerce_numeric_columns(
        native_df,
        NUMERIC_NATIVE_COLUMNS,
    )

    timestamp_unwrapped_ms = unwrap_uint32_timestamp_ms(
        native_df["timestamp_ms"]
    )

    first_timestamp_ms = first_valid_value(
        timestamp_unwrapped_ms,
        "timestamp_ms",
    )

    first_local_receive_time_ns = first_valid_value(
        native_df["local_receive_time_ns"],
        "local_receive_time_ns",
    )

    time_s = (
        timestamp_unwrapped_ms
        - first_timestamp_ms
    ) / 1000.0

    native_session_distance_m = compute_native_session_distance(
        native_df["distance_traveled_m"]
    )

    position_path_distance_m = compute_position_path_distance(
        native_df["position_x_m"],
        native_df["position_z_m"],
    )

    speed_integrated_distance_m = compute_speed_integrated_distance(
        time_s,
        native_df["speed_mps"],
    )

    selected_session_distance_m, distance_selection = select_session_distance(
        native_session_distance_m,
        position_path_distance_m,
        speed_integrated_distance_m,
    )

    normalized_df = pd.DataFrame(
        index=native_df.index
    )

    normalized_df["source"] = SOURCE_LABEL
    normalized_df["session_id"] = session_id
    normalized_df["sample_index"] = np.arange(
        len(native_df),
        dtype=int,
    )

    normalized_df["sequence_number"] = native_df[
        "sequence_number"
    ]

    normalized_df["time_s"] = time_s

    normalized_df["timestamp_ms"] = native_df[
        "timestamp_ms"
    ]

    normalized_df["timestamp_unwrapped_ms"] = (
        timestamp_unwrapped_ms
    )

    normalized_df["local_receive_time_ns"] = native_df[
        "local_receive_time_ns"
    ]

    normalized_df["local_receive_time_s"] = (
        native_df["local_receive_time_ns"]
        - first_local_receive_time_ns
    ) / 1_000_000_000.0

    normalized_df["sender_ip"] = native_df[
        "sender_ip"
    ]

    normalized_df["sender_port"] = native_df[
        "sender_port"
    ]

    normalized_df["packet_length"] = native_df[
        "packet_length"
    ]

    normalized_df["session_distance_m"] = (
        selected_session_distance_m
    )

    normalized_df["distance_source"] = (
        distance_selection.selected_source
    )

    normalized_df["session_distance_native_m"] = (
        native_session_distance_m
    )

    normalized_df["session_distance_position_m"] = (
        position_path_distance_m
    )

    normalized_df["session_distance_speed_integrated_m"] = (
        speed_integrated_distance_m
    )

    normalized_df["distance_traveled_total_m"] = native_df[
        "distance_traveled_m"
    ]

    normalized_df["speed_mps"] = native_df[
        "speed_mps"
    ]

    normalized_df["speed_kph"] = (
        native_df["speed_mps"] * 3.6
    )

    normalized_df["throttle_raw"] = native_df[
        "accel_raw"
    ]

    normalized_df["brake_raw"] = native_df[
        "brake_raw"
    ]

    normalized_df["steer_raw"] = native_df[
        "steer_raw"
    ]

    normalized_df["throttle_pct"] = normalize_raw_input_to_percent(
        native_df["accel_raw"]
    )

    normalized_df["brake_pct"] = normalize_raw_input_to_percent(
        native_df["brake_raw"]
    )

    normalized_df["steering_input_norm"] = normalize_steering_input(
        native_df["steer_raw"]
    )

    normalized_df["rpm"] = native_df[
        "current_engine_rpm"
    ]

    normalized_df["gear_raw"] = native_df[
        "gear"
    ]

    normalized_df["acceleration_x_mps2"] = native_df[
        "acceleration_x_mps2"
    ]

    normalized_df["acceleration_y_mps2"] = copy_column_or_nan(
        native_df,
        "acceleration_y_mps2",
    )

    normalized_df["acceleration_z_mps2"] = native_df[
        "acceleration_z_mps2"
    ]

    normalized_df["lat_g"] = (
        native_df["acceleration_x_mps2"]
        / STANDARD_GRAVITY_MPS2
    )

    normalized_df["lon_g"] = (
        native_df["acceleration_z_mps2"]
        / STANDARD_GRAVITY_MPS2
    )

    normalized_df["position_x_m"] = native_df[
        "position_x_m"
    ]

    normalized_df["position_y_m"] = native_df[
        "position_y_m"
    ]

    normalized_df["position_z_m"] = native_df[
        "position_z_m"
    ]

    normalized_df["power_w"] = native_df[
        "power_w"
    ]

    normalized_df["power_kw"] = (
        native_df["power_w"] / 1000.0
    )

    normalized_df["torque_nm"] = native_df[
        "torque_nm"
    ]

    normalized_df["boost_psi"] = native_df[
        "boost_psi"
    ]

    normalized_df["fuel_fraction"] = native_df[
        "fuel_fraction"
    ]

    normalized_df["lap_number_native"] = native_df[
        "lap_number"
    ]

    normalized_df["race_position_native"] = native_df[
        "race_position"
    ]

    normalized_df["current_lap_s"] = copy_column_or_nan(
        native_df,
        "current_lap_s",
    )

    normalized_df["current_race_time_s"] = copy_column_or_nan(
        native_df,
        "current_race_time_s",
    )

    normalized_df["best_lap_s"] = copy_column_or_nan(
        native_df,
        "best_lap_s",
    )

    normalized_df["last_lap_s"] = copy_column_or_nan(
        native_df,
        "last_lap_s",
    )

    normalized_df["is_race_on"] = native_df[
        "is_race_on"
    ]

    normalized_df = normalized_df.loc[
        :,
        NORMALIZED_COLUMN_ORDER,
    ]

    return NormalizationResult(
        normalized_df=normalized_df,
        distance_selection=distance_selection,
    )


def unique_values_text(
    series: pd.Series,
    maximum_values: int = 30,
) -> str:
    """
    Format sorted unique values for report output.
    """

    valid_values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if valid_values.empty:
        return "not available"

    unique_values = sorted(
        valid_values.astype(int).unique().tolist()
    )

    if len(unique_values) > maximum_values:
        displayed_values = unique_values[
            :maximum_values
        ]

        return (
            f"{displayed_values} "
            f"... ({len(unique_values)} unique values)"
        )

    return str(
        unique_values
    )


def count_missing_values(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> dict[str, int]:
    """
    Count missing values for selected columns.
    """

    missing_counts: dict[str, int] = {}

    for column in columns:
        if column in df.columns:
            missing_counts[column] = int(
                df[column].isna().sum()
            )
        else:
            missing_counts[column] = -1

    return missing_counts


def build_quality_warnings(
    normalized_df: pd.DataFrame,
    distance_selection: DistanceSelection,
) -> list[str]:
    """
    Build adapter-level warnings without rejecting the file.
    """

    warnings: list[str] = []

    if not is_monotonic_non_decreasing(
        normalized_df["time_s"],
        TIME_MONOTONIC_TOLERANCE_S,
    ):
        warnings.append(
            "time_s is not monotonic non-decreasing."
        )

    if not is_monotonic_non_decreasing(
        normalized_df["session_distance_m"],
        DISTANCE_MONOTONIC_TOLERANCE_M,
    ):
        warnings.append(
            "session_distance_m is not monotonic non-decreasing."
        )

    if distance_selection.selected_source == "unavailable":
        warnings.append(
            "No usable distance source was found. session_distance_m is zero-filled."
        )

    if finite_min(
        normalized_df["speed_mps"]
    ) < -1e-6:
        warnings.append(
            "speed_mps contains negative values."
        )

    if finite_min(
        normalized_df["throttle_pct"]
    ) < -1e-6 or finite_max(
        normalized_df["throttle_pct"]
    ) > 100.000001:
        warnings.append(
            "throttle_pct is outside the expected 0-100 range."
        )

    if finite_min(
        normalized_df["brake_pct"]
    ) < -1e-6 or finite_max(
        normalized_df["brake_pct"]
    ) > 100.000001:
        warnings.append(
            "brake_pct is outside the expected 0-100 range."
        )

    if finite_min(
        normalized_df["steering_input_norm"]
    ) < -1.000001 or finite_max(
        normalized_df["steering_input_norm"]
    ) > 1.000001:
        warnings.append(
            "steering_input_norm is outside the expected -1 to +1 range."
        )

    if finite_max(
        normalized_df["packet_length"]
    ) != 324 or finite_min(
        normalized_df["packet_length"]
    ) != 324:
        warnings.append(
            "packet_length contains values other than 324 bytes."
        )

    return warnings


def calculate_duration_s(
    normalized_df: pd.DataFrame,
) -> float:
    """
    Calculate normalized session duration from time_s.
    """

    if normalized_df.empty:
        return 0.0

    return finite_max(
        normalized_df["time_s"]
    ) - finite_min(
        normalized_df["time_s"]
    )


def calculate_sample_rate_hz(
    row_count: int,
    duration_s: float,
) -> float:
    """
    Estimate sample rate from row count and duration.
    """

    if row_count < 2 or duration_s <= 0.0:
        return 0.0

    return (
        row_count - 1
    ) / duration_s


def calculate_session_distance_m(
    normalized_df: pd.DataFrame,
) -> float:
    """
    Calculate total selected session-relative distance covered.
    """

    if normalized_df.empty:
        return 0.0

    return finite_range(
        normalized_df["session_distance_m"]
    )


def build_normalization_report(
    input_file: Path,
    normalized_file: Path,
    report_file: Path,
    session_id: str,
    native_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
    distance_selection: DistanceSelection,
) -> str:
    """
    Build a plain-text normalization report.
    """

    row_count = len(
        normalized_df
    )

    duration_s = calculate_duration_s(
        normalized_df
    )

    sample_rate_hz = calculate_sample_rate_hz(
        row_count,
        duration_s,
    )

    session_distance_m = calculate_session_distance_m(
        normalized_df
    )

    time_monotonic = is_monotonic_non_decreasing(
        normalized_df["time_s"],
        TIME_MONOTONIC_TOLERANCE_S,
    )

    selected_distance_monotonic = is_monotonic_non_decreasing(
        normalized_df["session_distance_m"],
        DISTANCE_MONOTONIC_TOLERANCE_M,
    )

    native_distance_monotonic = is_monotonic_non_decreasing(
        normalized_df["session_distance_native_m"],
        DISTANCE_MONOTONIC_TOLERANCE_M,
    )

    position_distance_monotonic = is_monotonic_non_decreasing(
        normalized_df["session_distance_position_m"],
        DISTANCE_MONOTONIC_TOLERANCE_M,
    )

    speed_distance_monotonic = is_monotonic_non_decreasing(
        normalized_df["session_distance_speed_integrated_m"],
        DISTANCE_MONOTONIC_TOLERANCE_M,
    )

    local_receive_duration_s = (
        finite_max(
            normalized_df["local_receive_time_s"]
        )
        - finite_min(
            normalized_df["local_receive_time_s"]
        )
    )

    local_receive_rate_hz = calculate_sample_rate_hz(
        row_count,
        local_receive_duration_s,
    )

    missing_counts = count_missing_values(
        normalized_df,
        NORMALIZED_COLUMN_ORDER,
    )

    quality_warnings = build_quality_warnings(
        normalized_df,
        distance_selection,
    )

    missing_count_lines = [
        f"  {column}: {count}"
        for column, count in missing_counts.items()
        if count != 0
    ]

    if not missing_count_lines:
        missing_count_lines = [
            "  none"
        ]

    warning_lines = [
        f"  - {warning}"
        for warning in quality_warnings
    ]

    if not warning_lines:
        warning_lines = [
            "  none"
        ]

    lines: list[str] = []

    lines.extend(
        [
            "FH6 NORMALIZATION REPORT",
            "=" * 72,
            "",
            "FILES",
            "-" * 72,
            f"Input native CSV: {input_file}",
            f"Normalized CSV: {normalized_file}",
            f"Report file: {report_file}",
            "",
            "SESSION",
            "-" * 72,
            f"Session ID: {session_id}",
            f"Source label: {SOURCE_LABEL}",
            f"Rows read from native CSV: {len(native_df)}",
            f"Rows written to normalized CSV: {row_count}",
            "",
            "TIME",
            "-" * 72,
            f"Duration from game timestamp: {duration_s:.3f} s",
            f"Estimated game-timestamp sample rate: {sample_rate_hz:.3f} Hz",
            f"Duration from local receive time: {local_receive_duration_s:.3f} s",
            f"Estimated local receive sample rate: {local_receive_rate_hz:.3f} Hz",
            f"time_s monotonic non-decreasing: {time_monotonic}",
            "",
            "DISTANCE",
            "-" * 72,
            f"Selected distance source: {distance_selection.selected_source}",
            f"Selection reason: {distance_selection.selection_reason}",
            "",
            f"Native distance range: {distance_selection.native_distance_range_m:.3f} m",
            f"Native distance usable: {distance_selection.native_distance_usable}",
            f"Native distance monotonic non-decreasing: {native_distance_monotonic}",
            "",
            f"Position-path distance range: {distance_selection.position_distance_range_m:.3f} m",
            f"Position-path distance usable: {distance_selection.position_distance_usable}",
            f"Position-path distance monotonic non-decreasing: {position_distance_monotonic}",
            "",
            f"Speed-integrated distance range: {distance_selection.speed_integrated_distance_range_m:.3f} m",
            f"Speed-integrated distance usable: {distance_selection.speed_integrated_distance_usable}",
            f"Speed-integrated distance monotonic non-decreasing: {speed_distance_monotonic}",
            "",
            f"Selected session-relative distance: {session_distance_m:.3f} m",
            f"Selected session-relative distance: {session_distance_m / 1000.0:.3f} km",
            f"session_distance_m monotonic non-decreasing: {selected_distance_monotonic}",
            "",
            "SIGNAL RANGES",
            "-" * 72,
            f"Speed [m/s]: {finite_range_text(normalized_df['speed_mps'], 3)}",
            f"Speed [km/h]: {finite_range_text(normalized_df['speed_kph'], 2)}",
            f"RPM: {finite_range_text(normalized_df['rpm'], 0)}",
            f"Throttle raw: {finite_range_text(normalized_df['throttle_raw'], 0)}",
            f"Throttle [%]: {finite_range_text(normalized_df['throttle_pct'], 2)}",
            f"Brake raw: {finite_range_text(normalized_df['brake_raw'], 0)}",
            f"Brake [%]: {finite_range_text(normalized_df['brake_pct'], 2)}",
            f"Steer raw: {finite_range_text(normalized_df['steer_raw'], 0)}",
            f"Steering input [-1, 1]: {finite_range_text(normalized_df['steering_input_norm'], 3)}",
            f"Lateral acceleration estimate [g]: {finite_range_text(normalized_df['lat_g'], 4)}",
            f"Longitudinal acceleration estimate [g]: {finite_range_text(normalized_df['lon_g'], 4)}",
            f"Power [kW]: {finite_range_text(normalized_df['power_kw'], 3)}",
            f"Torque [Nm]: {finite_range_text(normalized_df['torque_nm'], 3)}",
            f"Boost [psi]: {finite_range_text(normalized_df['boost_psi'], 3)}",
            f"Fuel fraction: {finite_range_text(normalized_df['fuel_fraction'], 3)}",
            "",
            "OBSERVED DISCRETE VALUES",
            "-" * 72,
            f"Gear raw values: {unique_values_text(normalized_df['gear_raw'])}",
            f"Native lap numbers: {unique_values_text(normalized_df['lap_number_native'])}",
            f"Native race positions: {unique_values_text(normalized_df['race_position_native'])}",
            f"is_race_on values: {unique_values_text(normalized_df['is_race_on'])}",
            f"Packet lengths: {unique_values_text(normalized_df['packet_length'])}",
            f"Distance source values: {sorted(normalized_df['distance_source'].dropna().unique().tolist())}",
            "",
            "MISSING VALUES",
            "-" * 72,
            *missing_count_lines,
            "",
            "QUALITY WARNINGS",
            "-" * 72,
            *warning_lines,
            "",
            "ADAPTER DECISIONS",
            "-" * 72,
            "This output is a session-level normalized dataset.",
            "It intentionally uses session_distance_m rather than lap_distance_m.",
            "The adapter does not create fake laps for free-roam data.",
            "gear_raw is preserved without interpreting special gear values.",
            "steering_input_norm is normalized from FH6 steering input and is not a physical steering angle in degrees.",
            "lat_g and lon_g are calculated from FH6 local acceleration channels; sign convention still requires controlled validation.",
            "session_distance_m is now selected from the best available source using native distance, position-path distance, then speed integration.",
            "",
            "NEXT STEP",
            "-" * 72,
            "Use this normalized file for early FH6 drive-session analysis or plotting.",
            "Formal lap-analysis integration should wait until a controlled race/lap capture validates lap-number and lap-distance behavior.",
            "",
            "Report generated automatically by forza_adapter.py.",
        ]
    )

    return "\n".join(
        lines
    )


def write_normalized_csv(
    normalized_df: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    Save normalized telemetry to CSV.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized_df.to_csv(
        output_file,
        index=False,
    )


def write_report(
    report_text: str,
    report_file: Path,
) -> None:
    """
    Save report text to disk.
    """

    report_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_file.write_text(
        report_text,
        encoding="utf-8",
    )


def build_output_paths(
    session_id: str,
    normalized_directory: Path,
    report_directory: Path,
) -> tuple[Path, Path]:
    """
    Build normalized CSV and report output paths.
    """

    normalized_file = (
        normalized_directory
        / f"{session_id}_normalized.csv"
    )

    report_file = (
        report_directory
        / f"{session_id}_normalization_report.txt"
    )

    return (
        normalized_file,
        report_file,
    )


def run_adapter(
    input_file: Path,
    session_id: str,
    normalized_directory: Path,
    report_directory: Path,
) -> AdapterOutputs:
    """
    Execute the complete native-to-normalized conversion.
    """

    native_df = load_native_session(
        input_file
    )

    normalization_result = normalize_native_session(
        native_df,
        session_id,
    )

    normalized_df = normalization_result.normalized_df
    distance_selection = normalization_result.distance_selection

    normalized_file, report_file = build_output_paths(
        session_id=session_id,
        normalized_directory=normalized_directory,
        report_directory=report_directory,
    )

    write_normalized_csv(
        normalized_df,
        normalized_file,
    )

    report_text = build_normalization_report(
        input_file=input_file,
        normalized_file=normalized_file,
        report_file=report_file,
        session_id=session_id,
        native_df=native_df,
        normalized_df=normalized_df,
        distance_selection=distance_selection,
    )

    write_report(
        report_text,
        report_file,
    )

    return AdapterOutputs(
        input_file=input_file,
        normalized_file=normalized_file,
        report_file=report_file,
        session_id=session_id,
        row_count=len(normalized_df),
        duration_s=calculate_duration_s(normalized_df),
        session_distance_m=calculate_session_distance_m(normalized_df),
        distance_source=distance_selection.selected_source,
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Normalize a native FH6 UDP telemetry CSV into a clean "
            "session-level analysis dataset."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Path to a native FH6 CSV. "
            "If omitted, the latest file in data/forza/native is used."
        ),
    )

    parser.add_argument(
        "--native-dir",
        type=Path,
        default=DEFAULT_NATIVE_DIRECTORY,
        help=(
            "Directory used when --input is omitted. "
            "Default: data/forza/native"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_NORMALIZED_DIRECTORY,
        help=(
            "Directory for normalized telemetry CSV output. "
            "Default: data/forza/normalized"
        ),
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIRECTORY,
        help=(
            "Directory for normalization reports. "
            "Default: outputs/reports"
        ),
    )

    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help=(
            "Optional session ID. "
            "Default: input filename stem."
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
        else find_latest_native_session(
            args.native_dir
        )
    )

    input_file = input_file.resolve()

    session_id = derive_session_id(
        input_file,
        args.session_id,
    )

    print("=" * 72)
    print(
        "FH6 Native-to-Normalized Session Adapter"
    )
    print("=" * 72)

    print(
        f"\nInput native CSV:\n{input_file}"
    )

    print(
        f"\nSession ID:\n{session_id}"
    )

    outputs = run_adapter(
        input_file=input_file,
        session_id=session_id,
        normalized_directory=args.output_dir,
        report_directory=args.report_dir,
    )

    print("\nAdapter complete.")

    print(
        f"\nRows normalized: "
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
        f"\nNormalized CSV:\n"
        f"{outputs.normalized_file}"
    )

    print(
        f"\nNormalization report:\n"
        f"{outputs.report_file}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
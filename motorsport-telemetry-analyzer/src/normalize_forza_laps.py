"""
normalize_forza_laps.py

Normalize validated extracted FH6 laps onto one common distance grid.

Development role
----------------
Raw telemetry laps contain different numbers of samples because sample timing,
speed, and lap duration vary. Direct row-by-row comparison would therefore
compare different physical locations.

This script converts each validated extracted lap onto the same distance axis:

    extracted laps
    -> common usable distance
    -> 400-point distance grid
    -> interpolated canonical telemetry

Interpolation policy
--------------------
Continuous channels:
    linear interpolation

Discrete gear channel:
    nearest-neighbor interpolation

No telemetry smoothing is performed.

Distance samples must remain non-decreasing. Duplicate distance samples are
collapsed only to make the independent interpolation coordinate unique; this
is coordinate sanitization rather than signal smoothing.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

DEFAULT_EXTRACTED_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "data"
    / "forza"
    / "laps"
)

DEFAULT_NORMALIZED_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "data"
    / "forza"
    / "lap_normalized"
)

DEFAULT_REPORT_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

CONTINUOUS_CHANNELS: Final[tuple[str, ...]] = (
    "lap_elapsed_time_s",
    "speed_mps",
    "speed_kph",
    "throttle_pct",
    "brake_pct",
    "steer_normalized",
    "longitudinal_g",
    "lateral_g",
)

REQUIRED_EXTRACTED_COLUMNS: Final[tuple[str, ...]] = (
    "session_id",
    "capture_id",
    "lap_id",
    "source_lap_number",
    "lap_elapsed_time_s",
    "lap_distance_m",
    "speed_mps",
    "speed_kph",
    "throttle_pct",
    "brake_pct",
    "steer_normalized",
    "gear",
    "longitudinal_g",
    "lateral_g",
)

NORMALIZED_SCHEMA: Final[tuple[str, ...]] = (
    "session_id",
    "capture_id",
    "lap_id",
    "lap_number",
    "distance_m",
    "elapsed_time_s",
    "speed_mps",
    "speed_kph",
    "throttle_pct",
    "brake_pct",
    "steer_normalized",
    "gear",
    "longitudinal_g",
    "lateral_g",
)


@dataclass(frozen=True, slots=True)
class PreparedLap:
    """One extracted lap prepared for interpolation."""

    lap_index: int
    lap_id: str
    lap_number: str
    session_id: str
    capture_id: str

    source_file: Path
    source_rows: int

    source_distance_m: float
    source_duration_s: float

    duplicate_distance_rows_removed: int

    distance: np.ndarray
    dataframe: pd.DataFrame


@dataclass(frozen=True, slots=True)
class NormalizedArtifact:
    """One normalized lap artifact."""

    lap_index: int
    lap_id: str
    output_file: Path

    source_distance_m: float
    source_duration_s: float

    common_distance_m: float

    normalized_final_elapsed_s: float
    source_elapsed_at_common_distance_s: float
    timing_error_s: float

    rows: int
    validation_status: str


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Complete normalization run."""

    session_id: str
    input_directory: Path
    output_directory: Path
    summary_file: Path
    report_file: Path

    grid_points: int
    common_distance_m: float

    artifacts: tuple[NormalizedArtifact, ...]


def resolve_project_path(path: Path) -> Path:
    """Resolve a path relative to the project root."""

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def project_relative(path: Path) -> str:
    """Return a project-relative path when possible."""

    resolved = path.resolve()

    try:
        return str(
            resolved.relative_to(PROJECT_ROOT.resolve())
        )
    except ValueError:
        return str(resolved)


def require_columns(
    dataframe: pd.DataFrame,
    required: Sequence[str],
    source_name: str,
) -> None:
    """Require a complete schema."""

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{source_name} is missing columns: "
            + ", ".join(missing)
        )


def atomic_csv_write(
    dataframe: pd.DataFrame,
    output_file: Path,
) -> None:
    """Write a CSV through a temporary file."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = output_file.with_suffix(
        output_file.suffix + ".tmp"
    )

    dataframe.to_csv(
        temporary,
        index=False,
    )

    temporary.replace(
        output_file
    )


def atomic_text_write(
    text: str,
    output_file: Path,
) -> None:
    """Write a text file through a temporary file."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = output_file.with_suffix(
        output_file.suffix + ".tmp"
    )

    temporary.write_text(
        text,
        encoding="utf-8",
    )

    temporary.replace(
        output_file
    )


def resolve_summary_output_file(
    stored_path: str,
) -> Path:
    """Resolve an extraction-summary output path."""

    path = Path(
        str(stored_path)
    )

    return resolve_project_path(
        path
    )


def prepare_lap(
    *,
    lap_index: int,
    source_file: Path,
) -> PreparedLap:
    """
    Validate and prepare one extracted lap.

    Duplicate distance coordinates are collapsed by keeping the last sample
    at that exact distance. Later timestamps are preferable when the vehicle
    spent more than one telemetry frame at the same physical distance.
    """

    dataframe = pd.read_csv(
        source_file
    )

    require_columns(
        dataframe,
        REQUIRED_EXTRACTED_COLUMNS,
        source_file.name,
    )

    if len(dataframe) < 2:
        raise ValueError(
            f"{source_file.name} contains fewer than two rows."
        )

    for column in (
        "lap_elapsed_time_s",
        "lap_distance_m",
        "speed_mps",
        "speed_kph",
        "throttle_pct",
        "brake_pct",
        "steer_normalized",
        "gear",
        "longitudinal_g",
        "lateral_g",
    ):
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    required_numeric = (
        "lap_elapsed_time_s",
        "lap_distance_m",
        "speed_mps",
        "speed_kph",
        "throttle_pct",
        "brake_pct",
        "steer_normalized",
        "gear",
    )

    for column in required_numeric:
        if dataframe[column].isna().any():
            raise ValueError(
                f"{source_file.name}: required channel "
                f"{column} contains NaN values."
            )

    distance = dataframe[
        "lap_distance_m"
    ].to_numpy(dtype=float)

    elapsed = dataframe[
        "lap_elapsed_time_s"
    ].to_numpy(dtype=float)

    if np.any(
        np.diff(distance) < -1e-6
    ):
        raise ValueError(
            f"{source_file.name}: lap distance is not monotonic."
        )

    if np.any(
        np.diff(elapsed) < -1e-9
    ):
        raise ValueError(
            f"{source_file.name}: lap elapsed time is not monotonic."
        )

    # Retain the final sample at each repeated distance coordinate.
    keep = np.ones(
        len(dataframe),
        dtype=bool,
    )

    if len(distance) > 1:
        keep[:-1] = (
            np.diff(distance) > 1e-9
        )

    prepared = dataframe.loc[
        keep
    ].reset_index(drop=True)

    duplicates_removed = (
        len(dataframe)
        - len(prepared)
    )

    prepared_distance = prepared[
        "lap_distance_m"
    ].to_numpy(dtype=float)

    if len(prepared_distance) < 2:
        raise ValueError(
            f"{source_file.name}: insufficient unique distance samples."
        )

    if np.any(
        np.diff(prepared_distance) <= 0.0
    ):
        raise ValueError(
            f"{source_file.name}: distance coordinate is not strictly "
            "increasing after duplicate removal."
        )

    source_distance = float(
        dataframe["lap_distance_m"].iloc[-1]
    )

    source_duration = float(
        dataframe["lap_elapsed_time_s"].iloc[-1]
    )

    return PreparedLap(
        lap_index=lap_index,
        lap_id=str(
            dataframe["lap_id"].iloc[0]
        ),
        lap_number=str(
            dataframe["source_lap_number"].iloc[0]
        ),
        session_id=str(
            dataframe["session_id"].iloc[0]
        ),
        capture_id=str(
            dataframe["capture_id"].iloc[0]
        ),
        source_file=source_file,
        source_rows=len(dataframe),
        source_distance_m=source_distance,
        source_duration_s=source_duration,
        duplicate_distance_rows_removed=(
            duplicates_removed
        ),
        distance=prepared_distance,
        dataframe=prepared,
    )


def linear_interpolate(
    lap: PreparedLap,
    channel: str,
    grid: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate one continuous channel."""

    values = lap.dataframe[
        channel
    ].to_numpy(dtype=float)

    finite = (
        np.isfinite(lap.distance)
        & np.isfinite(values)
    )

    if np.count_nonzero(finite) < 2:
        return np.full(
            len(grid),
            np.nan,
            dtype=float,
        )

    return np.interp(
        grid,
        lap.distance[finite],
        values[finite],
    )


def nearest_interpolate(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor interpolation for a discrete channel."""

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x_valid = x[finite]
    y_valid = y[finite]

    if len(x_valid) == 0:
        return np.full(
            len(grid),
            np.nan,
            dtype=float,
        )

    if len(x_valid) == 1:
        return np.full(
            len(grid),
            y_valid[0],
            dtype=float,
        )

    right = np.searchsorted(
        x_valid,
        grid,
        side="left",
    )

    right = np.clip(
        right,
        1,
        len(x_valid) - 1,
    )

    left = right - 1

    left_distance = (
        grid - x_valid[left]
    )

    right_distance = (
        x_valid[right] - grid
    )

    use_right = (
        right_distance < left_distance
    )

    nearest = np.where(
        use_right,
        right,
        left,
    )

    return y_valid[nearest]


def normalize_one_lap(
    *,
    lap: PreparedLap,
    distance_grid: np.ndarray,
) -> pd.DataFrame:
    """Normalize one lap to the shared distance grid."""

    normalized = pd.DataFrame(
        {
            "session_id": lap.session_id,
            "capture_id": lap.capture_id,
            "lap_id": lap.lap_id,
            "lap_number": lap.lap_number,
            "distance_m": distance_grid,
        }
    )

    mapping = {
        "lap_elapsed_time_s": "elapsed_time_s",
        "speed_mps": "speed_mps",
        "speed_kph": "speed_kph",
        "throttle_pct": "throttle_pct",
        "brake_pct": "brake_pct",
        "steer_normalized": "steer_normalized",
        "longitudinal_g": "longitudinal_g",
        "lateral_g": "lateral_g",
    }

    for source, destination in mapping.items():
        normalized[destination] = (
            linear_interpolate(
                lap,
                source,
                distance_grid,
            )
        )

    normalized["gear"] = (
        nearest_interpolate(
            lap.distance,
            lap.dataframe[
                "gear"
            ].to_numpy(dtype=float),
            distance_grid,
        )
    )

    return normalized.loc[
        :,
        NORMALIZED_SCHEMA,
    ]


def source_time_at_distance(
    lap: PreparedLap,
    distance_m: float,
) -> float:
    """Interpolate source elapsed time at one physical distance."""

    return float(
        np.interp(
            distance_m,
            lap.distance,
            lap.dataframe[
                "lap_elapsed_time_s"
            ].to_numpy(dtype=float),
        )
    )


def validate_normalized_lap(
    *,
    dataframe: pd.DataFrame,
    expected_rows: int,
) -> None:
    """Validate one normalized lap artifact."""

    require_columns(
        dataframe,
        NORMALIZED_SCHEMA,
        "normalized lap",
    )

    if len(dataframe) != expected_rows:
        raise ValueError(
            "Normalized lap has incorrect row count."
        )

    distance = dataframe[
        "distance_m"
    ].to_numpy(dtype=float)

    elapsed = dataframe[
        "elapsed_time_s"
    ].to_numpy(dtype=float)

    if not np.all(
        np.isfinite(distance)
    ):
        raise ValueError(
            "Normalized distance contains non-finite values."
        )

    if np.any(
        np.diff(distance) <= 0.0
    ):
        raise ValueError(
            "Normalized distance grid is not strictly increasing."
        )

    if not np.all(
        np.isfinite(elapsed)
    ):
        raise ValueError(
            "Normalized elapsed time contains non-finite values."
        )

    if np.any(
        np.diff(elapsed) < -1e-9
    ):
        raise ValueError(
            "Normalized elapsed time is not monotonic."
        )

    required = (
        "speed_mps",
        "speed_kph",
        "throttle_pct",
        "brake_pct",
        "steer_normalized",
        "gear",
    )

    for column in required:
        if dataframe[column].isna().any():
            raise ValueError(
                f"Normalized required channel contains NaN: {column}"
            )


def build_report(
    *,
    session_id: str,
    grid_points: int,
    common_distance_m: float,
    prepared_laps: Sequence[PreparedLap],
    artifacts: Sequence[NormalizedArtifact],
    summary_file: Path,
) -> str:
    """Build the human-readable normalization report."""

    lines = [
        "FH6 LAP DISTANCE-NORMALIZATION REPORT",
        "=" * 72,
        "",
        "SOURCE",
        "-" * 72,
        f"Session ID: {session_id}",
        f"Validated extracted laps: {len(prepared_laps)}",
        "",
        "NORMALIZATION MODEL",
        "-" * 72,
        f"Grid points: {grid_points}",
        f"Common distance: {common_distance_m:.3f} m",
        "Common distance selection: shortest validated extracted lap",
        "Continuous channels: linear interpolation",
        "Gear: nearest-neighbor interpolation",
        "Signal smoothing: none",
        "",
        "NORMALIZED LAPS",
        "-" * 72,
    ]

    for lap, artifact in zip(
        prepared_laps,
        artifacts,
    ):
        lines.extend(
            [
                "",
                f"Lap {artifact.lap_index:02d}",
                f"  Lap ID: {artifact.lap_id}",
                (
                    "  Source distance: "
                    f"{artifact.source_distance_m:.3f} m"
                ),
                (
                    "  Source duration: "
                    f"{artifact.source_duration_s:.3f} s"
                ),
                (
                    "  Duplicate distance rows removed: "
                    f"{lap.duplicate_distance_rows_removed}"
                ),
                (
                    "  Elapsed time at common distance: "
                    f"{artifact.normalized_final_elapsed_s:.6f} s"
                ),
                (
                    "  Timing interpolation error: "
                    f"{artifact.timing_error_s:.9f} s"
                ),
                f"  Rows: {artifact.rows}",
                (
                    "  Validation: "
                    f"{artifact.validation_status}"
                ),
                f"  Output: {artifact.output_file}",
            ]
        )

    overall = (
        "PASS"
        if artifacts
        and all(
            artifact.validation_status == "PASS"
            for artifact in artifacts
        )
        else "FAIL"
    )

    lines.extend(
        [
            "",
            "OUTPUTS",
            "-" * 72,
            f"Normalization summary: {summary_file}",
            "",
            "FINAL STATUS",
            "-" * 72,
            f"Normalized laps generated: {len(artifacts)}",
            f"Distance-grid validation: {overall}",
            "",
            "The normalized files are ready for future V0.3 comparison.",
            "No time-delta or gain/loss analysis was performed.",
        ]
    )

    return "\n".join(lines)


def run_normalization(
    args: argparse.Namespace,
) -> NormalizationResult:
    """Execute the complete normalization stage."""

    input_directory = (
        resolve_project_path(args.input_dir)
        if args.input_dir is not None
        else (
            DEFAULT_EXTRACTED_ROOT
            / args.session_id
        )
    )

    output_directory = (
        resolve_project_path(args.output_dir)
        if args.output_dir is not None
        else (
            DEFAULT_NORMALIZED_ROOT
            / args.session_id
        )
    )

    report_file = (
        resolve_project_path(args.report)
        if args.report is not None
        else (
            DEFAULT_REPORT_ROOT
            / (
                args.session_id
                + "_lap_normalization_report.txt"
            )
        )
    )

    extraction_summary_file = (
        input_directory
        / "lap_extraction_summary.csv"
    )

    if not extraction_summary_file.exists():
        raise FileNotFoundError(
            "Extraction summary does not exist: "
            f"{extraction_summary_file}"
        )

    extraction_summary = pd.read_csv(
        extraction_summary_file,
        dtype={
            "source_lap_number": str,
        },
    )

    required_summary_columns = (
        "extracted_lap_index",
        "validation_status",
        "output_csv",
    )

    require_columns(
        extraction_summary,
        required_summary_columns,
        "lap_extraction_summary.csv",
    )

    if extraction_summary.empty:
        raise ValueError(
            "Extraction summary contains no lap rows."
        )

    if not all(
        extraction_summary[
            "validation_status"
        ].astype(str).str.upper()
        == "PASS"
    ):
        raise ValueError(
            "One or more extracted laps are not validated PASS."
        )

    extraction_summary = extraction_summary.sort_values(
        "extracted_lap_index"
    )

    prepared_laps: list[PreparedLap] = []

    for _, row in extraction_summary.iterrows():
        lap_index = int(
            row["extracted_lap_index"]
        )

        source_file = resolve_summary_output_file(
            str(row["output_csv"])
        )

        if not source_file.exists():
            raise FileNotFoundError(
                f"Extracted lap file does not exist: {source_file}"
            )

        prepared_laps.append(
            prepare_lap(
                lap_index=lap_index,
                source_file=source_file,
            )
        )

    session_ids = {
        lap.session_id
        for lap in prepared_laps
    }

    if session_ids != {
        args.session_id
    }:
        raise ValueError(
            "Extracted lap session IDs do not match --session-id."
        )

    common_distance_m = min(
        lap.source_distance_m
        for lap in prepared_laps
    )

    if (
        not math.isfinite(common_distance_m)
        or common_distance_m <= 0.0
    ):
        raise ValueError(
            "Common normalization distance is invalid."
        )

    distance_grid = np.linspace(
        0.0,
        common_distance_m,
        args.grid_points,
        dtype=float,
    )

    built: list[
        tuple[PreparedLap, pd.DataFrame, Path]
    ] = []

    artifacts: list[NormalizedArtifact] = []

    for lap in prepared_laps:
        normalized = normalize_one_lap(
            lap=lap,
            distance_grid=distance_grid,
        )

        validate_normalized_lap(
            dataframe=normalized,
            expected_rows=args.grid_points,
        )

        source_common_time = (
            source_time_at_distance(
                lap,
                common_distance_m,
            )
        )

        normalized_final_time = float(
            normalized[
                "elapsed_time_s"
            ].iloc[-1]
        )

        timing_error = abs(
            normalized_final_time
            - source_common_time
        )

        validation_status = (
            "PASS"
            if timing_error <= 1e-6
            else "FAIL"
        )

        output_file = (
            output_directory
            / (
                f"lap_{lap.lap_index:02d}"
                "_normalized.csv"
            )
        )

        built.append(
            (
                lap,
                normalized,
                output_file,
            )
        )

        artifacts.append(
            NormalizedArtifact(
                lap_index=lap.lap_index,
                lap_id=lap.lap_id,
                output_file=output_file,
                source_distance_m=(
                    lap.source_distance_m
                ),
                source_duration_s=(
                    lap.source_duration_s
                ),
                common_distance_m=(
                    common_distance_m
                ),
                normalized_final_elapsed_s=(
                    normalized_final_time
                ),
                source_elapsed_at_common_distance_s=(
                    source_common_time
                ),
                timing_error_s=(
                    timing_error
                ),
                rows=len(normalized),
                validation_status=(
                    validation_status
                ),
            )
        )

    if not all(
        artifact.validation_status == "PASS"
        for artifact in artifacts
    ):
        raise ValueError(
            "Normalization validation failed before outputs were committed."
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for _, dataframe, output_file in built:
        atomic_csv_write(
            dataframe,
            output_file,
        )

    summary_file = (
        output_directory
        / "lap_normalization_summary.csv"
    )

    summary_dataframe = pd.DataFrame(
        [
            {
                "session_id": args.session_id,
                "lap_index": artifact.lap_index,
                "lap_id": artifact.lap_id,
                "source_distance_m": (
                    artifact.source_distance_m
                ),
                "source_duration_s": (
                    artifact.source_duration_s
                ),
                "common_distance_m": (
                    artifact.common_distance_m
                ),
                "grid_points": args.grid_points,
                "normalized_final_elapsed_s": (
                    artifact.normalized_final_elapsed_s
                ),
                "source_elapsed_at_common_distance_s": (
                    artifact.source_elapsed_at_common_distance_s
                ),
                "timing_error_s": (
                    artifact.timing_error_s
                ),
                "rows": artifact.rows,
                "validation_status": (
                    artifact.validation_status
                ),
                "output_csv": project_relative(
                    artifact.output_file
                ),
            }
            for artifact in artifacts
        ]
    )

    atomic_csv_write(
        summary_dataframe,
        summary_file,
    )

    report = build_report(
        session_id=args.session_id,
        grid_points=args.grid_points,
        common_distance_m=common_distance_m,
        prepared_laps=prepared_laps,
        artifacts=artifacts,
        summary_file=summary_file,
    )

    atomic_text_write(
        report,
        report_file,
    )

    return NormalizationResult(
        session_id=args.session_id,
        input_directory=input_directory,
        output_directory=output_directory,
        summary_file=summary_file,
        report_file=report_file,
        grid_points=args.grid_points,
        common_distance_m=common_distance_m,
        artifacts=tuple(artifacts),
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Normalize validated FH6 extracted laps onto "
            "a common distance grid."
        )
    )

    parser.add_argument(
        "--session-id",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--grid-points",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
    )

    parser.add_argument(
        "--report",
        type=Path,
    )

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""

    args = parse_arguments()

    if args.grid_points < 2:
        raise ValueError(
            "--grid-points must be at least 2."
        )

    print("=" * 72)
    print("FH6 Complete-Lap Distance Normalizer")
    print("=" * 72)

    result = run_normalization(
        args
    )

    print(
        f"\nSession ID: {result.session_id}"
    )

    print(
        f"Validated extracted laps: {len(result.artifacts)}"
    )

    print(
        f"Common distance: {result.common_distance_m:.3f} m"
    )

    print(
        f"Grid points per lap: {result.grid_points}"
    )

    for artifact in result.artifacts:
        print(
            f"  Lap {artifact.lap_index:02d}: "
            f"{artifact.rows} rows | "
            f"final t={artifact.normalized_final_elapsed_s:.3f} s | "
            f"{artifact.validation_status}"
        )

    print(
        "\nDistance-grid validation: PASS"
    )

    print(
        f"\nOutput directory:\n{result.output_directory}"
    )

    print(
        f"\nSummary CSV:\n{result.summary_file}"
    )

    print(
        f"\nReport:\n{result.report_file}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
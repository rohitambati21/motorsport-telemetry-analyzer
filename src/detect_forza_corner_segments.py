"""
detect_forza_corner_segments.py

Detect sustained cornering behavior from the automatically selected V1.1
reference lap and convert it into a complete corner/straight track partition.

Development role
----------------
The detector is intentionally driving-derived rather than geometric.

Primary signal:
    normalized steering magnitude

Supporting signal:
    lateral acceleration when finite/available

The canonical normalized telemetry is never modified. Smoothing is used only
inside this detector to stabilize semantic boundary detection.

Output:
    corner_segment_boundaries.csv
    corner_detection_trace.csv
    <session>_corner_detection_report.txt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

DEFAULT_V11_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "data"
    / "forza"
    / "lap_analysis_v11"
)

DEFAULT_REPORT_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "v11"
)

REQUIRED_REFERENCE_COLUMNS: Final[tuple[str, ...]] = (
    "session_id",
    "capture_id",
    "lap_id",
    "distance_m",
    "elapsed_time_s",
    "speed_kph",
    "steer_normalized",
)

DISTANCE_TOLERANCE_M: Final[float] = 1e-9


def resolve_project_path(path: Path) -> Path:
    """Resolve a path relative to the repository root."""

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def project_relative(path: Path) -> str:
    """Return repository-relative paths where possible."""

    resolved = path.resolve()

    try:
        return str(
            resolved.relative_to(
                PROJECT_ROOT.resolve()
            )
        )
    except ValueError:
        return str(resolved)


def require_columns(
    dataframe: pd.DataFrame,
    required: Sequence[str],
    source_name: str,
) -> None:
    """Require a complete source schema."""

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: "
            + ", ".join(missing)
        )


def parse_bool_series(series: pd.Series) -> pd.Series:
    """Parse loose CSV boolean representations."""

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
            }
        )
    )


def atomic_csv_write(
    dataframe: pd.DataFrame,
    output_file: Path,
) -> None:
    """Write a CSV atomically."""

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

    temporary.replace(output_file)


def atomic_text_write(
    text: str,
    output_file: Path,
) -> None:
    """Write UTF-8 text atomically."""

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

    temporary.replace(output_file)


def physical_window_points(
    distance_m: np.ndarray,
    window_distance_m: float,
) -> int:
    """
    Convert a physical smoothing distance to an odd rolling-window size.

    The normalized laps are uniformly spaced, but deriving the point count
    from the actual grid keeps the detector robust to future grid changes.
    """

    spacing = np.diff(distance_m)

    spacing = spacing[
        np.isfinite(spacing)
        & (spacing > 0.0)
    ]

    if spacing.size == 0:
        raise ValueError(
            "Cannot determine normalized distance-grid spacing."
        )

    median_spacing = float(
        np.median(spacing)
    )

    points = max(
        1,
        int(
            round(
                window_distance_m
                / median_spacing
            )
        ),
    )

    if points % 2 == 0:
        points += 1

    return points


def load_reference_lap(
    *,
    session_id: str,
    selection_file: Path,
) -> tuple[pd.DataFrame, pd.Series, Path]:
    """Load the automatically selected V1.1 reference lap."""

    if not selection_file.exists():
        raise FileNotFoundError(
            "Reference-selection artifact missing: "
            f"{selection_file}"
        )

    selection = pd.read_csv(
        selection_file
    )

    required_selection = (
        "session_id",
        "capture_id",
        "lap_index",
        "lap_id",
        "source_lap_csv",
        "is_reference",
        "validation_status",
    )

    require_columns(
        selection,
        required_selection,
        selection_file.name,
    )

    if not bool(
        (
            selection[
                "validation_status"
            ]
            .astype(str)
            .str.upper()
            == "PASS"
        ).all()
    ):
        raise ValueError(
            "Reference-selection artifact contains non-PASS rows."
        )

    if not bool(
        (
            selection[
                "session_id"
            ].astype(str)
            == session_id
        ).all()
    ):
        raise ValueError(
            "Reference-selection session mismatch."
        )

    reference_mask = parse_bool_series(
        selection[
            "is_reference"
        ]
    )

    if int(
        reference_mask.sum()
    ) != 1:
        raise ValueError(
            "Exactly one automatically selected reference lap is required."
        )

    reference_row = selection.loc[
        reference_mask
    ].iloc[0]

    source_file = resolve_project_path(
        Path(
            str(
                reference_row[
                    "source_lap_csv"
                ]
            )
        )
    )

    if not source_file.exists():
        raise FileNotFoundError(
            f"Reference normalized lap missing: {source_file}"
        )

    dataframe = pd.read_csv(
        source_file
    )

    require_columns(
        dataframe,
        REQUIRED_REFERENCE_COLUMNS,
        source_file.name,
    )

    if "lateral_g" not in dataframe.columns:
        dataframe[
            "lateral_g"
        ] = np.nan

    numeric_columns = (
        "distance_m",
        "elapsed_time_s",
        "speed_kph",
        "steer_normalized",
        "lateral_g",
    )

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    required_finite = (
        "distance_m",
        "elapsed_time_s",
        "speed_kph",
        "steer_normalized",
    )

    if dataframe[
        list(required_finite)
    ].isna().any().any():
        raise ValueError(
            "Reference normalized lap contains invalid required telemetry."
        )

    distance = dataframe[
        "distance_m"
    ].to_numpy(dtype=float)

    if np.any(
        np.diff(distance) <= 0.0
    ):
        raise ValueError(
            "Reference distance grid must be strictly increasing."
        )

    return (
        dataframe,
        reference_row,
        source_file,
    )


def build_detection_trace(
    *,
    dataframe: pd.DataFrame,
    smoothing_distance_m: float,
    enter_abs_steer: float,
    exit_abs_steer: float,
    secondary_min_abs_steer: float,
    secondary_abs_lateral_g: float,
) -> pd.DataFrame:
    """Build the detection-only smoothed signals and hysteresis triggers."""

    trace = dataframe[
        [
            "distance_m",
            "speed_kph",
            "steer_normalized",
            "lateral_g",
        ]
    ].copy()

    distance = trace[
        "distance_m"
    ].to_numpy(dtype=float)

    window_points = physical_window_points(
        distance,
        smoothing_distance_m,
    )

    abs_steer = trace[
        "steer_normalized"
    ].abs()

    trace[
        "smoothed_abs_steer"
    ] = (
        abs_steer
        .rolling(
            window=window_points,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    abs_lateral = trace[
        "lateral_g"
    ].abs()

    if bool(
        abs_lateral.notna().any()
    ):
        trace[
            "smoothed_abs_lateral_g"
        ] = (
            abs_lateral
            .rolling(
                window=window_points,
                center=True,
                min_periods=1,
            )
            .mean()
        )
    else:
        trace[
            "smoothed_abs_lateral_g"
        ] = np.nan

    trace[
        "primary_enter_trigger"
    ] = (
        trace[
            "smoothed_abs_steer"
        ]
        >= enter_abs_steer
    )

    trace[
        "primary_hold_trigger"
    ] = (
        trace[
            "smoothed_abs_steer"
        ]
        >= exit_abs_steer
    )

    lateral_available = trace[
        "smoothed_abs_lateral_g"
    ].notna()

    trace[
        "secondary_trigger"
    ] = (
        lateral_available
        & (
            trace[
                "smoothed_abs_steer"
            ]
            >= secondary_min_abs_steer
        )
        & (
            trace[
                "smoothed_abs_lateral_g"
            ]
            >= secondary_abs_lateral_g
        )
    )

    trace[
        "smoothing_window_points"
    ] = window_points

    trace[
        "smoothing_distance_m"
    ] = smoothing_distance_m

    trace[
        "enter_abs_steer"
    ] = enter_abs_steer

    trace[
        "exit_abs_steer"
    ] = exit_abs_steer

    return trace


def hysteresis_mask(
    trace: pd.DataFrame,
) -> np.ndarray:
    """
    Build a stable corner mask.

    A corner begins on the stronger entry condition and remains active while
    either the lower steering hold threshold or the lateral-g support signal
    remains active.
    """

    enter = (
        trace[
            "primary_enter_trigger"
        ].to_numpy(dtype=bool)
        | trace[
            "secondary_trigger"
        ].to_numpy(dtype=bool)
    )

    hold = (
        trace[
            "primary_hold_trigger"
        ].to_numpy(dtype=bool)
        | trace[
            "secondary_trigger"
        ].to_numpy(dtype=bool)
    )

    mask = np.zeros(
        len(trace),
        dtype=bool,
    )

    active = False

    for index in range(
        len(trace)
    ):
        if not active:
            if enter[index]:
                active = True
                mask[index] = True
        else:
            if hold[index]:
                mask[index] = True
            else:
                active = False

    return mask


def intervals_from_mask(
    distance_m: np.ndarray,
    mask: np.ndarray,
) -> list[tuple[float, float]]:
    """Convert a boolean mask to contiguous physical-distance intervals."""

    intervals: list[
        tuple[float, float]
    ] = []

    start_index: int | None = None

    for index, active in enumerate(
        mask
    ):
        if active and start_index is None:
            start_index = index

        if (
            start_index is not None
            and (
                not active
                or index
                == len(mask) - 1
            )
        ):
            end_index = (
                index
                if active
                and index
                == len(mask) - 1
                else index - 1
            )

            intervals.append(
                (
                    float(
                        distance_m[
                            start_index
                        ]
                    ),
                    float(
                        distance_m[
                            end_index
                        ]
                    ),
                )
            )

            start_index = None

    return intervals


def merge_intervals(
    intervals: list[
        tuple[float, float]
    ],
    maximum_gap_m: float,
) -> list[
    tuple[float, float]
]:
    """Merge intervals separated by only a short physical-distance gap."""

    if not intervals:
        return []

    ordered = sorted(
        intervals,
        key=lambda value: value[0],
    )

    merged: list[
        tuple[float, float]
    ] = []

    current_start, current_end = (
        ordered[0]
    )

    for start, end in ordered[1:]:
        gap = start - current_end

        if gap <= maximum_gap_m:
            current_end = max(
                current_end,
                end,
            )
        else:
            merged.append(
                (
                    current_start,
                    current_end,
                )
            )

            current_start = start
            current_end = end

    merged.append(
        (
            current_start,
            current_end,
        )
    )

    return merged


def filter_minimum_length(
    intervals: list[
        tuple[float, float]
    ],
    minimum_length_m: float,
) -> list[
    tuple[float, float]
]:
    """Remove physically tiny candidate corners."""

    return [
        (start, end)
        for start, end in intervals
        if (
            end - start
        )
        >= minimum_length_m
    ]


def pad_intervals(
    intervals: list[
        tuple[float, float]
    ],
    padding_m: float,
    minimum_distance_m: float,
    maximum_distance_m: float,
) -> list[
    tuple[float, float]
]:
    """Add modest entry/exit context around sustained corner cores."""

    padded = [
        (
            max(
                minimum_distance_m,
                start - padding_m,
            ),
            min(
                maximum_distance_m,
                end + padding_m,
            ),
        )
        for start, end in intervals
    ]

    # Padding can create overlaps; collapse them to retain a valid partition.
    return merge_intervals(
        padded,
        maximum_gap_m=0.0,
    )


def classify_corner_direction(
    steer_values: np.ndarray,
) -> str:
    """
    Classify the dominant steering direction.

    Positive normalized steering is treated as RIGHT and negative as LEFT.
    Mixed sustained sign behavior is classified conservatively as COMPLEX.
    """

    finite = steer_values[
        np.isfinite(
            steer_values
        )
    ]

    if finite.size == 0:
        return "COMPLEX"

    meaningful = finite[
        np.abs(finite)
        >= 0.02
    ]

    if meaningful.size == 0:
        return "COMPLEX"

    positive_fraction = float(
        np.mean(
            meaningful > 0.0
        )
    )

    negative_fraction = float(
        np.mean(
            meaningful < 0.0
        )
    )

    if (
        positive_fraction
        >= 0.25
        and negative_fraction
        >= 0.25
    ):
        return "COMPLEX"

    mean_signed = float(
        np.mean(
            meaningful
        )
    )

    if mean_signed > 0.0:
        return "RIGHT"

    if mean_signed < 0.0:
        return "LEFT"

    return "COMPLEX"


def segment_statistics(
    *,
    dataframe: pd.DataFrame,
    trace: pd.DataFrame,
    start_distance_m: float,
    end_distance_m: float,
    segment_type: str,
) -> dict[str, object]:
    """Calculate descriptive reference-lap telemetry for one segment."""

    distance = dataframe[
        "distance_m"
    ].to_numpy(dtype=float)

    speed = dataframe[
        "speed_kph"
    ].to_numpy(dtype=float)

    steer = dataframe[
        "steer_normalized"
    ].to_numpy(dtype=float)

    lateral = dataframe[
        "lateral_g"
    ].to_numpy(dtype=float)

    inside = (
        (distance >= start_distance_m)
        & (distance <= end_distance_m)
    )

    if not bool(
        inside.any()
    ):
        midpoint = (
            start_distance_m
            + end_distance_m
        ) / 2.0

        nearest = int(
            np.argmin(
                np.abs(
                    distance - midpoint
                )
            )
        )

        inside[
            nearest
        ] = True

    speed_inside = speed[
        inside
    ]

    steer_inside = steer[
        inside
    ]

    lateral_inside = lateral[
        inside
    ]

    finite_lateral = lateral_inside[
        np.isfinite(
            lateral_inside
        )
    ]

    entry_speed = float(
        np.interp(
            start_distance_m,
            distance,
            speed,
        )
    )

    exit_speed = float(
        np.interp(
            end_distance_m,
            distance,
            speed,
        )
    )

    trace_distance = trace[
        "distance_m"
    ].to_numpy(dtype=float)

    trace_inside = (
        (trace_distance >= start_distance_m)
        & (trace_distance <= end_distance_m)
    )

    primary_fraction = float(
        trace.loc[
            trace_inside,
            "primary_enter_trigger",
        ].astype(bool).mean()
    )

    secondary_fraction = float(
        trace.loc[
            trace_inside,
            "secondary_trigger",
        ].astype(bool).mean()
    )

    direction = (
        classify_corner_direction(
            steer_inside
        )
        if segment_type
        == "CORNER"
        else "STRAIGHT"
    )

    return {
        "direction": direction,
        "entry_speed_kph": (
            entry_speed
        ),
        "minimum_speed_kph": float(
            np.min(
                speed_inside
            )
        ),
        "maximum_speed_kph": float(
            np.max(
                speed_inside
            )
        ),
        "exit_speed_kph": (
            exit_speed
        ),
        "mean_abs_steer": float(
            np.mean(
                np.abs(
                    steer_inside
                )
            )
        ),
        "peak_abs_steer": float(
            np.max(
                np.abs(
                    steer_inside
                )
            )
        ),
        "mean_signed_steer": float(
            np.mean(
                steer_inside
            )
        ),
        "mean_abs_lateral_g": (
            float(
                np.mean(
                    np.abs(
                        finite_lateral
                    )
                )
            )
            if finite_lateral.size
            else math.nan
        ),
        "peak_abs_lateral_g": (
            float(
                np.max(
                    np.abs(
                        finite_lateral
                    )
                )
            )
            if finite_lateral.size
            else math.nan
        ),
        "primary_detection_fraction": (
            primary_fraction
        ),
        "secondary_detection_fraction": (
            secondary_fraction
        ),
    }


def build_partition(
    *,
    corner_intervals: list[
        tuple[float, float]
    ],
    minimum_distance_m: float,
    maximum_distance_m: float,
) -> list[
    tuple[str, float, float]
]:
    """Create a complete non-overlapping CORNER/STRAIGHT partition."""

    partition: list[
        tuple[str, float, float]
    ] = []

    cursor = minimum_distance_m

    for corner_start, corner_end in (
        corner_intervals
    ):
        if (
            corner_start
            > cursor
            + DISTANCE_TOLERANCE_M
        ):
            partition.append(
                (
                    "STRAIGHT",
                    cursor,
                    corner_start,
                )
            )

        partition.append(
            (
                "CORNER",
                corner_start,
                corner_end,
            )
        )

        cursor = corner_end

    if (
        cursor
        < maximum_distance_m
        - DISTANCE_TOLERANCE_M
    ):
        partition.append(
            (
                "STRAIGHT",
                cursor,
                maximum_distance_m,
            )
        )

    return [
        item
        for item in partition
        if (
            item[2]
            - item[1]
        )
        > DISTANCE_TOLERANCE_M
    ]


def validate_partition(
    partition: list[
        tuple[str, float, float]
    ],
    minimum_distance_m: float,
    maximum_distance_m: float,
) -> None:
    """Prove exact coverage, positive length, no gaps, and no overlaps."""

    if not partition:
        raise ValueError(
            "Corner detector produced an empty partition."
        )

    if abs(
        partition[0][1]
        - minimum_distance_m
    ) > DISTANCE_TOLERANCE_M:
        raise ValueError(
            "Partition does not begin at the lap origin."
        )

    if abs(
        partition[-1][2]
        - maximum_distance_m
    ) > DISTANCE_TOLERANCE_M:
        raise ValueError(
            "Partition does not end at the common lap endpoint."
        )

    for index, (
        _,
        start,
        end,
    ) in enumerate(
        partition
    ):
        if (
            end - start
        ) <= 0.0:
            raise ValueError(
                "Partition contains a non-positive-length segment."
            )

        if index > 0:
            previous_end = (
                partition[
                    index - 1
                ][2]
            )

            if abs(
                start - previous_end
            ) > DISTANCE_TOLERANCE_M:
                raise ValueError(
                    "Partition contains a gap or overlap."
                )


def build_report(
    *,
    session_id: str,
    reference_lap: int,
    boundaries: pd.DataFrame,
    common_distance_m: float,
    smoothing_distance_m: float,
    enter_abs_steer: float,
    exit_abs_steer: float,
    minimum_corner_length_m: float,
    merge_gap_m: float,
    padding_m: float,
) -> str:
    """Build the detector report."""

    corners = boundaries.loc[
        boundaries[
            "segment_type"
        ]
        == "CORNER"
    ]

    straights = boundaries.loc[
        boundaries[
            "segment_type"
        ]
        == "STRAIGHT"
    ]

    corner_distance = float(
        corners[
            "length_m"
        ].sum()
    )

    straight_distance = float(
        straights[
            "length_m"
        ].sum()
    )

    lines = [
        "FH6 V1.1 CORNER-AWARE SEGMENT DETECTOR",
        "=" * 72,
        "",
        f"Session ID: {session_id}",
        (
            f"Automatic reference lap: "
            f"Lap {reference_lap:02d}"
        ),
        "",
        "DETECTION MODEL",
        "-" * 72,
        (
            f"Detection smoothing distance: "
            f"{smoothing_distance_m:.3f} m"
        ),
        (
            f"Corner entry steering threshold: "
            f"{enter_abs_steer:.6f}"
        ),
        (
            f"Corner exit steering threshold: "
            f"{exit_abs_steer:.6f}"
        ),
        (
            f"Minimum corner length: "
            f"{minimum_corner_length_m:.3f} m"
        ),
        (
            f"Maximum merge gap: "
            f"{merge_gap_m:.3f} m"
        ),
        (
            f"Entry/exit context padding: "
            f"{padding_m:.3f} m"
        ),
        "",
        "PARTITION",
        "-" * 72,
        (
            f"Common distance: "
            f"{common_distance_m:.6f} m"
        ),
        (
            f"Detected corners: "
            f"{len(corners)}"
        ),
        (
            f"Detected straights: "
            f"{len(straights)}"
        ),
        (
            f"Corner distance: "
            f"{corner_distance:.6f} m"
        ),
        (
            f"Straight distance: "
            f"{straight_distance:.6f} m"
        ),
        (
            "Partition coverage: "
            f"{corner_distance + straight_distance:.6f} m"
        ),
        "",
        "CORNER LIST",
        "-" * 72,
    ]

    for _, row in corners.iterrows():
        lines.append(
            (
                f"Corner {int(row['corner_id']):02d} | "
                f"{row['direction']} | "
                f"{float(row['start_distance_m']):.3f}-"
                f"{float(row['end_distance_m']):.3f} m | "
                f"{float(row['length_m']):.3f} m | "
                f"min speed={float(row['minimum_speed_kph']):.2f} km/h | "
                f"peak |steer|={float(row['peak_abs_steer']):.4f}"
            )
        )

    lines.extend(
        [
            "",
            "INTERPRETATION BOUNDARY",
            "-" * 72,
            (
                "These segments are derived from reference-lap driving "
                "telemetry. They represent sustained cornering behavior, "
                "not a reconstructed geometric road centerline."
            ),
            "",
            "STATUS",
            "-" * 72,
            "Corner/straight partition generation: PASS",
            "Track coverage: PASS",
            "Gap/overlap check: PASS",
            "",
            (
                "Manual visual QA of the corner-detection figure is "
                "required before treating the detector parameters as "
                "the validated V1.1 baseline."
            ),
        ]
    )

    return "\n".join(lines)


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Detect sustained reference-lap cornering and build a complete "
            "V1.1 corner/straight partition."
        )
    )

    parser.add_argument(
        "--session-id",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--selection-file",
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

    parser.add_argument(
        "--smoothing-distance-m",
        type=float,
        default=45.0,
    )

    parser.add_argument(
        "--enter-abs-steer",
        type=float,
        default=0.080,
    )

    parser.add_argument(
        "--exit-abs-steer",
        type=float,
        default=0.035,
    )

    parser.add_argument(
        "--secondary-min-abs-steer",
        type=float,
        default=0.040,
    )

    parser.add_argument(
        "--secondary-abs-lateral-g",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--minimum-corner-length-m",
        type=float,
        default=45.0,
    )

    parser.add_argument(
        "--merge-gap-m",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--padding-m",
        type=float,
        default=15.0,
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_arguments()

    if args.smoothing_distance_m <= 0.0:
        raise ValueError(
            "--smoothing-distance-m must be positive."
        )

    if not (
        0.0
        <= args.exit_abs_steer
        < args.enter_abs_steer
        <= 1.0
    ):
        raise ValueError(
            "Steering hysteresis requires "
            "0 <= exit < enter <= 1."
        )

    if not (
        0.0
        <= args.secondary_min_abs_steer
        <= 1.0
    ):
        raise ValueError(
            "--secondary-min-abs-steer must be in [0, 1]."
        )

    if args.secondary_abs_lateral_g < 0.0:
        raise ValueError(
            "--secondary-abs-lateral-g cannot be negative."
        )

    if args.minimum_corner_length_m <= 0.0:
        raise ValueError(
            "--minimum-corner-length-m must be positive."
        )

    if args.merge_gap_m < 0.0:
        raise ValueError(
            "--merge-gap-m cannot be negative."
        )

    if args.padding_m < 0.0:
        raise ValueError(
            "--padding-m cannot be negative."
        )

    output_directory = (
        resolve_project_path(
            args.output_dir
        )
        if args.output_dir
        else (
            DEFAULT_V11_ROOT
            / args.session_id
        )
    )

    selection_file = (
        resolve_project_path(
            args.selection_file
        )
        if args.selection_file
        else (
            output_directory
            / "reference_lap_selection.csv"
        )
    )

    report_file = (
        resolve_project_path(
            args.report
        )
        if args.report
        else (
            DEFAULT_REPORT_ROOT
            / (
                f"{args.session_id}"
                "_corner_detection_report.txt"
            )
        )
    )

    (
        reference,
        reference_row,
        source_file,
    ) = load_reference_lap(
        session_id=args.session_id,
        selection_file=(
            selection_file
        ),
    )

    reference_lap = int(
        reference_row[
            "lap_index"
        ]
    )

    capture_id = str(
        reference_row[
            "capture_id"
        ]
    )

    reference_lap_id = str(
        reference_row[
            "lap_id"
        ]
    )

    distance = reference[
        "distance_m"
    ].to_numpy(dtype=float)

    minimum_distance_m = float(
        distance[0]
    )

    maximum_distance_m = float(
        distance[-1]
    )

    trace = build_detection_trace(
        dataframe=reference,
        smoothing_distance_m=(
            args.smoothing_distance_m
        ),
        enter_abs_steer=(
            args.enter_abs_steer
        ),
        exit_abs_steer=(
            args.exit_abs_steer
        ),
        secondary_min_abs_steer=(
            args.secondary_min_abs_steer
        ),
        secondary_abs_lateral_g=(
            args.secondary_abs_lateral_g
        ),
    )

    raw_mask = hysteresis_mask(
        trace
    )

    intervals = intervals_from_mask(
        distance,
        raw_mask,
    )

    intervals = merge_intervals(
        intervals,
        maximum_gap_m=(
            args.merge_gap_m
        ),
    )

    intervals = filter_minimum_length(
        intervals,
        minimum_length_m=(
            args.minimum_corner_length_m
        ),
    )

    intervals = pad_intervals(
        intervals,
        padding_m=args.padding_m,
        minimum_distance_m=(
            minimum_distance_m
        ),
        maximum_distance_m=(
            maximum_distance_m
        ),
    )

    if not intervals:
        raise ValueError(
            "No validated corner segments were detected. "
            "Inspect detector thresholds before continuing."
        )

    partition = build_partition(
        corner_intervals=intervals,
        minimum_distance_m=(
            minimum_distance_m
        ),
        maximum_distance_m=(
            maximum_distance_m
        ),
    )

    validate_partition(
        partition,
        minimum_distance_m=(
            minimum_distance_m
        ),
        maximum_distance_m=(
            maximum_distance_m
        ),
    )

    records: list[
        dict[str, object]
    ] = []

    corner_id = 0

    for segment_index, (
        segment_type,
        start_distance_m,
        end_distance_m,
    ) in enumerate(
        partition,
        start=1,
    ):
        if segment_type == "CORNER":
            corner_id += 1
            current_corner_id: int | float = (
                corner_id
            )
        else:
            current_corner_id = math.nan

        statistics = segment_statistics(
            dataframe=reference,
            trace=trace,
            start_distance_m=(
                start_distance_m
            ),
            end_distance_m=(
                end_distance_m
            ),
            segment_type=(
                segment_type
            ),
        )

        records.append(
            {
                "session_id": (
                    args.session_id
                ),
                "capture_id": (
                    capture_id
                ),
                "reference_lap_index": (
                    reference_lap
                ),
                "reference_lap_id": (
                    reference_lap_id
                ),
                "segment_id": (
                    segment_index
                ),
                "segment_type": (
                    segment_type
                ),
                "corner_id": (
                    current_corner_id
                ),
                "direction": (
                    statistics[
                        "direction"
                    ]
                ),
                "start_distance_m": (
                    start_distance_m
                ),
                "end_distance_m": (
                    end_distance_m
                ),
                "length_m": (
                    end_distance_m
                    - start_distance_m
                ),
                "entry_speed_kph": (
                    statistics[
                        "entry_speed_kph"
                    ]
                ),
                "minimum_speed_kph": (
                    statistics[
                        "minimum_speed_kph"
                    ]
                ),
                "maximum_speed_kph": (
                    statistics[
                        "maximum_speed_kph"
                    ]
                ),
                "exit_speed_kph": (
                    statistics[
                        "exit_speed_kph"
                    ]
                ),
                "mean_abs_steer": (
                    statistics[
                        "mean_abs_steer"
                    ]
                ),
                "peak_abs_steer": (
                    statistics[
                        "peak_abs_steer"
                    ]
                ),
                "mean_signed_steer": (
                    statistics[
                        "mean_signed_steer"
                    ]
                ),
                "mean_abs_lateral_g": (
                    statistics[
                        "mean_abs_lateral_g"
                    ]
                ),
                "peak_abs_lateral_g": (
                    statistics[
                        "peak_abs_lateral_g"
                    ]
                ),
                "primary_detection_fraction": (
                    statistics[
                        "primary_detection_fraction"
                    ]
                ),
                "secondary_detection_fraction": (
                    statistics[
                        "secondary_detection_fraction"
                    ]
                ),
                "smoothing_distance_m": (
                    args.smoothing_distance_m
                ),
                "enter_abs_steer": (
                    args.enter_abs_steer
                ),
                "exit_abs_steer": (
                    args.exit_abs_steer
                ),
                "secondary_min_abs_steer": (
                    args.secondary_min_abs_steer
                ),
                "secondary_abs_lateral_g": (
                    args.secondary_abs_lateral_g
                ),
                "minimum_corner_length_m": (
                    args.minimum_corner_length_m
                ),
                "merge_gap_m": (
                    args.merge_gap_m
                ),
                "padding_m": (
                    args.padding_m
                ),
                "source_reference_csv": (
                    project_relative(
                        source_file
                    )
                ),
                "validation_status": (
                    "PASS"
                ),
            }
        )

    boundaries = pd.DataFrame(
        records
    )

    corners = boundaries.loc[
        boundaries[
            "segment_type"
        ]
        == "CORNER"
    ]

    straights = boundaries.loc[
        boundaries[
            "segment_type"
        ]
        == "STRAIGHT"
    ]

    if corners.empty:
        raise ValueError(
            "Partition contains no CORNER segments."
        )

    if straights.empty:
        raise ValueError(
            "Partition contains no STRAIGHT segments."
        )

    trace[
        "final_corner_mask"
    ] = False

    for start, end in intervals:
        trace.loc[
            (
                trace[
                    "distance_m"
                ]
                >= start
            )
            & (
                trace[
                    "distance_m"
                ]
                <= end
            ),
            "final_corner_mask",
        ] = True

    trace[
        "session_id"
    ] = args.session_id

    trace[
        "capture_id"
    ] = capture_id

    trace[
        "reference_lap_index"
    ] = reference_lap

    trace[
        "reference_lap_id"
    ] = reference_lap_id

    boundary_file = (
        output_directory
        / "corner_segment_boundaries.csv"
    )

    trace_file = (
        output_directory
        / "corner_detection_trace.csv"
    )

    atomic_csv_write(
        boundaries,
        boundary_file,
    )

    atomic_csv_write(
        trace,
        trace_file,
    )

    atomic_text_write(
        build_report(
            session_id=args.session_id,
            reference_lap=(
                reference_lap
            ),
            boundaries=boundaries,
            common_distance_m=(
                maximum_distance_m
                - minimum_distance_m
            ),
            smoothing_distance_m=(
                args.smoothing_distance_m
            ),
            enter_abs_steer=(
                args.enter_abs_steer
            ),
            exit_abs_steer=(
                args.exit_abs_steer
            ),
            minimum_corner_length_m=(
                args.minimum_corner_length_m
            ),
            merge_gap_m=(
                args.merge_gap_m
            ),
            padding_m=(
                args.padding_m
            ),
        ),
        report_file,
    )

    print("=" * 72)
    print(
        "FH6 V1.1 Corner-Aware Segment Detector"
    )
    print("=" * 72)

    print(
        f"\nAutomatic reference: "
        f"Lap {reference_lap:02d}"
    )

    print(
        f"Detected corners: "
        f"{len(corners)}"
    )

    print(
        f"Detected straights: "
        f"{len(straights)}"
    )

    print(
        f"Track segments: "
        f"{len(boundaries)}"
    )

    print(
        f"Coverage: "
        f"{minimum_distance_m:.6f}-"
        f"{maximum_distance_m:.6f} m"
    )

    print(
        "\nCorner/straight partition: PASS"
    )

    print(
        f"\nBoundary CSV:\n"
        f"{boundary_file}"
    )

    print(
        f"\nDetection trace:\n"
        f"{trace_file}"
    )

    print(
        f"\nDetection report:\n"
        f"{report_file}"
    )

    print()
    print(
        "Manual detection-figure QA is still required."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
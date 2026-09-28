"""
build_forza_corner_analysis.py

Apply the reference-derived V1.1 corner/straight partition to every validated
normalized lap and calculate exact segment timing.

Development role
----------------
The segmentation is generated once from the automatically selected reference.

Every lap then uses exactly the same physical-distance boundaries.

This preserves fair comparison and guarantees that:

    sum(segment deltas)
    =
    full normalized-lap delta

within floating-point tolerance.
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

TIMING_TOLERANCE_S: Final[float] = 1e-8
DISTANCE_TOLERANCE_M: Final[float] = 1e-9
CLASSIFICATION_TOLERANCE_S: Final[float] = 1e-9


def resolve_project_path(path: Path) -> Path:
    """Resolve repository-relative paths."""

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def project_relative(path: Path) -> str:
    """Return repository-relative paths when possible."""

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
    """Require a complete artifact schema."""

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
    """Parse loose boolean representations."""

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
    """Write CSV output atomically."""

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
    """Write text output atomically."""

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


def classify_delta(delta_s: float) -> str:
    """Classify one comparison segment by its net timing difference."""

    if delta_s > CLASSIFICATION_TOLERANCE_S:
        return "LOSS"

    if delta_s < -CLASSIFICATION_TOLERANCE_S:
        return "GAIN"

    return "NEUTRAL"


def load_selection(
    selection_file: Path,
    session_id: str,
) -> pd.DataFrame:
    """Load the validated V1.1 automatic-reference artifact."""

    if not selection_file.exists():
        raise FileNotFoundError(
            f"Reference selection missing: {selection_file}"
        )

    selection = pd.read_csv(
        selection_file
    )

    required = (
        "session_id",
        "capture_id",
        "lap_index",
        "lap_id",
        "final_elapsed_s",
        "source_lap_csv",
        "is_reference",
        "validation_status",
    )

    require_columns(
        selection,
        required,
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
            "Reference selection contains non-PASS rows."
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
            "Exactly one reference lap is required."
        )

    return selection.sort_values(
        "lap_index"
    ).reset_index(drop=True)


def load_boundaries(
    boundary_file: Path,
    session_id: str,
) -> pd.DataFrame:
    """Load and structurally validate the semantic track partition."""

    if not boundary_file.exists():
        raise FileNotFoundError(
            f"Corner boundaries missing: {boundary_file}"
        )

    boundaries = pd.read_csv(
        boundary_file
    )

    required = (
        "session_id",
        "reference_lap_index",
        "segment_id",
        "segment_type",
        "corner_id",
        "direction",
        "start_distance_m",
        "end_distance_m",
        "length_m",
        "validation_status",
    )

    require_columns(
        boundaries,
        required,
        boundary_file.name,
    )

    if not bool(
        (
            boundaries[
                "validation_status"
            ]
            .astype(str)
            .str.upper()
            == "PASS"
        ).all()
    ):
        raise ValueError(
            "Corner partition contains non-PASS rows."
        )

    if not bool(
        (
            boundaries[
                "session_id"
            ].astype(str)
            == session_id
        ).all()
    ):
        raise ValueError(
            "Corner-boundary session mismatch."
        )

    boundaries = boundaries.sort_values(
        "segment_id"
    ).reset_index(drop=True)

    starts = pd.to_numeric(
        boundaries[
            "start_distance_m"
        ],
        errors="coerce",
    ).to_numpy(dtype=float)

    ends = pd.to_numeric(
        boundaries[
            "end_distance_m"
        ],
        errors="coerce",
    ).to_numpy(dtype=float)

    if (
        not np.all(
            np.isfinite(starts)
        )
        or not np.all(
            np.isfinite(ends)
        )
    ):
        raise ValueError(
            "Corner partition contains invalid distances."
        )

    if np.any(
        ends <= starts
    ):
        raise ValueError(
            "Corner partition contains non-positive-length segments."
        )

    if len(starts) > 1:
        if not np.allclose(
            starts[1:],
            ends[:-1],
            rtol=0.0,
            atol=DISTANCE_TOLERANCE_M,
        ):
            raise ValueError(
                "Corner partition contains a gap or overlap."
            )

    segment_ids = pd.to_numeric(
        boundaries[
            "segment_id"
        ],
        errors="coerce",
    ).to_numpy(dtype=float)

    expected_ids = np.arange(
        1,
        len(boundaries) + 1,
        dtype=float,
    )

    if not np.array_equal(
        segment_ids,
        expected_ids,
    ):
        raise ValueError(
            "Segment IDs are not sequential."
        )

    types = set(
        boundaries[
            "segment_type"
        ]
        .astype(str)
        .str.upper()
    )

    if not types.issubset(
        {
            "CORNER",
            "STRAIGHT",
        }
    ):
        raise ValueError(
            "Unknown segment type found."
        )

    if "CORNER" not in types:
        raise ValueError(
            "Partition contains no corners."
        )

    if "STRAIGHT" not in types:
        raise ValueError(
            "Partition contains no straights."
        )

    return boundaries


def load_normalized_lap(
    source_file: Path,
    session_id: str,
) -> pd.DataFrame:
    """Load the frozen normalized lap used by the additive V1.1 analysis."""

    if not source_file.exists():
        raise FileNotFoundError(
            f"Normalized lap missing: {source_file}"
        )

    dataframe = pd.read_csv(
        source_file
    )

    required = (
        "session_id",
        "capture_id",
        "lap_id",
        "distance_m",
        "elapsed_time_s",
    )

    require_columns(
        dataframe,
        required,
        source_file.name,
    )

    if (
        str(
            dataframe[
                "session_id"
            ].iloc[0]
        )
        != session_id
    ):
        raise ValueError(
            f"{source_file.name}: session mismatch."
        )

    dataframe[
        "distance_m"
    ] = pd.to_numeric(
        dataframe[
            "distance_m"
        ],
        errors="coerce",
    )

    dataframe[
        "elapsed_time_s"
    ] = pd.to_numeric(
        dataframe[
            "elapsed_time_s"
        ],
        errors="coerce",
    )

    if dataframe[
        [
            "distance_m",
            "elapsed_time_s",
        ]
    ].isna().any().any():
        raise ValueError(
            f"{source_file.name}: invalid timing/distance values."
        )

    return dataframe


def boundary_vector(
    boundaries: pd.DataFrame,
) -> np.ndarray:
    """Return the complete ordered physical boundary vector."""

    starts = boundaries[
        "start_distance_m"
    ].to_numpy(dtype=float)

    ends = boundaries[
        "end_distance_m"
    ].to_numpy(dtype=float)

    return np.concatenate(
        (
            np.array(
                [
                    starts[0]
                ],
                dtype=float,
            ),
            ends,
        )
    )


def interpolate_boundary_times(
    dataframe: pd.DataFrame,
    boundaries_m: np.ndarray,
) -> np.ndarray:
    """Interpolate exact elapsed time at every semantic boundary."""

    distance = dataframe[
        "distance_m"
    ].to_numpy(dtype=float)

    elapsed = dataframe[
        "elapsed_time_s"
    ].to_numpy(dtype=float)

    if (
        boundaries_m[0]
        < distance[0]
        - DISTANCE_TOLERANCE_M
        or boundaries_m[-1]
        > distance[-1]
        + DISTANCE_TOLERANCE_M
    ):
        raise ValueError(
            "Semantic boundaries extend outside normalized telemetry."
        )

    return np.interp(
        boundaries_m,
        distance,
        elapsed,
    )


def build_lap_segment_times(
    *,
    selection: pd.DataFrame,
    boundaries: pd.DataFrame,
    session_id: str,
) -> tuple[
    pd.DataFrame,
    dict[int, np.ndarray],
]:
    """Apply identical semantic boundaries to every normalized lap."""

    boundaries_m = boundary_vector(
        boundaries
    )

    records: list[
        dict[str, object]
    ] = []

    boundary_times_by_lap: dict[
        int,
        np.ndarray,
    ] = {}

    reference_mask = parse_bool_series(
        selection[
            "is_reference"
        ]
    )

    for row_index, row in (
        selection.iterrows()
    ):
        lap_index = int(
            row[
                "lap_index"
            ]
        )

        source_file = resolve_project_path(
            Path(
                str(
                    row[
                        "source_lap_csv"
                    ]
                )
            )
        )

        dataframe = load_normalized_lap(
            source_file,
            session_id,
        )

        boundary_times = (
            interpolate_boundary_times(
                dataframe,
                boundaries_m,
            )
        )

        segment_times = np.diff(
            boundary_times
        )

        if np.any(
            segment_times <= 0.0
        ):
            raise ValueError(
                f"Lap {lap_index:02d}: non-positive semantic segment time."
            )

        expected_final = float(
            row[
                "final_elapsed_s"
            ]
        )

        reconstructed_final = (
            float(
                boundary_times[0]
            )
            + float(
                segment_times.sum()
            )
        )

        reconstruction_error = abs(
            reconstructed_final
            - expected_final
        )

        if (
            reconstruction_error
            > TIMING_TOLERANCE_S
        ):
            raise ValueError(
                f"Lap {lap_index:02d}: semantic segment timing "
                "does not reconstruct the normalized endpoint."
            )

        boundary_times_by_lap[
            lap_index
        ] = boundary_times

        for segment_position, (
            _,
            segment,
        ) in enumerate(
            boundaries.iterrows()
        ):
            records.append(
                {
                    "session_id": (
                        session_id
                    ),
                    "capture_id": str(
                        row[
                            "capture_id"
                        ]
                    ),
                    "lap_index": (
                        lap_index
                    ),
                    "lap_id": str(
                        row[
                            "lap_id"
                        ]
                    ),
                    "is_reference": bool(
                        reference_mask.iloc[
                            row_index
                        ]
                    ),
                    "segment_id": int(
                        segment[
                            "segment_id"
                        ]
                    ),
                    "segment_type": str(
                        segment[
                            "segment_type"
                        ]
                    ),
                    "corner_id": (
                        segment[
                            "corner_id"
                        ]
                    ),
                    "direction": str(
                        segment[
                            "direction"
                        ]
                    ),
                    "start_distance_m": float(
                        segment[
                            "start_distance_m"
                        ]
                    ),
                    "end_distance_m": float(
                        segment[
                            "end_distance_m"
                        ]
                    ),
                    "length_m": float(
                        segment[
                            "length_m"
                        ]
                    ),
                    "elapsed_start_s": float(
                        boundary_times[
                            segment_position
                        ]
                    ),
                    "elapsed_end_s": float(
                        boundary_times[
                            segment_position
                            + 1
                        ]
                    ),
                    "segment_time_s": float(
                        segment_times[
                            segment_position
                        ]
                    ),
                    "lap_reconstruction_error_s": (
                        reconstruction_error
                    ),
                    "source_lap_csv": (
                        project_relative(
                            source_file
                        )
                    ),
                    "validation_status": (
                        "PASS"
                    ),
                }
            )

    return (
        pd.DataFrame(
            records
        ),
        boundary_times_by_lap,
    )


def add_ranks(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Add overall and corner-only gain/loss ranking columns."""

    result = dataframe.copy()

    result[
        "segment_rank_by_loss"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )

    result[
        "segment_rank_by_gain"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )

    result[
        "corner_rank_by_loss"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )

    result[
        "corner_rank_by_gain"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )

    loss_mask = (
        result[
            "classification"
        ]
        == "LOSS"
    )

    gain_mask = (
        result[
            "classification"
        ]
        == "GAIN"
    )

    corner_mask = (
        result[
            "segment_type"
        ]
        == "CORNER"
    )

    if bool(
        loss_mask.any()
    ):
        result.loc[
            loss_mask,
            "segment_rank_by_loss",
        ] = (
            result.loc[
                loss_mask,
                "segment_delta_s",
            ]
            .rank(
                method="dense",
                ascending=False,
            )
            .astype("Int64")
        )

    if bool(
        gain_mask.any()
    ):
        result.loc[
            gain_mask,
            "segment_rank_by_gain",
        ] = (
            result.loc[
                gain_mask,
                "segment_delta_s",
            ]
            .rank(
                method="dense",
                ascending=True,
            )
            .astype("Int64")
        )

    corner_loss = (
        corner_mask
        & loss_mask
    )

    corner_gain = (
        corner_mask
        & gain_mask
    )

    if bool(
        corner_loss.any()
    ):
        result.loc[
            corner_loss,
            "corner_rank_by_loss",
        ] = (
            result.loc[
                corner_loss,
                "segment_delta_s",
            ]
            .rank(
                method="dense",
                ascending=False,
            )
            .astype("Int64")
        )

    if bool(
        corner_gain.any()
    ):
        result.loc[
            corner_gain,
            "corner_rank_by_gain",
        ] = (
            result.loc[
                corner_gain,
                "segment_delta_s",
            ]
            .rank(
                method="dense",
                ascending=True,
            )
            .astype("Int64")
        )

    return result


def build_comparison(
    *,
    session_id: str,
    capture_id: str,
    boundaries: pd.DataFrame,
    reference_lap: int,
    comparison_lap: int,
    reference_boundary_times: np.ndarray,
    comparison_boundary_times: np.ndarray,
) -> pd.DataFrame:
    """Build one corner-aware semantic timing comparison."""

    reference_times = np.diff(
        reference_boundary_times
    )

    comparison_times = np.diff(
        comparison_boundary_times
    )

    segment_delta = (
        comparison_times
        - reference_times
    )

    cumulative_delta = (
        comparison_boundary_times
        - reference_boundary_times
    )

    records: list[
        dict[str, object]
    ] = []

    for index, (
        _,
        segment,
    ) in enumerate(
        boundaries.iterrows()
    ):
        delta = float(
            segment_delta[
                index
            ]
        )

        records.append(
            {
                "session_id": (
                    session_id
                ),
                "capture_id": (
                    capture_id
                ),
                "reference_lap_index": (
                    reference_lap
                ),
                "comparison_lap_index": (
                    comparison_lap
                ),
                "segment_id": int(
                    segment[
                        "segment_id"
                    ]
                ),
                "segment_type": str(
                    segment[
                        "segment_type"
                    ]
                ),
                "corner_id": (
                    segment[
                        "corner_id"
                    ]
                ),
                "direction": str(
                    segment[
                        "direction"
                    ]
                ),
                "start_distance_m": float(
                    segment[
                        "start_distance_m"
                    ]
                ),
                "end_distance_m": float(
                    segment[
                        "end_distance_m"
                    ]
                ),
                "length_m": float(
                    segment[
                        "length_m"
                    ]
                ),
                "reference_segment_time_s": float(
                    reference_times[
                        index
                    ]
                ),
                "comparison_segment_time_s": float(
                    comparison_times[
                        index
                    ]
                ),
                "segment_delta_s": (
                    delta
                ),
                "cumulative_delta_start_s": float(
                    cumulative_delta[
                        index
                    ]
                ),
                "cumulative_delta_end_s": float(
                    cumulative_delta[
                        index + 1
                    ]
                ),
                "classification": (
                    classify_delta(
                        delta
                    )
                ),
            }
        )

    return add_ranks(
        pd.DataFrame(
            records
        )
    )


def safe_corner_summary(
    dataframe: pd.DataFrame,
    classification: str,
) -> tuple[
    float,
    str,
    float,
]:
    """Return corner ID, direction, and delta for the strongest corner."""

    candidates = dataframe.loc[
        (
            dataframe[
                "segment_type"
            ]
            == "CORNER"
        )
        & (
            dataframe[
                "classification"
            ]
            == classification
        )
    ]

    if candidates.empty:
        return (
            math.nan,
            "",
            math.nan,
        )

    if classification == "LOSS":
        index = candidates[
            "segment_delta_s"
        ].idxmax()
    else:
        index = candidates[
            "segment_delta_s"
        ].idxmin()

    row = dataframe.loc[
        index
    ]

    return (
        float(
            row[
                "corner_id"
            ]
        ),
        str(
            row[
                "direction"
            ]
        ),
        float(
            row[
                "segment_delta_s"
            ]
        ),
    )


def build_summary_row(
    *,
    comparison: pd.DataFrame,
    reference_final_elapsed_s: float,
    comparison_final_elapsed_s: float,
    output_file: Path,
) -> dict[str, object]:
    """Build and validate one semantic-comparison summary."""

    final_delta = (
        comparison_final_elapsed_s
        - reference_final_elapsed_s
    )

    segment_sum = float(
        comparison[
            "segment_delta_s"
        ].sum()
    )

    decomposition_error = abs(
        segment_sum
        - final_delta
    )

    if (
        decomposition_error
        > TIMING_TOLERANCE_S
    ):
        raise ValueError(
            "Corner-aware segment deltas do not reconstruct "
            "the full normalized-lap delta."
        )

    corner_rows = comparison.loc[
        comparison[
            "segment_type"
        ]
        == "CORNER"
    ]

    straight_rows = comparison.loc[
        comparison[
            "segment_type"
        ]
        == "STRAIGHT"
    ]

    corner_delta = float(
        corner_rows[
            "segment_delta_s"
        ].sum()
    )

    straight_delta = float(
        straight_rows[
            "segment_delta_s"
        ].sum()
    )

    contribution_error = abs(
        corner_delta
        + straight_delta
        - final_delta
    )

    if (
        contribution_error
        > TIMING_TOLERANCE_S
    ):
        raise ValueError(
            "Corner + straight contribution does not reconstruct "
            "the full normalized-lap delta."
        )

    (
        largest_loss_corner,
        largest_loss_direction,
        largest_loss_delta,
    ) = safe_corner_summary(
        comparison,
        "LOSS",
    )

    (
        largest_gain_corner,
        largest_gain_direction,
        largest_gain_delta,
    ) = safe_corner_summary(
        comparison,
        "GAIN",
    )

    return {
        "session_id": str(
            comparison[
                "session_id"
            ].iloc[0]
        ),
        "capture_id": str(
            comparison[
                "capture_id"
            ].iloc[0]
        ),
        "reference_lap_index": int(
            comparison[
                "reference_lap_index"
            ].iloc[0]
        ),
        "comparison_lap_index": int(
            comparison[
                "comparison_lap_index"
            ].iloc[0]
        ),
        "segment_count": len(
            comparison
        ),
        "corner_count": len(
            corner_rows
        ),
        "straight_count": len(
            straight_rows
        ),
        "common_distance_m": float(
            comparison[
                "end_distance_m"
            ].iloc[-1]
        ),
        "reference_final_elapsed_s": (
            reference_final_elapsed_s
        ),
        "comparison_final_elapsed_s": (
            comparison_final_elapsed_s
        ),
        "final_delta_s": (
            final_delta
        ),
        "segment_delta_sum_s": (
            segment_sum
        ),
        "decomposition_error_s": (
            decomposition_error
        ),
        "corner_delta_total_s": (
            corner_delta
        ),
        "straight_delta_total_s": (
            straight_delta
        ),
        "contribution_error_s": (
            contribution_error
        ),
        "corner_gain_count": int(
            (
                corner_rows[
                    "classification"
                ]
                == "GAIN"
            ).sum()
        ),
        "corner_loss_count": int(
            (
                corner_rows[
                    "classification"
                ]
                == "LOSS"
            ).sum()
        ),
        "corner_neutral_count": int(
            (
                corner_rows[
                    "classification"
                ]
                == "NEUTRAL"
            ).sum()
        ),
        "largest_loss_corner_id": (
            largest_loss_corner
        ),
        "largest_loss_corner_direction": (
            largest_loss_direction
        ),
        "largest_loss_corner_delta_s": (
            largest_loss_delta
        ),
        "largest_gain_corner_id": (
            largest_gain_corner
        ),
        "largest_gain_corner_direction": (
            largest_gain_direction
        ),
        "largest_gain_corner_delta_s": (
            largest_gain_delta
        ),
        "output_csv": (
            project_relative(
                output_file
            )
        ),
        "validation_status": (
            "PASS"
        ),
    }


def build_report(
    summary: pd.DataFrame,
) -> str:
    """Build a concise generation report."""

    lines = [
        "FH6 V1.1 CORNER-AWARE TIMING ANALYSIS BUILD",
        "=" * 72,
        "",
        "COMPARISONS",
        "-" * 72,
    ]

    for _, row in summary.iterrows():
        lines.extend(
            [
                "",
                (
                    f"Lap "
                    f"{int(row['comparison_lap_index']):02d} "
                    f"vs Lap "
                    f"{int(row['reference_lap_index']):02d}"
                ),
                (
                    "  Final delta: "
                    f"{float(row['final_delta_s']):+.9f} s"
                ),
                (
                    "  Segment sum: "
                    f"{float(row['segment_delta_sum_s']):+.9f} s"
                ),
                (
                    "  Decomposition error: "
                    f"{float(row['decomposition_error_s']):.12f} s"
                ),
                (
                    "  Corner contribution: "
                    f"{float(row['corner_delta_total_s']):+.9f} s"
                ),
                (
                    "  Straight contribution: "
                    f"{float(row['straight_delta_total_s']):+.9f} s"
                ),
                (
                    "  Corner + straight error: "
                    f"{float(row['contribution_error_s']):.12f} s"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "STATUS",
            "-" * 72,
            "Corner-aware timing analysis generation: PASS",
        ]
    )

    return "\n".join(lines)


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Apply the V1.1 corner/straight partition to every "
            "validated normalized lap."
        )
    )

    parser.add_argument(
        "--session-id",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--analysis-dir",
        type=Path,
    )

    parser.add_argument(
        "--selection-file",
        type=Path,
    )

    parser.add_argument(
        "--boundary-file",
        type=Path,
    )

    parser.add_argument(
        "--report",
        type=Path,
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_arguments()

    analysis_directory = (
        resolve_project_path(
            args.analysis_dir
        )
        if args.analysis_dir
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
            analysis_directory
            / "reference_lap_selection.csv"
        )
    )

    boundary_file = (
        resolve_project_path(
            args.boundary_file
        )
        if args.boundary_file
        else (
            analysis_directory
            / "corner_segment_boundaries.csv"
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
                "_corner_analysis_build.txt"
            )
        )
    )

    selection = load_selection(
        selection_file,
        args.session_id,
    )

    boundaries = load_boundaries(
        boundary_file,
        args.session_id,
    )

    reference_mask = parse_bool_series(
        selection[
            "is_reference"
        ]
    )

    reference_row = selection.loc[
        reference_mask
    ].iloc[0]

    reference_lap = int(
        reference_row[
            "lap_index"
        ]
    )

    reference_final_elapsed_s = float(
        reference_row[
            "final_elapsed_s"
        ]
    )

    (
        lap_segment_times,
        boundary_times_by_lap,
    ) = build_lap_segment_times(
        selection=selection,
        boundaries=boundaries,
        session_id=args.session_id,
    )

    lap_segment_file = (
        analysis_directory
        / "lap_corner_segment_times.csv"
    )

    comparison_outputs: list[
        tuple[pd.DataFrame, Path]
    ] = []

    summary_rows: list[
        dict[str, object]
    ] = []

    for _, row in selection.loc[
        ~reference_mask
    ].iterrows():
        comparison_lap = int(
            row[
                "lap_index"
            ]
        )

        comparison_file = (
            analysis_directory
            / (
                f"lap_{comparison_lap:02d}"
                f"_vs_lap_{reference_lap:02d}"
                "_corner_analysis.csv"
            )
        )

        comparison = build_comparison(
            session_id=args.session_id,
            capture_id=str(
                row[
                    "capture_id"
                ]
            ),
            boundaries=boundaries,
            reference_lap=(
                reference_lap
            ),
            comparison_lap=(
                comparison_lap
            ),
            reference_boundary_times=(
                boundary_times_by_lap[
                    reference_lap
                ]
            ),
            comparison_boundary_times=(
                boundary_times_by_lap[
                    comparison_lap
                ]
            ),
        )

        summary_rows.append(
            build_summary_row(
                comparison=comparison,
                reference_final_elapsed_s=(
                    reference_final_elapsed_s
                ),
                comparison_final_elapsed_s=float(
                    row[
                        "final_elapsed_s"
                    ]
                ),
                output_file=(
                    comparison_file
                ),
            )
        )

        comparison_outputs.append(
            (
                comparison,
                comparison_file,
            )
        )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        "comparison_lap_index"
    ).reset_index(drop=True)

    summary_file = (
        analysis_directory
        / "corner_analysis_summary.csv"
    )

    # Commit only after all comparisons pass exact reconstruction.
    atomic_csv_write(
        lap_segment_times,
        lap_segment_file,
    )

    for dataframe, output_file in (
        comparison_outputs
    ):
        atomic_csv_write(
            dataframe,
            output_file,
        )

    atomic_csv_write(
        summary,
        summary_file,
    )

    atomic_text_write(
        build_report(
            summary
        ),
        report_file,
    )

    print("=" * 72)
    print(
        "FH6 V1.1 Corner-Aware Timing Analysis Builder"
    )
    print("=" * 72)

    print(
        f"\nReference lap: "
        f"Lap {reference_lap:02d}"
    )

    print(
        f"Track segments: "
        f"{len(boundaries)}"
    )

    print(
        f"Detected corners: "
        f"{int((boundaries['segment_type'] == 'CORNER').sum())}"
    )

    print(
        f"Detected straights: "
        f"{int((boundaries['segment_type'] == 'STRAIGHT').sum())}"
    )

    print()

    for _, row in summary.iterrows():
        print(
            f"Lap "
            f"{int(row['comparison_lap_index']):02d} "
            f"vs Lap "
            f"{int(row['reference_lap_index']):02d}"
        )

        print(
            "  Final delta: "
            f"{float(row['final_delta_s']):+.9f} s"
        )

        print(
            "  Segment sum: "
            f"{float(row['segment_delta_sum_s']):+.9f} s"
        )

        print(
            "  Decomposition error: "
            f"{float(row['decomposition_error_s']):.12f} s"
        )

        print(
            "  Corner contribution: "
            f"{float(row['corner_delta_total_s']):+.9f} s"
        )

        print(
            "  Straight contribution: "
            f"{float(row['straight_delta_total_s']):+.9f} s"
        )

    print(
        "\nCorner-aware timing analysis generation: PASS"
    )

    print(
        f"\nLap-segment times:\n"
        f"{lap_segment_file}"
    )

    print(
        f"\nSummary:\n"
        f"{summary_file}"
    )

    print(
        f"\nBuild report:\n"
        f"{report_file}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
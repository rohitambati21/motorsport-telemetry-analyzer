"""
select_forza_reference_lap.py

Automatically select the fastest validated normalized FH6 lap for V1.1.

Development role
----------------
Formal-Lap Analysis V1.0 intentionally used an explicit manual reference
(Lap 03). V1.1 introduces deterministic automatic reference selection without
changing the frozen V1.0 artifacts.

Selection rule:

    validated normalized laps only
    -> compare elapsed_time_s at the common distance endpoint
    -> lowest elapsed time wins

The V1.1 selector also checks agreement with the frozen V1.0 manual reference.
For the first formal-lap dataset, automatic selection is expected to choose
Lap 03.

No V1.0 files are modified.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

DEFAULT_NORMALIZED_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "data"
    / "forza"
    / "lap_normalized"
)

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

REQUIRED_NORMALIZED_COLUMNS: Final[tuple[str, ...]] = (
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

GRID_TOLERANCE_M: Final[float] = 1e-9
TIE_TOLERANCE_S: Final[float] = 1e-9


def resolve_project_path(path: Path) -> Path:
    """Resolve repository-relative paths."""

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def project_relative(path: Path) -> str:
    """Return a project-relative path where possible."""

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
    """Require a complete input schema."""

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


def summary_path_to_file(
    value: object,
) -> Path:
    """Resolve a file path stored in a generated summary."""

    return resolve_project_path(
        Path(str(value))
    )


def load_normalization_summary(
    normalized_directory: Path,
    expected_laps: int,
    expected_grid_points: int,
) -> pd.DataFrame:
    """Load and validate the frozen normalized-lap summary."""

    summary_file = (
        normalized_directory
        / "lap_normalization_summary.csv"
    )

    if not summary_file.exists():
        raise FileNotFoundError(
            "Normalization summary does not exist: "
            f"{summary_file}"
        )

    summary = pd.read_csv(summary_file)

    required = (
        "session_id",
        "lap_index",
        "lap_id",
        "grid_points",
        "common_distance_m",
        "validation_status",
        "output_csv",
    )

    require_columns(
        summary,
        required,
        "lap_normalization_summary.csv",
    )

    if len(summary) != expected_laps:
        raise ValueError(
            f"Expected {expected_laps} normalized laps; "
            f"found {len(summary)}."
        )

    statuses = (
        summary["validation_status"]
        .astype(str)
        .str.upper()
    )

    if not bool(
        (statuses == "PASS").all()
    ):
        raise ValueError(
            "One or more normalized laps are not validated PASS."
        )

    grids = pd.to_numeric(
        summary["grid_points"],
        errors="coerce",
    )

    if (
        grids.isna().any()
        or not bool(
            (
                grids
                == expected_grid_points
            ).all()
        )
    ):
        raise ValueError(
            "Normalized grid-point count does not match "
            f"expected value {expected_grid_points}."
        )

    return summary.sort_values(
        "lap_index"
    ).reset_index(drop=True)


def load_validated_lap(
    summary_row: pd.Series,
    session_id: str,
    expected_grid_points: int,
) -> tuple[pd.DataFrame, Path]:
    """Load one normalized lap and verify its basic V1.0 contract."""

    source_file = summary_path_to_file(
        summary_row["output_csv"]
    )

    if not source_file.exists():
        raise FileNotFoundError(
            f"Normalized lap file does not exist: {source_file}"
        )

    dataframe = pd.read_csv(source_file)

    require_columns(
        dataframe,
        REQUIRED_NORMALIZED_COLUMNS,
        source_file.name,
    )

    if len(dataframe) != expected_grid_points:
        raise ValueError(
            f"{source_file.name}: expected "
            f"{expected_grid_points} rows; found {len(dataframe)}."
        )

    if (
        str(dataframe["session_id"].iloc[0])
        != session_id
    ):
        raise ValueError(
            f"{source_file.name}: session ID mismatch."
        )

    distance = pd.to_numeric(
        dataframe["distance_m"],
        errors="coerce",
    ).to_numpy(dtype=float)

    elapsed = pd.to_numeric(
        dataframe["elapsed_time_s"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if (
        not np.all(np.isfinite(distance))
        or not np.all(np.isfinite(elapsed))
    ):
        raise ValueError(
            f"{source_file.name}: non-finite time/distance values."
        )

    if np.any(
        np.diff(distance) <= 0.0
    ):
        raise ValueError(
            f"{source_file.name}: distance grid is not strictly increasing."
        )

    if np.any(
        np.diff(elapsed) < -1e-9
    ):
        raise ValueError(
            f"{source_file.name}: elapsed time is not monotonic."
        )

    return dataframe, source_file


def build_reference_selection(
    *,
    session_id: str,
    normalized_directory: Path,
    expected_laps: int,
    expected_grid_points: int,
    expected_v10_reference_lap: int,
) -> pd.DataFrame:
    """Build the deterministic V1.1 reference ranking."""

    summary = load_normalization_summary(
        normalized_directory,
        expected_laps,
        expected_grid_points,
    )

    records: list[dict[str, object]] = []

    reference_grid: np.ndarray | None = None
    capture_ids: set[str] = set()

    for _, row in summary.iterrows():
        lap_index = int(
            row["lap_index"]
        )

        dataframe, source_file = (
            load_validated_lap(
                row,
                session_id,
                expected_grid_points,
            )
        )

        distance = dataframe[
            "distance_m"
        ].to_numpy(dtype=float)

        if reference_grid is None:
            reference_grid = distance
        else:
            if not np.allclose(
                distance,
                reference_grid,
                rtol=0.0,
                atol=GRID_TOLERANCE_M,
            ):
                raise ValueError(
                    "Normalized laps do not share an identical "
                    "distance grid."
                )

        capture_id = str(
            dataframe["capture_id"].iloc[0]
        )

        capture_ids.add(capture_id)

        final_elapsed_s = float(
            dataframe[
                "elapsed_time_s"
            ].iloc[-1]
        )

        records.append(
            {
                "session_id": session_id,
                "capture_id": capture_id,
                "lap_index": lap_index,
                "lap_id": str(
                    dataframe[
                        "lap_id"
                    ].iloc[0]
                ),
                "final_elapsed_s": (
                    final_elapsed_s
                ),
                "common_distance_m": float(
                    distance[-1]
                ),
                "grid_points": len(dataframe),
                "source_lap_csv": (
                    project_relative(
                        source_file
                    )
                ),
            }
        )

    if len(capture_ids) != 1:
        raise ValueError(
            "Normalized laps do not share one capture ID."
        )

    ranking = pd.DataFrame(records)

    ranking = ranking.sort_values(
        [
            "final_elapsed_s",
            "lap_index",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(drop=True)

    ranking["rank"] = (
        np.arange(
            1,
            len(ranking) + 1,
        )
    )

    selected_index = int(
        ranking.iloc[0][
            "lap_index"
        ]
    )

    tie_detected = False

    if len(ranking) >= 2:
        fastest = float(
            ranking.iloc[0][
                "final_elapsed_s"
            ]
        )

        second = float(
            ranking.iloc[1][
                "final_elapsed_s"
            ]
        )

        tie_detected = (
            abs(second - fastest)
            <= TIE_TOLERANCE_S
        )

    ranking["is_reference"] = (
        ranking["lap_index"]
        == selected_index
    )

    agreement = (
        selected_index
        == expected_v10_reference_lap
    )

    ranking[
        "v10_reference_lap"
    ] = expected_v10_reference_lap

    ranking[
        "v10_reference_agreement"
    ] = agreement

    ranking[
        "tie_detected"
    ] = tie_detected

    selected_time = float(
        ranking.iloc[0][
            "final_elapsed_s"
        ]
    )

    selection_reasons: list[str] = []

    for _, row in ranking.iterrows():
        lap_index = int(
            row["lap_index"]
        )

        elapsed = float(
            row["final_elapsed_s"]
        )

        if lap_index == selected_index:
            if tie_detected:
                reason = (
                    "lowest validated normalized endpoint elapsed time; "
                    "timing tie resolved deterministically by lower lap index"
                )
            else:
                reason = (
                    "lowest validated normalized endpoint elapsed time"
                )
        else:
            deficit = (
                elapsed
                - selected_time
            )

            reason = (
                "slower than selected reference by "
                f"{deficit:.9f} s at the common endpoint"
            )

        selection_reasons.append(reason)

    ranking[
        "selection_reason"
    ] = selection_reasons

    ranking[
        "validation_status"
    ] = "PASS"

    return ranking.sort_values(
        "lap_index"
    ).reset_index(drop=True)


def build_report(
    *,
    selection: pd.DataFrame,
    expected_v10_reference_lap: int,
) -> str:
    """Build the human-readable reference-selection report."""

    reference_row = selection.loc[
        selection["is_reference"]
        .astype(bool)
    ].iloc[0]

    selected_lap = int(
        reference_row["lap_index"]
    )

    agreement = bool(
        reference_row[
            "v10_reference_agreement"
        ]
    )

    lines = [
        "FH6 V1.1 AUTOMATIC REFERENCE-LAP SELECTION",
        "=" * 72,
        "",
        "SELECTION RULE",
        "-" * 72,
        (
            "Use validated normalized laps only and select the lap "
            "with the lowest elapsed time at the common distance endpoint."
        ),
        "",
        "RANKING",
        "-" * 72,
    ]

    ranked = selection.sort_values(
        "rank"
    )

    for _, row in ranked.iterrows():
        lines.append(
            (
                f"Rank {int(row['rank'])}: "
                f"Lap {int(row['lap_index']):02d} | "
                f"{float(row['final_elapsed_s']):.9f} s | "
                f"{float(row['common_distance_m']):.3f} m"
            )
        )

    lines.extend(
        [
            "",
            "RESULT",
            "-" * 72,
            (
                f"Automatic reference: "
                f"Lap {selected_lap:02d}"
            ),
            (
                "Frozen V1.0 manual reference: "
                f"Lap {expected_v10_reference_lap:02d}"
            ),
            (
                "V1.0 reference agreement: "
                f"{'PASS' if agreement else 'FAIL'}"
            ),
            "",
            "DEVELOPMENT BOUNDARY",
            "-" * 72,
            (
                "This V1.1 selection artifact is additive. "
                "No frozen V1.0 reference/comparison artifact is modified."
            ),
        ]
    )

    return "\n".join(lines)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Automatically select the fastest validated normalized "
            "FH6 lap for V1.1."
        )
    )

    parser.add_argument(
        "--session-id",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--expected-laps",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--expected-grid-points",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--expected-v10-reference-lap",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--normalized-dir",
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
        "--allow-v10-disagreement",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""

    args = parse_arguments()

    if args.expected_laps < 1:
        raise ValueError(
            "--expected-laps must be positive."
        )

    if args.expected_grid_points < 2:
        raise ValueError(
            "--expected-grid-points must be at least 2."
        )

    if args.expected_v10_reference_lap < 1:
        raise ValueError(
            "--expected-v10-reference-lap must be positive."
        )

    normalized_directory = (
        resolve_project_path(
            args.normalized_dir
        )
        if args.normalized_dir
        else (
            DEFAULT_NORMALIZED_ROOT
            / args.session_id
        )
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

    output_file = (
        output_directory
        / "reference_lap_selection.csv"
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
                "_reference_lap_selection.txt"
            )
        )
    )

    print("=" * 72)
    print("FH6 V1.1 Automatic Reference-Lap Selector")
    print("=" * 72)

    selection = build_reference_selection(
        session_id=args.session_id,
        normalized_directory=(
            normalized_directory
        ),
        expected_laps=args.expected_laps,
        expected_grid_points=(
            args.expected_grid_points
        ),
        expected_v10_reference_lap=(
            args.expected_v10_reference_lap
        ),
    )

    reference_row = selection.loc[
        selection["is_reference"]
        .astype(bool)
    ].iloc[0]

    selected_lap = int(
        reference_row["lap_index"]
    )

    agreement = bool(
        reference_row[
            "v10_reference_agreement"
        ]
    )

    if (
        not agreement
        and not args.allow_v10_disagreement
    ):
        raise ValueError(
            "Automatic reference selection does not agree with "
            "the frozen V1.0 reference."
        )

    atomic_csv_write(
        selection,
        output_file,
    )

    atomic_text_write(
        build_report(
            selection=selection,
            expected_v10_reference_lap=(
                args.expected_v10_reference_lap
            ),
        ),
        report_file,
    )

    print(
        f"\nValidated laps ranked: "
        f"{len(selection)}"
    )

    for _, row in selection.sort_values(
        "rank"
    ).iterrows():
        print(
            f"  Rank {int(row['rank'])}: "
            f"Lap {int(row['lap_index']):02d} | "
            f"{float(row['final_elapsed_s']):.6f} s"
        )

    print(
        f"\nAutomatic reference: "
        f"Lap {selected_lap:02d}"
    )

    print(
        "V1.0 reference agreement: "
        + (
            "PASS"
            if agreement
            else "FAIL"
        )
    )

    print(
        f"\nSelection CSV:\n{output_file}"
    )

    print(
        f"\nReport:\n{report_file}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
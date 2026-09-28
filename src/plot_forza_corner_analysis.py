"""
plot_forza_corner_analysis.py

Generate the focused V1.1 corner-aware visualization package.

Figures
-------
1. reference_corner_detection.png
2. segment_delta_comparison.png
3. corner_delta_comparison.png
4. cumulative_corner_segment_delta.png

The plotter consumes generated V1.1 artifacts only. It never re-detects
corners or recalculates timing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

DEFAULT_V11_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "data"
    / "forza"
    / "lap_analysis_v11"
)

DEFAULT_FIGURE_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "forza_lap_analysis_v11"
)

DEFAULT_REPORT_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "v11"
)

EXPECTED_FIGURES: Final[tuple[str, ...]] = (
    "reference_corner_detection.png",
    "segment_delta_comparison.png",
    "corner_delta_comparison.png",
    "cumulative_corner_segment_delta.png",
)


def resolve_project_path(path: Path) -> Path:
    """Resolve repository-relative paths."""

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def atomic_text_write(
    text: str,
    output_file: Path,
) -> None:
    """Write a text report atomically."""

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


def save_figure(
    figure: plt.Figure,
    output_file: Path,
) -> None:
    """Save one figure consistently."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.tight_layout()

    figure.savefig(
        output_file,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def load_inputs(
    analysis_directory: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[pd.DataFrame],
]:
    """Load all generated corner-analysis artifacts."""

    boundaries_file = (
        analysis_directory
        / "corner_segment_boundaries.csv"
    )

    trace_file = (
        analysis_directory
        / "corner_detection_trace.csv"
    )

    summary_file = (
        analysis_directory
        / "corner_analysis_summary.csv"
    )

    for path in (
        boundaries_file,
        trace_file,
        summary_file,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required plotting input missing: {path}"
            )

    boundaries = pd.read_csv(
        boundaries_file
    )

    trace = pd.read_csv(
        trace_file
    )

    summary = pd.read_csv(
        summary_file
    )

    comparisons: list[
        pd.DataFrame
    ] = []

    for _, row in summary.sort_values(
        "comparison_lap_index"
    ).iterrows():
        comparison_file = (
            resolve_project_path(
                Path(
                    str(
                        row[
                            "output_csv"
                        ]
                    )
                )
            )
        )

        if not comparison_file.exists():
            raise FileNotFoundError(
                f"Corner comparison missing: {comparison_file}"
            )

        comparisons.append(
            pd.read_csv(
                comparison_file
            ).sort_values(
                "segment_id"
            )
        )

    return (
        boundaries,
        trace,
        summary,
        comparisons,
    )


def plot_reference_detection(
    *,
    boundaries: pd.DataFrame,
    trace: pd.DataFrame,
    output_file: Path,
) -> None:
    """Plot speed, steering, smoothed detection signal, and corner spans."""

    figure, speed_axis = plt.subplots(
        figsize=(14, 7)
    )

    distance = trace[
        "distance_m"
    ].to_numpy(dtype=float)

    speed_axis.plot(
        distance,
        trace[
            "speed_kph"
        ],
        label="Reference speed",
        linewidth=1.5,
    )

    speed_axis.set_xlabel(
        "Distance [m]"
    )

    speed_axis.set_ylabel(
        "Speed [km/h]"
    )

    speed_axis.grid(
        True,
        alpha=0.25,
    )

    steering_axis = (
        speed_axis.twinx()
    )

    steering_axis.plot(
        distance,
        trace[
            "steer_normalized"
        ],
        label="Raw steering",
        alpha=0.55,
        linewidth=1.0,
    )

    steering_axis.plot(
        distance,
        trace[
            "smoothed_abs_steer"
        ],
        label="Smoothed |steering|",
        linewidth=1.25,
    )

    enter_threshold = float(
        trace[
            "enter_abs_steer"
        ].iloc[0]
    )

    exit_threshold = float(
        trace[
            "exit_abs_steer"
        ].iloc[0]
    )

    steering_axis.axhline(
        enter_threshold,
        linestyle="--",
        linewidth=1.0,
        label="Corner-entry threshold",
    )

    steering_axis.axhline(
        exit_threshold,
        linestyle=":",
        linewidth=1.0,
        label="Corner-exit threshold",
    )

    steering_axis.set_ylabel(
        "Normalized steering"
    )

    first_corner = True

    for _, row in boundaries.loc[
        boundaries[
            "segment_type"
        ]
        == "CORNER"
    ].iterrows():
        speed_axis.axvspan(
            float(
                row[
                    "start_distance_m"
                ]
            ),
            float(
                row[
                    "end_distance_m"
                ]
            ),
            alpha=0.10,
            hatch="//",
            label=(
                "Detected corner"
                if first_corner
                else None
            ),
        )

        first_corner = False

    speed_axis.set_title(
        "V1.1 Reference-Lap Corner Detection"
    )

    handles_1, labels_1 = (
        speed_axis.get_legend_handles_labels()
    )

    handles_2, labels_2 = (
        steering_axis.get_legend_handles_labels()
    )

    speed_axis.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="upper right",
    )

    save_figure(
        figure,
        output_file,
    )


def plot_segment_delta_comparison(
    *,
    comparisons: list[pd.DataFrame],
    output_file: Path,
) -> None:
    """Plot semantic-segment delta for every comparison."""

    if not comparisons:
        raise ValueError(
            "No corner comparisons available."
        )

    base = comparisons[0]

    x = np.arange(
        len(base),
        dtype=float,
    )

    segment_labels = [
        (
            "C"
            if row[
                "segment_type"
            ]
            == "CORNER"
            else "S"
        )
        + str(
            int(
                row[
                    "segment_id"
                ]
            )
        )
        for _, row in base.iterrows()
    ]

    width = (
        0.8
        / len(comparisons)
    )

    figure, axis = plt.subplots(
        figsize=(14, 7)
    )

    for comparison_position, dataframe in enumerate(
        comparisons
    ):
        comparison_lap = int(
            dataframe[
                "comparison_lap_index"
            ].iloc[0]
        )

        reference_lap = int(
            dataframe[
                "reference_lap_index"
            ].iloc[0]
        )

        offset = (
            comparison_position
            - (
                len(comparisons)
                - 1
            )
            / 2
        ) * width

        bars = axis.bar(
            x + offset,
            dataframe[
                "segment_delta_s"
            ].to_numpy(dtype=float),
            width=width,
            label=(
                f"Lap {comparison_lap:02d} "
                f"vs Lap {reference_lap:02d}"
            ),
        )

        for patch, segment_type in zip(
            bars,
            dataframe[
                "segment_type"
            ],
        ):
            if segment_type == "CORNER":
                patch.set_hatch(
                    "//"
                )

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    axis.set_title(
        "V1.1 Corner/Straight Segment Time Delta"
    )

    axis.set_xlabel(
        "Semantic segment "
        "(C = corner, S = straight)"
    )

    axis.set_ylabel(
        "Segment delta [s]\n"
        "positive = comparison lost time"
    )

    axis.set_xticks(
        x,
        segment_labels,
        rotation=45,
        ha="right",
    )

    axis.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    save_figure(
        figure,
        output_file,
    )


def plot_corner_delta_comparison(
    *,
    comparisons: list[pd.DataFrame],
    output_file: Path,
) -> None:
    """Plot timing deltas for detected corners only."""

    corner_data = [
        dataframe.loc[
            dataframe[
                "segment_type"
            ]
            == "CORNER"
        ].copy()
        for dataframe in comparisons
    ]

    if (
        not corner_data
        or corner_data[0].empty
    ):
        raise ValueError(
            "No detected corners available for plotting."
        )

    base = corner_data[0]

    corner_ids = base[
        "corner_id"
    ].astype(int).to_numpy()

    x = np.arange(
        len(corner_ids),
        dtype=float,
    )

    width = (
        0.8
        / len(corner_data)
    )

    figure, axis = plt.subplots(
        figsize=(13, 6)
    )

    for position, dataframe in enumerate(
        corner_data
    ):
        comparison_lap = int(
            dataframe[
                "comparison_lap_index"
            ].iloc[0]
        )

        reference_lap = int(
            dataframe[
                "reference_lap_index"
            ].iloc[0]
        )

        offset = (
            position
            - (
                len(corner_data)
                - 1
            )
            / 2
        ) * width

        axis.bar(
            x + offset,
            dataframe[
                "segment_delta_s"
            ].to_numpy(dtype=float),
            width=width,
            label=(
                f"Lap {comparison_lap:02d} "
                f"vs Lap {reference_lap:02d}"
            ),
        )

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    axis.set_title(
        "V1.1 Corner-Only Time Delta"
    )

    axis.set_xlabel(
        "Detected corner"
    )

    axis.set_ylabel(
        "Corner delta [s]\n"
        "positive = comparison lost time"
    )

    axis.set_xticks(
        x,
        [
            str(value)
            for value in corner_ids
        ],
    )

    axis.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    save_figure(
        figure,
        output_file,
    )


def plot_cumulative_delta(
    *,
    comparisons: list[pd.DataFrame],
    output_file: Path,
) -> None:
    """Plot cumulative delta at each semantic segment boundary."""

    figure, axis = plt.subplots(
        figsize=(13, 6)
    )

    for dataframe in comparisons:
        comparison_lap = int(
            dataframe[
                "comparison_lap_index"
            ].iloc[0]
        )

        reference_lap = int(
            dataframe[
                "reference_lap_index"
            ].iloc[0]
        )

        segment_ids = dataframe[
            "segment_id"
        ].to_numpy(dtype=int)

        x = np.concatenate(
            (
                np.array(
                    [0],
                    dtype=int,
                ),
                segment_ids,
            )
        )

        y = np.concatenate(
            (
                np.array(
                    [
                        float(
                            dataframe[
                                "cumulative_delta_start_s"
                            ].iloc[0]
                        )
                    ]
                ),
                dataframe[
                    "cumulative_delta_end_s"
                ].to_numpy(dtype=float),
            )
        )

        axis.plot(
            x,
            y,
            marker="o",
            label=(
                f"Lap {comparison_lap:02d} "
                f"vs Lap {reference_lap:02d}"
            ),
        )

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    axis.set_title(
        "V1.1 Cumulative Corner-Aware Segment Delta"
    )

    axis.set_xlabel(
        "Completed semantic segments"
    )

    axis.set_ylabel(
        "Cumulative comparison - reference [s]"
    )

    axis.grid(
        True,
        alpha=0.25,
    )

    axis.legend()

    save_figure(
        figure,
        output_file,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot the V1.1 FH6 corner-aware semantic analysis."
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
        "--figure-dir",
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

    figure_directory = (
        resolve_project_path(
            args.figure_dir
        )
        if args.figure_dir
        else (
            DEFAULT_FIGURE_ROOT
            / args.session_id
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
                "_corner_plotting_summary.txt"
            )
        )
    )

    (
        boundaries,
        trace,
        _,
        comparisons,
    ) = load_inputs(
        analysis_directory
    )

    figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_files = {
        "reference_corner_detection.png": (
            figure_directory
            / "reference_corner_detection.png"
        ),
        "segment_delta_comparison.png": (
            figure_directory
            / "segment_delta_comparison.png"
        ),
        "corner_delta_comparison.png": (
            figure_directory
            / "corner_delta_comparison.png"
        ),
        "cumulative_corner_segment_delta.png": (
            figure_directory
            / "cumulative_corner_segment_delta.png"
        ),
    }

    plot_reference_detection(
        boundaries=boundaries,
        trace=trace,
        output_file=(
            figure_files[
                "reference_corner_detection.png"
            ]
        ),
    )

    plot_segment_delta_comparison(
        comparisons=comparisons,
        output_file=(
            figure_files[
                "segment_delta_comparison.png"
            ]
        ),
    )

    plot_corner_delta_comparison(
        comparisons=comparisons,
        output_file=(
            figure_files[
                "corner_delta_comparison.png"
            ]
        ),
    )

    plot_cumulative_delta(
        comparisons=comparisons,
        output_file=(
            figure_files[
                "cumulative_corner_segment_delta.png"
            ]
        ),
    )

    missing = [
        name
        for name, path in (
            figure_files.items()
        )
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing generated figures: "
            + ", ".join(
                missing
            )
        )

    report_lines = [
        "FH6 V1.1 CORNER-AWARE PLOTTING SUMMARY",
        "=" * 72,
        "",
        f"Session ID: {args.session_id}",
        "",
        "GENERATED FIGURES",
        "-" * 72,
    ]

    for path in (
        figure_files.values()
    ):
        report_lines.append(
            str(path)
        )

    report_lines.extend(
        [
            "",
            "SUMMARY",
            "-" * 72,
            (
                f"Figures generated: "
                f"{len(figure_files)}"
            ),
            (
                f"Figures expected: "
                f"{len(EXPECTED_FIGURES)}"
            ),
            "Plotting status: PASS",
            "",
            (
                "Manual visual QA is specifically required for "
                "reference_corner_detection.png before detector "
                "parameters are treated as validated."
            ),
        ]
    )

    atomic_text_write(
        "\n".join(
            report_lines
        ),
        report_file,
    )

    print("=" * 72)
    print(
        "FH6 V1.1 Corner-Aware Plotter"
    )
    print("=" * 72)

    print(
        f"\nFigures generated: "
        f"{len(figure_files)}"
    )

    print(
        "Plotting status: PASS"
    )

    print(
        f"\nFigure directory:\n"
        f"{figure_directory}"
    )

    print(
        f"\nPlotting report:\n"
        f"{report_file}"
    )

    print()
    print(
        "Manual QA priority: reference_corner_detection.png"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
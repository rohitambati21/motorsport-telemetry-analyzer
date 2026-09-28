"""
plot_forza_lap_comparison.py

Generate a focused visual-validation suite for real FH6 lap comparisons.

Six figures are produced per comparison:

1. speed_vs_distance
2. time_delta_vs_distance
3. throttle_brake_vs_distance
4. steering_vs_distance
5. longitudinal_g_vs_distance
6. gain_loss_regions

For the first validated session there are two comparisons, therefore the
expected output count is twelve figures.

This module only visualizes generated comparison artifacts. It does not
recalculate time delta or alter detector regions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

DEFAULT_COMPARISON_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "data"
    / "forza"
    / "lap_comparisons"
)

DEFAULT_FIGURE_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "forza_lap_comparison"
)

DEFAULT_REPORT_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

FIGURE_SUFFIXES: Final[tuple[str, ...]] = (
    "speed_vs_distance",
    "time_delta_vs_distance",
    "throttle_brake_vs_distance",
    "steering_vs_distance",
    "longitudinal_g_vs_distance",
    "gain_loss_regions",
)


def resolve_project_path(path: Path) -> Path:
    """Resolve a project-relative path."""

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def path_from_summary(value: object) -> Path:
    """Resolve a path stored in the comparison summary."""

    return resolve_project_path(
        Path(str(value))
    )


def atomic_text_write(
    text: str,
    output_file: Path,
) -> None:
    """Write text atomically."""

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
    """Save and close one figure."""

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

    plt.close(figure)


def add_reference_line(
    axis: plt.Axes,
) -> None:
    """Add the zero-delta reference line."""

    axis.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )


def plot_speed(
    dataframe: pd.DataFrame,
    title: str,
    output_file: Path,
) -> None:
    """Plot reference and comparison speed."""

    figure, axis = plt.subplots(
        figsize=(11, 5)
    )

    axis.plot(
        dataframe["distance_m"],
        dataframe["reference_speed_kph"],
        label="Reference",
    )

    axis.plot(
        dataframe["distance_m"],
        dataframe["comparison_speed_kph"],
        label="Comparison",
    )

    axis.set_title(
        f"{title} — Speed vs Distance"
    )

    axis.set_xlabel(
        "Distance [m]"
    )

    axis.set_ylabel(
        "Speed [km/h]"
    )

    axis.grid(True, alpha=0.25)
    axis.legend()

    save_figure(
        figure,
        output_file,
    )


def plot_delta(
    dataframe: pd.DataFrame,
    title: str,
    output_file: Path,
) -> None:
    """Plot raw and smoothed cumulative time delta."""

    figure, axis = plt.subplots(
        figsize=(11, 5)
    )

    axis.plot(
        dataframe["distance_m"],
        dataframe["delta_time_s"],
        label="Raw time delta",
    )

    axis.plot(
        dataframe["distance_m"],
        dataframe["smoothed_delta_time_s"],
        linestyle="--",
        label="V0.3 smoothed delta",
    )

    add_reference_line(axis)

    axis.set_title(
        f"{title} — Time Delta vs Distance"
    )

    axis.set_xlabel(
        "Distance [m]"
    )

    axis.set_ylabel(
        "Comparison - reference [s]"
    )

    axis.grid(True, alpha=0.25)
    axis.legend()

    save_figure(
        figure,
        output_file,
    )


def plot_inputs(
    dataframe: pd.DataFrame,
    title: str,
    output_file: Path,
) -> None:
    """Plot reference/comparison throttle and brake."""

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    axis.plot(
        dataframe["distance_m"],
        dataframe["reference_throttle_pct"],
        label="Reference throttle",
    )

    axis.plot(
        dataframe["distance_m"],
        dataframe["comparison_throttle_pct"],
        label="Comparison throttle",
    )

    axis.plot(
        dataframe["distance_m"],
        dataframe["reference_brake_pct"],
        linestyle="--",
        label="Reference brake",
    )

    axis.plot(
        dataframe["distance_m"],
        dataframe["comparison_brake_pct"],
        linestyle="--",
        label="Comparison brake",
    )

    axis.set_title(
        f"{title} — Throttle / Brake vs Distance"
    )

    axis.set_xlabel(
        "Distance [m]"
    )

    axis.set_ylabel(
        "Input [%]"
    )

    axis.set_ylim(
        -5.0,
        105.0,
    )

    axis.grid(True, alpha=0.25)
    axis.legend(
        ncols=2
    )

    save_figure(
        figure,
        output_file,
    )


def plot_steering(
    dataframe: pd.DataFrame,
    title: str,
    output_file: Path,
) -> None:
    """Plot normalized steering input."""

    figure, axis = plt.subplots(
        figsize=(11, 5)
    )

    axis.plot(
        dataframe["distance_m"],
        dataframe["reference_steer_normalized"],
        label="Reference",
    )

    axis.plot(
        dataframe["distance_m"],
        dataframe["comparison_steer_normalized"],
        label="Comparison",
    )

    add_reference_line(axis)

    axis.set_title(
        f"{title} — Steering vs Distance"
    )

    axis.set_xlabel(
        "Distance [m]"
    )

    axis.set_ylabel(
        "Normalized steering input"
    )

    axis.set_ylim(
        -1.05,
        1.05,
    )

    axis.grid(True, alpha=0.25)
    axis.legend()

    save_figure(
        figure,
        output_file,
    )


def plot_longitudinal_g(
    dataframe: pd.DataFrame,
    title: str,
    output_file: Path,
) -> None:
    """Plot longitudinal acceleration estimate."""

    figure, axis = plt.subplots(
        figsize=(11, 5)
    )

    axis.plot(
        dataframe["distance_m"],
        dataframe["reference_longitudinal_g"],
        label="Reference",
    )

    axis.plot(
        dataframe["distance_m"],
        dataframe["comparison_longitudinal_g"],
        label="Comparison",
    )

    add_reference_line(axis)

    axis.set_title(
        f"{title} — Longitudinal G vs Distance"
    )

    axis.set_xlabel(
        "Distance [m]"
    )

    axis.set_ylabel(
        "Longitudinal acceleration estimate [g]"
    )

    axis.grid(True, alpha=0.25)
    axis.legend()

    save_figure(
        figure,
        output_file,
    )


def plot_gain_loss_regions(
    dataframe: pd.DataFrame,
    regions: pd.DataFrame,
    title: str,
    output_file: Path,
) -> None:
    """Plot cumulative delta with validated V0.3 region overlays."""

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    axis.plot(
        dataframe["distance_m"],
        dataframe["delta_time_s"],
        label="Time delta",
    )

    axis.plot(
        dataframe["distance_m"],
        dataframe["smoothed_delta_time_s"],
        linestyle="--",
        label="Smoothed delta",
    )

    gain_label_used = False
    loss_label_used = False

    for _, region in regions.iterrows():
        classification = str(
            region["classification"]
        ).upper()

        if classification == "GAIN":
            label = (
                "Validated gain region"
                if not gain_label_used
                else None
            )

            axis.axvspan(
                float(
                    region["start_distance_m"]
                ),
                float(
                    region["end_distance_m"]
                ),
                alpha=0.12,
                hatch="//",
                label=label,
            )

            gain_label_used = True

        elif classification == "LOSS":
            label = (
                "Validated loss region"
                if not loss_label_used
                else None
            )

            axis.axvspan(
                float(
                    region["start_distance_m"]
                ),
                float(
                    region["end_distance_m"]
                ),
                alpha=0.12,
                hatch="\\\\",
                label=label,
            )

            loss_label_used = True

    add_reference_line(axis)

    axis.set_title(
        f"{title} — V0.3 Gain / Loss Regions"
    )

    axis.set_xlabel(
        "Distance [m]"
    )

    axis.set_ylabel(
        "Comparison - reference [s]"
    )

    axis.grid(True, alpha=0.25)
    axis.legend()

    save_figure(
        figure,
        output_file,
    )


def run_plotting(
    args: argparse.Namespace,
) -> tuple[Path, ...]:
    """Generate all figures for all comparison rows."""

    comparison_directory = (
        resolve_project_path(
            args.comparison_dir
        )
        if args.comparison_dir
        else (
            DEFAULT_COMPARISON_ROOT
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
                args.session_id
                + "_lap_comparison_plotting_summary.txt"
            )
        )
    )

    summary_file = (
        comparison_directory
        / "lap_comparison_summary.csv"
    )

    if not summary_file.exists():
        raise FileNotFoundError(
            f"Comparison summary does not exist: {summary_file}"
        )

    summary = pd.read_csv(
        summary_file
    )

    if summary.empty:
        raise ValueError(
            "Comparison summary contains no comparisons."
        )

    generated: list[Path] = []

    report_lines = [
        "FH6 REAL-LAP COMPARISON PLOTTING SUMMARY",
        "=" * 72,
        "",
        f"Session ID: {args.session_id}",
        f"Comparisons loaded: {len(summary)}",
        "",
        "GENERATED FIGURES",
        "-" * 72,
    ]

    for _, row in summary.iterrows():
        comparison_id = str(
            row["comparison_id"]
        )

        comparison_file = (
            path_from_summary(
                row["comparison_csv"]
            )
        )

        regions_file = (
            path_from_summary(
                row["regions_csv"]
            )
        )

        comparison = pd.read_csv(
            comparison_file
        )

        regions = pd.read_csv(
            regions_file
        )

        title = comparison_id.replace(
            "_",
            " ",
        ).title()

        file_map = {
            "speed_vs_distance": (
                plot_speed
            ),
            "time_delta_vs_distance": (
                plot_delta
            ),
            "throttle_brake_vs_distance": (
                plot_inputs
            ),
            "steering_vs_distance": (
                plot_steering
            ),
            "longitudinal_g_vs_distance": (
                plot_longitudinal_g
            ),
        }

        for suffix, function in (
            file_map.items()
        ):
            output_file = (
                figure_directory
                / (
                    f"{comparison_id}_"
                    f"{suffix}.png"
                )
            )

            function(
                comparison,
                title,
                output_file,
            )

            generated.append(
                output_file
            )

            report_lines.append(
                str(output_file)
            )

        gain_loss_file = (
            figure_directory
            / (
                f"{comparison_id}"
                "_gain_loss_regions.png"
            )
        )

        plot_gain_loss_regions(
            comparison,
            regions,
            title,
            gain_loss_file,
        )

        generated.append(
            gain_loss_file
        )

        report_lines.append(
            str(gain_loss_file)
        )

    expected_count = (
        len(summary)
        * len(FIGURE_SUFFIXES)
    )

    status = (
        "PASS"
        if len(generated)
        == expected_count
        else "FAIL"
    )

    report_lines.extend(
        [
            "",
            "SUMMARY",
            "-" * 72,
            (
                f"Figures generated: "
                f"{len(generated)}"
            ),
            (
                f"Figures expected: "
                f"{expected_count}"
            ),
            (
                f"Plotting status: "
                f"{status}"
            ),
        ]
    )

    atomic_text_write(
        "\n".join(
            report_lines
        ),
        report_file,
    )

    if status != "PASS":
        raise RuntimeError(
            "Lap-comparison figure generation count failed."
        )

    return tuple(generated)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate visual-validation figures for "
            "real FH6 normalized-lap comparisons."
        )
    )

    parser.add_argument(
        "--session-id",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--comparison-dir",
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
    """Command-line entry point."""

    args = parse_arguments()

    print("=" * 72)
    print("FH6 Real-Lap Comparison Plotter")
    print("=" * 72)

    generated = run_plotting(
        args
    )

    print(
        f"\nSession ID: {args.session_id}"
    )

    print(
        f"Figures generated: {len(generated)}"
    )

    print(
        "Plotting status: PASS"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
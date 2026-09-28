"""
compare_forza_normalized_laps.py

Compare validated, distance-normalized FH6 laps and run the established
V0.3 gain/loss-region detector on real closed-circuit telemetry.

Development role
----------------
This module is the bridge between the validated 400-point formal-lap
normalization pipeline and the earlier V0.3 lap-comparison concepts.

Pipeline position:

    native race telemetry
    -> Candidate Auditor V2
    -> complete-lap extraction
    -> distance normalization
    -> THIS MODULE
    -> figures / report / validation

Important design decisions
--------------------------
- Reference and comparison laps are selected explicitly.
- Time delta is defined as:

      comparison elapsed time - reference elapsed time

  Positive delta therefore means the comparison lap is behind.
  Negative delta means the comparison lap is ahead.

- The established V0.3 detector settings are preserved unchanged.
- No detector threshold tuning is performed from this first real dataset.
- No source telemetry is modified.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
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

DEFAULT_COMPARISON_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "data"
    / "forza"
    / "lap_comparisons"
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

COMPARISON_CHANNELS: Final[tuple[str, ...]] = (
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
class DetectorConfig:
    """Frozen V0.3 gain/loss detector settings."""

    gradient_threshold_s_per_m: float = 0.0001
    smoothing_distance_m: float = 30.0
    neutral_merge_distance_m: float = 25.0
    minimum_region_length_m: float = 25.0
    minimum_net_time_change_s: float = 0.005


@dataclass(frozen=True, slots=True)
class Region:
    """One final gain/loss/neutral distance region."""

    region_id: int
    classification: str

    start_index: int
    end_index: int

    start_distance_m: float
    end_distance_m: float
    length_m: float
    point_count: int

    delta_start_s: float
    delta_end_s: float
    net_delta_change_s: float
    absolute_net_change_s: float

    gradient_mean_s_per_m: float
    gradient_peak_abs_s_per_m: float

    meets_v03_region_thresholds: bool
    interpretation: str


@dataclass(frozen=True, slots=True)
class ComparisonArtifact:
    """Generated outputs for one reference/comparison pair."""

    comparison_id: str

    reference_lap_index: int
    comparison_lap_index: int

    reference_lap_id: str
    comparison_lap_id: str

    reference_final_elapsed_s: float
    comparison_final_elapsed_s: float

    final_delta_s: float
    minimum_delta_s: float
    maximum_delta_s: float

    gain_regions: int
    loss_regions: int
    neutral_regions: int

    largest_gain_region_s: float
    largest_loss_region_s: float

    comparison_csv: Path
    regions_csv: Path


def resolve_project_path(path: Path) -> Path:
    """Resolve a project-relative path."""

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def project_relative(path: Path) -> str:
    """Return a project-relative path whenever possible."""

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
    """Require all expected columns."""

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


def path_from_summary(value: object) -> Path:
    """Resolve a path stored in a generated summary CSV."""

    return resolve_project_path(
        Path(str(value))
    )


def load_normalization_summary(
    normalized_directory: Path,
) -> pd.DataFrame:
    """Read and validate the normalized-lap summary."""

    summary_file = (
        normalized_directory
        / "lap_normalization_summary.csv"
    )

    if not summary_file.exists():
        raise FileNotFoundError(
            "Lap normalization summary does not exist: "
            f"{summary_file}"
        )

    summary = pd.read_csv(summary_file)

    required = (
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

    if summary.empty:
        raise ValueError(
            "Lap normalization summary contains no laps."
        )

    statuses = (
        summary["validation_status"]
        .astype(str)
        .str.upper()
    )

    if not bool((statuses == "PASS").all()):
        raise ValueError(
            "One or more normalized laps are not validated PASS."
        )

    return summary.sort_values(
        "lap_index"
    ).reset_index(drop=True)


def load_lap_by_index(
    summary: pd.DataFrame,
    lap_index: int,
) -> tuple[pd.Series, pd.DataFrame, Path]:
    """Load exactly one validated normalized lap."""

    matches = summary.loc[
        pd.to_numeric(
            summary["lap_index"],
            errors="coerce",
        )
        == lap_index
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one normalized lap with index "
            f"{lap_index}; found {len(matches)}."
        )

    row = matches.iloc[0]

    source_file = path_from_summary(
        row["output_csv"]
    )

    if not source_file.exists():
        raise FileNotFoundError(
            f"Normalized lap file does not exist: {source_file}"
        )

    dataframe = pd.read_csv(
        source_file
    )

    require_columns(
        dataframe,
        REQUIRED_NORMALIZED_COLUMNS,
        source_file.name,
    )

    numeric_columns = (
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

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    required_finite = (
        "distance_m",
        "elapsed_time_s",
        "speed_mps",
        "speed_kph",
        "throttle_pct",
        "brake_pct",
        "steer_normalized",
        "gear",
    )

    for column in required_finite:
        if dataframe[column].isna().any():
            raise ValueError(
                f"{source_file.name}: required channel "
                f"{column} contains NaN values."
            )

    distance = dataframe[
        "distance_m"
    ].to_numpy(dtype=float)

    elapsed = dataframe[
        "elapsed_time_s"
    ].to_numpy(dtype=float)

    if np.any(np.diff(distance) <= 0.0):
        raise ValueError(
            f"{source_file.name}: distance grid is not "
            "strictly increasing."
        )

    if np.any(np.diff(elapsed) < -1e-9):
        raise ValueError(
            f"{source_file.name}: elapsed time is not monotonic."
        )

    return row, dataframe, source_file


def validate_common_grid(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
) -> np.ndarray:
    """Require identical normalized distance grids."""

    reference_distance = reference[
        "distance_m"
    ].to_numpy(dtype=float)

    comparison_distance = comparison[
        "distance_m"
    ].to_numpy(dtype=float)

    if len(reference_distance) != len(
        comparison_distance
    ):
        raise ValueError(
            "Reference and comparison laps contain different "
            "numbers of distance-grid points."
        )

    if not np.allclose(
        reference_distance,
        comparison_distance,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "Reference and comparison distance grids are not identical."
        )

    return reference_distance


def smoothing_window_points(
    distance: np.ndarray,
    smoothing_distance_m: float,
) -> int:
    """
    Convert a physical smoothing distance into an odd rolling window size.
    """

    differences = np.diff(distance)

    positive = differences[
        differences > 0.0
    ]

    if len(positive) == 0:
        raise ValueError(
            "Cannot determine grid spacing for detector smoothing."
        )

    spacing = float(
        np.median(positive)
    )

    raw_points = max(
        1,
        int(
            round(
                smoothing_distance_m
                / spacing
            )
        ),
    )

    if raw_points % 2 == 0:
        raw_points += 1

    return raw_points


def smooth_delta(
    distance: np.ndarray,
    delta: np.ndarray,
    smoothing_distance_m: float,
) -> np.ndarray:
    """Apply the V0.3 centered distance-window smoothing."""

    window = smoothing_window_points(
        distance,
        smoothing_distance_m,
    )

    smoothed = (
        pd.Series(delta)
        .rolling(
            window=window,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy(dtype=float)
    )

    return smoothed


def classify_gradient(
    gradient: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Classify delta gradient.

    Negative gradient:
        comparison lap is reducing its deficit -> GAIN

    Positive gradient:
        comparison lap is increasing its deficit -> LOSS
    """

    labels = np.full(
        len(gradient),
        "NEUTRAL",
        dtype=object,
    )

    labels[
        gradient <= -abs(threshold)
    ] = "GAIN"

    labels[
        gradient >= abs(threshold)
    ] = "LOSS"

    return labels


def contiguous_segments(
    labels: np.ndarray,
) -> list[tuple[str, int, int]]:
    """Convert point labels into contiguous index segments."""

    if len(labels) == 0:
        return []

    segments: list[
        tuple[str, int, int]
    ] = []

    start = 0
    current = str(labels[0])

    for index in range(
        1,
        len(labels),
    ):
        label = str(labels[index])

        if label != current:
            segments.append(
                (
                    current,
                    start,
                    index - 1,
                )
            )

            start = index
            current = label

    segments.append(
        (
            current,
            start,
            len(labels) - 1,
        )
    )

    return segments


def segment_length(
    distance: np.ndarray,
    start_index: int,
    end_index: int,
) -> float:
    """Return physical length of one segment."""

    return float(
        distance[end_index]
        - distance[start_index]
    )


def merge_short_neutral_segments(
    *,
    labels: np.ndarray,
    distance: np.ndarray,
    maximum_neutral_length_m: float,
) -> np.ndarray:
    """
    Merge short neutral gaps only when both neighboring regions have the
    same non-neutral classification.

    This prevents a small threshold dead-zone from splitting one genuine
    gain or loss region while avoiding arbitrary gain/loss boundary changes.
    """

    merged = labels.copy()

    changed = True

    while changed:
        changed = False

        segments = contiguous_segments(
            merged
        )

        for segment_index, (
            classification,
            start_index,
            end_index,
        ) in enumerate(segments):
            if classification != "NEUTRAL":
                continue

            if (
                segment_index == 0
                or segment_index
                == len(segments) - 1
            ):
                continue

            length_m = segment_length(
                distance,
                start_index,
                end_index,
            )

            if (
                length_m
                > maximum_neutral_length_m
            ):
                continue

            previous_class = segments[
                segment_index - 1
            ][0]

            next_class = segments[
                segment_index + 1
            ][0]

            if (
                previous_class == next_class
                and previous_class
                in {"GAIN", "LOSS"}
            ):
                merged[
                    start_index:
                    end_index + 1
                ] = previous_class

                changed = True
                break

    return merged


def apply_region_quality_gates(
    *,
    labels: np.ndarray,
    distance: np.ndarray,
    raw_delta: np.ndarray,
    config: DetectorConfig,
) -> np.ndarray:
    """
    Enforce V0.3 minimum physical-region and net-time-change gates.

    A region must also have a net delta change consistent with its label:
    - GAIN -> negative net delta change
    - LOSS -> positive net delta change
    """

    final = labels.copy()

    for (
        classification,
        start_index,
        end_index,
    ) in contiguous_segments(labels):
        if classification not in {
            "GAIN",
            "LOSS",
        }:
            continue

        length_m = segment_length(
            distance,
            start_index,
            end_index,
        )

        net_change = float(
            raw_delta[end_index]
            - raw_delta[start_index]
        )

        sign_consistent = (
            (
                classification == "GAIN"
                and net_change < 0.0
            )
            or (
                classification == "LOSS"
                and net_change > 0.0
            )
        )

        valid = (
            length_m
            >= config.minimum_region_length_m
            and abs(net_change)
            >= config.minimum_net_time_change_s
            and sign_consistent
        )

        if not valid:
            final[
                start_index:
                end_index + 1
            ] = "NEUTRAL"

    return final


def build_regions(
    *,
    labels: np.ndarray,
    distance: np.ndarray,
    raw_delta: np.ndarray,
    gradient: np.ndarray,
    config: DetectorConfig,
) -> tuple[Region, ...]:
    """Build the final machine-readable region table."""

    regions: list[Region] = []

    for region_id, (
        classification,
        start_index,
        end_index,
    ) in enumerate(
        contiguous_segments(labels),
        start=1,
    ):
        start_distance = float(
            distance[start_index]
        )

        end_distance = float(
            distance[end_index]
        )

        length_m = (
            end_distance
            - start_distance
        )

        delta_start = float(
            raw_delta[start_index]
        )

        delta_end = float(
            raw_delta[end_index]
        )

        net_change = (
            delta_end
            - delta_start
        )

        region_gradient = gradient[
            start_index:
            end_index + 1
        ]

        gradient_mean = float(
            np.mean(region_gradient)
        )

        gradient_peak = float(
            np.max(
                np.abs(region_gradient)
            )
        )

        meets_thresholds = (
            classification
            in {"GAIN", "LOSS"}
            and length_m
            >= config.minimum_region_length_m
            and abs(net_change)
            >= config.minimum_net_time_change_s
        )

        if classification == "GAIN":
            interpretation = (
                "Comparison lap gains time relative to the reference "
                "over this section; cumulative delta decreases."
            )

        elif classification == "LOSS":
            interpretation = (
                "Comparison lap loses time relative to the reference "
                "over this section; cumulative delta increases."
            )

        else:
            interpretation = (
                "No validated sustained gain/loss region under the "
                "frozen V0.3 thresholds."
            )

        regions.append(
            Region(
                region_id=region_id,
                classification=classification,
                start_index=start_index,
                end_index=end_index,
                start_distance_m=start_distance,
                end_distance_m=end_distance,
                length_m=length_m,
                point_count=(
                    end_index
                    - start_index
                    + 1
                ),
                delta_start_s=delta_start,
                delta_end_s=delta_end,
                net_delta_change_s=(
                    net_change
                ),
                absolute_net_change_s=(
                    abs(net_change)
                ),
                gradient_mean_s_per_m=(
                    gradient_mean
                ),
                gradient_peak_abs_s_per_m=(
                    gradient_peak
                ),
                meets_v03_region_thresholds=(
                    meets_thresholds
                ),
                interpretation=interpretation,
            )
        )

    return tuple(regions)


def largest_gain(
    regions: Sequence[Region],
) -> float:
    """Return the most negative validated gain-region net change."""

    values = [
        region.net_delta_change_s
        for region in regions
        if region.classification == "GAIN"
    ]

    return (
        min(values)
        if values
        else math.nan
    )


def largest_loss(
    regions: Sequence[Region],
) -> float:
    """Return the largest positive validated loss-region net change."""

    values = [
        region.net_delta_change_s
        for region in regions
        if region.classification == "LOSS"
    ]

    return (
        max(values)
        if values
        else math.nan
    )


def build_comparison_dataframe(
    *,
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    reference_lap_index: int,
    comparison_lap_index: int,
    config: DetectorConfig,
) -> tuple[
    pd.DataFrame,
    tuple[Region, ...],
]:
    """Calculate time delta, detector features, and final region labels."""

    distance = validate_common_grid(
        reference,
        comparison,
    )

    reference_time = reference[
        "elapsed_time_s"
    ].to_numpy(dtype=float)

    comparison_time = comparison[
        "elapsed_time_s"
    ].to_numpy(dtype=float)

    delta = (
        comparison_time
        - reference_time
    )

    smoothed_delta = smooth_delta(
        distance,
        delta,
        config.smoothing_distance_m,
    )

    gradient = np.gradient(
        smoothed_delta,
        distance,
    )

    labels = classify_gradient(
        gradient,
        config.gradient_threshold_s_per_m,
    )

    labels = merge_short_neutral_segments(
        labels=labels,
        distance=distance,
        maximum_neutral_length_m=(
            config.neutral_merge_distance_m
        ),
    )

    labels = apply_region_quality_gates(
        labels=labels,
        distance=distance,
        raw_delta=delta,
        config=config,
    )

    regions = build_regions(
        labels=labels,
        distance=distance,
        raw_delta=delta,
        gradient=gradient,
        config=config,
    )

    dataframe = pd.DataFrame(
        {
            "distance_m": distance,
            "reference_lap_index": (
                reference_lap_index
            ),
            "comparison_lap_index": (
                comparison_lap_index
            ),
            "reference_elapsed_time_s": (
                reference_time
            ),
            "comparison_elapsed_time_s": (
                comparison_time
            ),
            "delta_time_s": delta,
            "smoothed_delta_time_s": (
                smoothed_delta
            ),
            "delta_gradient_s_per_m": (
                gradient
            ),
            "region_classification": (
                labels
            ),
        }
    )

    for channel in COMPARISON_CHANNELS:
        dataframe[
            f"reference_{channel}"
        ] = reference[
            channel
        ].to_numpy()

        dataframe[
            f"comparison_{channel}"
        ] = comparison[
            channel
        ].to_numpy()

    return dataframe, regions


def run_comparisons(
    args: argparse.Namespace,
) -> tuple[
    ComparisonArtifact,
    ...,
]:
    """Execute all explicit reference/comparison pairs."""

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
            DEFAULT_COMPARISON_ROOT
            / args.session_id
        )
    )

    summary = load_normalization_summary(
        normalized_directory
    )

    (
        reference_summary_row,
        reference,
        reference_file,
    ) = load_lap_by_index(
        summary,
        args.reference_lap,
    )

    if len(reference) != args.expected_grid_points:
        raise ValueError(
            "Reference lap grid-point count does not match "
            f"expected value {args.expected_grid_points}."
        )

    reference_lap_id = str(
        reference["lap_id"].iloc[0]
    )

    reference_session = str(
        reference["session_id"].iloc[0]
    )

    if reference_session != args.session_id:
        raise ValueError(
            "Reference lap session ID does not match --session-id."
        )

    config = DetectorConfig()

    artifacts: list[
        ComparisonArtifact
    ] = []

    for comparison_lap_index in (
        args.comparison_laps
    ):
        if (
            comparison_lap_index
            == args.reference_lap
        ):
            raise ValueError(
                "Reference lap cannot also appear in --comparison-laps."
            )

        (
            comparison_summary_row,
            comparison,
            comparison_file,
        ) = load_lap_by_index(
            summary,
            comparison_lap_index,
        )

        if (
            len(comparison)
            != args.expected_grid_points
        ):
            raise ValueError(
                f"Comparison lap {comparison_lap_index} does not "
                "match expected grid-point count."
            )

        if (
            str(
                comparison[
                    "session_id"
                ].iloc[0]
            )
            != args.session_id
        ):
            raise ValueError(
                "Comparison lap session ID does not match --session-id."
            )

        comparison_lap_id = str(
            comparison["lap_id"].iloc[0]
        )

        comparison_id = (
            f"lap_{comparison_lap_index:02d}"
            f"_vs_lap_{args.reference_lap:02d}"
        )

        (
            comparison_dataframe,
            regions,
        ) = build_comparison_dataframe(
            reference=reference,
            comparison=comparison,
            reference_lap_index=(
                args.reference_lap
            ),
            comparison_lap_index=(
                comparison_lap_index
            ),
            config=config,
        )

        comparison_dataframe.insert(
            0,
            "comparison_id",
            comparison_id,
        )

        comparison_dataframe.insert(
            1,
            "session_id",
            args.session_id,
        )

        comparison_dataframe.insert(
            2,
            "reference_lap_id",
            reference_lap_id,
        )

        comparison_dataframe.insert(
            3,
            "comparison_lap_id",
            comparison_lap_id,
        )

        comparison_csv = (
            output_directory
            / f"{comparison_id}_comparison.csv"
        )

        regions_csv = (
            output_directory
            / f"{comparison_id}_regions.csv"
        )

        region_dataframe = pd.DataFrame(
            asdict(region)
            for region in regions
        )

        if region_dataframe.empty:
            raise ValueError(
                f"{comparison_id}: detector produced no regions."
            )

        gain_regions = sum(
            region.classification == "GAIN"
            for region in regions
        )

        loss_regions = sum(
            region.classification == "LOSS"
            for region in regions
        )

        neutral_regions = sum(
            region.classification
            == "NEUTRAL"
            for region in regions
        )

        reference_final = float(
            reference[
                "elapsed_time_s"
            ].iloc[-1]
        )

        comparison_final = float(
            comparison[
                "elapsed_time_s"
            ].iloc[-1]
        )

        final_delta = float(
            comparison_dataframe[
                "delta_time_s"
            ].iloc[-1]
        )

        artifact = ComparisonArtifact(
            comparison_id=comparison_id,
            reference_lap_index=(
                args.reference_lap
            ),
            comparison_lap_index=(
                comparison_lap_index
            ),
            reference_lap_id=(
                reference_lap_id
            ),
            comparison_lap_id=(
                comparison_lap_id
            ),
            reference_final_elapsed_s=(
                reference_final
            ),
            comparison_final_elapsed_s=(
                comparison_final
            ),
            final_delta_s=final_delta,
            minimum_delta_s=float(
                comparison_dataframe[
                    "delta_time_s"
                ].min()
            ),
            maximum_delta_s=float(
                comparison_dataframe[
                    "delta_time_s"
                ].max()
            ),
            gain_regions=gain_regions,
            loss_regions=loss_regions,
            neutral_regions=(
                neutral_regions
            ),
            largest_gain_region_s=(
                largest_gain(regions)
            ),
            largest_loss_region_s=(
                largest_loss(regions)
            ),
            comparison_csv=(
                comparison_csv
            ),
            regions_csv=regions_csv,
        )

        atomic_csv_write(
            comparison_dataframe,
            comparison_csv,
        )

        atomic_csv_write(
            region_dataframe,
            regions_csv,
        )

        artifacts.append(artifact)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_file = (
        output_directory
        / "lap_comparison_summary.csv"
    )

    common_distance = float(
        reference[
            "distance_m"
        ].iloc[-1]
    )

    summary_dataframe = pd.DataFrame(
        [
            {
                "session_id": args.session_id,
                "comparison_id": (
                    artifact.comparison_id
                ),
                "reference_lap_index": (
                    artifact.reference_lap_index
                ),
                "comparison_lap_index": (
                    artifact.comparison_lap_index
                ),
                "reference_lap_id": (
                    artifact.reference_lap_id
                ),
                "comparison_lap_id": (
                    artifact.comparison_lap_id
                ),
                "reference_final_elapsed_s": (
                    artifact.reference_final_elapsed_s
                ),
                "comparison_final_elapsed_s": (
                    artifact.comparison_final_elapsed_s
                ),
                "final_delta_s": (
                    artifact.final_delta_s
                ),
                "minimum_delta_s": (
                    artifact.minimum_delta_s
                ),
                "maximum_delta_s": (
                    artifact.maximum_delta_s
                ),
                "gain_regions": (
                    artifact.gain_regions
                ),
                "loss_regions": (
                    artifact.loss_regions
                ),
                "neutral_regions": (
                    artifact.neutral_regions
                ),
                "largest_gain_region_s": (
                    artifact.largest_gain_region_s
                ),
                "largest_loss_region_s": (
                    artifact.largest_loss_region_s
                ),
                "grid_points": len(reference),
                "common_distance_m": (
                    common_distance
                ),
                "gradient_threshold_s_per_m": (
                    config.gradient_threshold_s_per_m
                ),
                "smoothing_distance_m": (
                    config.smoothing_distance_m
                ),
                "neutral_merge_distance_m": (
                    config.neutral_merge_distance_m
                ),
                "minimum_region_length_m": (
                    config.minimum_region_length_m
                ),
                "minimum_net_time_change_s": (
                    config.minimum_net_time_change_s
                ),
                "reference_csv": (
                    project_relative(
                        reference_file
                    )
                ),
                "comparison_source_csv": (
                    project_relative(
                        path_from_summary(
                            comparison_summary_row[
                                "output_csv"
                            ]
                        )
                    )
                ),
                "comparison_csv": (
                    project_relative(
                        artifact.comparison_csv
                    )
                ),
                "regions_csv": (
                    project_relative(
                        artifact.regions_csv
                    )
                ),
                "comparison_validation_status": (
                    "PASS"
                ),
            }
            for artifact in artifacts
        ]
    )

    atomic_csv_write(
        summary_dataframe,
        summary_file,
    )

    return tuple(artifacts)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare validated normalized FH6 laps using "
            "the frozen V0.3 gain/loss detector."
        )
    )

    parser.add_argument(
        "--session-id",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--reference-lap",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--comparison-laps",
        type=int,
        nargs="+",
        default=[1, 2],
    )

    parser.add_argument(
        "--expected-grid-points",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--normalized-dir",
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
    )

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""

    args = parse_arguments()

    if args.reference_lap < 1:
        raise ValueError(
            "--reference-lap must be positive."
        )

    if any(
        lap < 1
        for lap in args.comparison_laps
    ):
        raise ValueError(
            "--comparison-laps values must be positive."
        )

    if args.expected_grid_points < 2:
        raise ValueError(
            "--expected-grid-points must be at least 2."
        )

    print("=" * 72)
    print("FH6 Real Normalized-Lap Comparison — V0.3")
    print("=" * 72)

    artifacts = run_comparisons(
        args
    )

    print(
        f"\nSession ID: {args.session_id}"
    )

    print(
        f"Reference lap: {args.reference_lap:02d}"
    )

    print(
        f"Real-lap comparisons generated: {len(artifacts)}"
    )

    for artifact in artifacts:
        print(
            f"\n{artifact.comparison_id}"
        )

        print(
            "  Reference final time: "
            f"{artifact.reference_final_elapsed_s:.6f} s"
        )

        print(
            "  Comparison final time: "
            f"{artifact.comparison_final_elapsed_s:.6f} s"
        )

        print(
            "  Final delta: "
            f"{artifact.final_delta_s:+.6f} s"
        )

        print(
            "  Regions: "
            f"{artifact.gain_regions} gain | "
            f"{artifact.loss_regions} loss | "
            f"{artifact.neutral_regions} neutral"
        )

        print(
            f"  Comparison CSV: {artifact.comparison_csv}"
        )

        print(
            f"  Regions CSV: {artifact.regions_csv}"
        )

    print(
        "\nReal normalized-lap comparison: PASS"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
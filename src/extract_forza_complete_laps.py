"""
extract_forza_complete_laps.py

Extract clean complete laps from a validated FH6 formal-lap capture.

Development role
----------------
This script is the first transformation layer after Candidate Auditor V2.

It intentionally reuses the boundary-aware audit engine rather than creating
a second lap detector. The same validated boundaries that qualified a
candidate are therefore used to produce the standalone lap artifacts.

Pipeline position:

    native FH6 capture
    -> Candidate Auditor V2 PASS
    -> complete-lap extraction
    -> distance normalization
    -> future V0.3 lap comparison

Safety / integrity rules
------------------------
- Native telemetry is read-only.
- Capture must already have PASS status in the formal-lap manifest.
- Boundary-aware V2 audit is rerun before extraction.
- Only clean qualified complete intervals are exported.
- Raw source columns are preserved with a ``raw__`` prefix.
- Canonical engineering fields are added for downstream analysis.
- No smoothing, interpolation, or gain/loss analysis occurs here.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd

from audit_forza_lap_readiness import (
    AuditConfig,
    DetailedSessionAudit,
    LapIntervalAudit,
    audit_one_session_detailed,
    build_audit_time_axis,
    numeric_series,
    resolve_column,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST: Final[Path] = (
    PROJECT_ROOT
    / "data"
    / "forza"
    / "lap_captures"
    / "forza_lap_capture_manifest.csv"
)

DEFAULT_LAP_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "data"
    / "forza"
    / "laps"
)

DEFAULT_REPORT_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

G_ACCELERATION: Final[float] = 9.80665

EXTRACTION_TIME_TOLERANCE_S: Final[float] = 0.100
EXTRACTION_DISTANCE_TOLERANCE_M: Final[float] = 5.0

LAT_G_ALIASES: Final[tuple[str, ...]] = (
    "lat_g",
    "lateral_g",
    "lateral_acceleration_g",
)

LON_G_ALIASES: Final[tuple[str, ...]] = (
    "lon_g",
    "longitudinal_g",
    "longitudinal_acceleration_g",
)

LAT_ACCEL_ALIASES: Final[tuple[str, ...]] = (
    "acceleration_x_mps2",
    "accel_x_mps2",
    "lateral_acceleration_mps2",
)

LON_ACCEL_ALIASES: Final[tuple[str, ...]] = (
    "acceleration_z_mps2",
    "accel_z_mps2",
    "longitudinal_acceleration_mps2",
)

EXPLICIT_THROTTLE_PCT_ALIASES: Final[tuple[str, ...]] = (
    "throttle_pct",
    "accelerator_pct",
)

EXPLICIT_BRAKE_PCT_ALIASES: Final[tuple[str, ...]] = (
    "brake_pct",
)

EXPLICIT_STEER_NORM_ALIASES: Final[tuple[str, ...]] = (
    "steering_input_norm",
    "steer_normalized",
    "steering_normalized",
)


@dataclass(frozen=True, slots=True)
class ExtractionArtifact:
    """One validated extracted lap."""

    lap_index: int
    lap_id: str
    source_lap_number: str

    output_file: Path

    start_row: int
    end_row: int
    row_count: int

    audit_duration_s: float
    extracted_duration_s: float
    duration_error_s: float

    audit_distance_m: float
    extracted_distance_m: float
    distance_error_m: float

    required_channel_coverage_pct: float
    validation_status: str


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Complete extraction-run result."""

    session_id: str
    capture_id: str
    output_directory: Path
    summary_file: Path
    report_file: Path
    artifacts: tuple[ExtractionArtifact, ...]


def clean_text(value: object) -> str:
    """Return stripped text from mixed CSV values."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass

    return str(value).strip()


def resolve_project_path(path: Path) -> Path:
    """Resolve relative paths from the project root."""

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


def normalized_name(value: str) -> str:
    """Normalize a CSV field name for simple alias matching."""

    return "".join(
        character
        for character in value.lower()
        if character.isalnum()
    )


def find_column(
    dataframe: pd.DataFrame,
    aliases: Sequence[str],
) -> str:
    """Resolve the first matching column from a local alias set."""

    available = {
        normalized_name(str(column)): str(column)
        for column in dataframe.columns
    }

    for alias in aliases:
        match = available.get(
            normalized_name(alias)
        )

        if match:
            return match

    return ""


def numeric_column(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a numeric column or a NaN series."""

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


def finite_start_value(
    series: pd.Series,
    name: str,
) -> float:
    """Return the first finite value in a lap segment."""

    valid = series.dropna()

    if valid.empty:
        raise ValueError(
            f"Cannot derive {name}; no finite values exist."
        )

    return float(valid.iloc[0])


def percentage_from_source(
    series: pd.Series,
) -> pd.Series:
    """
    Convert a throttle/brake signal to percent.

    Supports:
    - normalized 0..1
    - percent 0..100
    - FH6 raw 0..255
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).astype(float)

    valid = numeric.dropna()

    if valid.empty:
        return numeric

    maximum = float(
        valid.abs().max()
    )

    if maximum <= 1.5:
        result = numeric * 100.0

    elif maximum <= 100.5:
        result = numeric

    else:
        result = numeric / 255.0 * 100.0

    return result.clip(
        lower=0.0,
        upper=100.0,
    )


def steering_from_source(
    series: pd.Series,
) -> pd.Series:
    """
    Convert steering to approximately -1..1.

    Supports normalized steering directly and FH6 raw -127..127 input.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).astype(float)

    valid = numeric.dropna()

    if valid.empty:
        return numeric

    maximum = float(
        valid.abs().max()
    )

    if maximum <= 1.5:
        result = numeric

    else:
        result = numeric / 127.0

    return result.clip(
        lower=-1.0,
        upper=1.0,
    )


def resolve_g_channel(
    dataframe: pd.DataFrame,
    *,
    direct_aliases: Sequence[str],
    acceleration_aliases: Sequence[str],
) -> pd.Series:
    """
    Resolve an existing g channel or derive it from m/s^2 acceleration.
    """

    direct_column = find_column(
        dataframe,
        direct_aliases,
    )

    if direct_column:
        return numeric_column(
            dataframe,
            direct_column,
        )

    acceleration_column = find_column(
        dataframe,
        acceleration_aliases,
    )

    if acceleration_column:
        return (
            numeric_column(
                dataframe,
                acceleration_column,
            )
            / G_ACCELERATION
        )

    return pd.Series(
        np.nan,
        index=dataframe.index,
        dtype=float,
    )


def read_manifest(
    manifest_file: Path,
) -> pd.DataFrame:
    """Load the formal-lap capture manifest."""

    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Formal-lap manifest does not exist: {manifest_file}"
        )

    manifest = pd.read_csv(
        manifest_file,
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "capture_id",
        "audit_status",
        "actual_native_filename",
        "actual_native_path",
    }

    missing = required.difference(
        manifest.columns
    )

    if missing:
        raise ValueError(
            "Formal-lap manifest is missing columns: "
            + ", ".join(sorted(missing))
        )

    return manifest


def validate_manifest_capture(
    *,
    manifest: pd.DataFrame,
    capture_id: str,
    native_file: Path,
) -> pd.Series:
    """
    Require one manifest row with a validated PASS result.
    """

    matches = manifest.loc[
        manifest["capture_id"].map(clean_text)
        == capture_id
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one manifest row for {capture_id}; "
            f"found {len(matches)}."
        )

    row = matches.iloc[0]

    audit_status = clean_text(
        row["audit_status"]
    ).upper()

    if audit_status != "PASS":
        raise ValueError(
            f"Capture {capture_id} is not approved for extraction. "
            f"Manifest audit_status={audit_status or 'blank'}."
        )

    recorded_filename = clean_text(
        row["actual_native_filename"]
    )

    if (
        recorded_filename
        and recorded_filename
        != native_file.name
    ):
        raise ValueError(
            "Input native CSV does not match the file registered "
            f"for {capture_id}: {recorded_filename}"
        )

    return row


def build_audit_config(
    native_file: Path,
    minimum_clean_laps: int,
) -> AuditConfig:
    """Build a non-writing configuration for the shared V2 audit engine."""

    return AuditConfig(
        catalog_file=Path(""),
        native_directory=native_file.parent,
        summary_file=Path(""),
        report_file=Path(""),
        minimum_complete_laps=minimum_clean_laps,
    )


def rerun_boundary_audit(
    *,
    native_file: Path,
    capture_id: str,
    minimum_clean_laps: int,
) -> DetailedSessionAudit:
    """Reconfirm the source using the same V2 boundary logic."""

    synthetic_catalog_row = pd.Series(
        {
            "session_id": native_file.stem,
            "short_label": capture_id,
            "display_name": "Validated Formal-Lap Capture",
            "native_csv": str(native_file),
        }
    )

    detailed = audit_one_session_detailed(
        catalog_row=synthetic_catalog_row,
        config=build_audit_config(
            native_file,
            minimum_clean_laps,
        ),
    )

    audit = detailed.summary

    if (
        audit.qualified_complete_lap_candidates
        < minimum_clean_laps
    ):
        raise ValueError(
            "Boundary-aware source audit no longer finds enough clean "
            "complete laps for extraction."
        )

    if (
        audit.in_lap_capture_time_jumps
        or audit.in_lap_race_time_jumps
        or audit.in_lap_distance_jumps
    ):
        raise ValueError(
            "Boundary-aware source audit reports an in-lap "
            "telemetry discontinuity."
        )

    return detailed


def required_coverage_pct(
    dataframe: pd.DataFrame,
) -> float:
    """Return the lowest coverage among required canonical channels."""

    required = (
        "lap_elapsed_time_s",
        "lap_distance_m",
        "speed_mps",
        "speed_kph",
        "throttle_pct",
        "brake_pct",
        "steer_normalized",
        "gear",
    )

    coverages = [
        float(
            dataframe[column]
            .notna()
            .mean()
            * 100.0
        )
        for column in required
    ]

    return min(coverages)


def validate_monotonic(
    series: pd.Series,
    name: str,
    tolerance: float = 1e-9,
) -> None:
    """Require a non-decreasing finite signal."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    if np.any(
        np.diff(values) < -abs(tolerance)
    ):
        raise ValueError(
            f"{name} is not monotonic."
        )


def build_canonical_lap(
    *,
    native: pd.DataFrame,
    detailed_audit: DetailedSessionAudit,
    interval: LapIntervalAudit,
    lap_index: int,
    capture_id: str,
) -> pd.DataFrame:
    """
    Build one standalone lap while retaining raw source channels.
    """

    audit = detailed_audit.summary

    source = native.iloc[
        interval.start_position:
        interval.end_position + 1
    ].copy()

    if source.empty:
        raise ValueError(
            f"Lap {lap_index} extraction produced no rows."
        )

    time_axis = build_audit_time_axis(
        dataframe=native,
        capture_time_column=audit.capture_time_column,
        current_race_time_column=audit.current_race_time_column,
    ).elapsed_seconds

    lap_time_source = time_axis.iloc[
        interval.start_position:
        interval.end_position + 1
    ].reset_index(drop=True)

    time_origin = finite_start_value(
        lap_time_source,
        "lap time origin",
    )

    lap_elapsed_time = (
        lap_time_source
        - time_origin
    )

    distance_source = numeric_series(
        native,
        audit.distance_column,
    ).iloc[
        interval.start_position:
        interval.end_position + 1
    ].reset_index(drop=True)

    distance_origin = finite_start_value(
        distance_source,
        "lap distance origin",
    )

    lap_distance = (
        distance_source
        - distance_origin
    )

    speed_mps = numeric_series(
        native,
        audit.speed_column,
    ).iloc[
        interval.start_position:
        interval.end_position + 1
    ].reset_index(drop=True)

    explicit_speed_kph = find_column(
        native,
        ("speed_kph", "vehicle_speed_kph"),
    )

    if explicit_speed_kph:
        speed_kph = numeric_column(
            native,
            explicit_speed_kph,
        ).iloc[
            interval.start_position:
            interval.end_position + 1
        ].reset_index(drop=True)
    else:
        speed_kph = speed_mps * 3.6

    explicit_throttle = find_column(
        native,
        EXPLICIT_THROTTLE_PCT_ALIASES,
    )

    throttle_source_column = (
        explicit_throttle
        or audit.throttle_column
    )

    throttle_pct = percentage_from_source(
        native[
            throttle_source_column
        ].iloc[
            interval.start_position:
            interval.end_position + 1
        ].reset_index(drop=True)
    )

    explicit_brake = find_column(
        native,
        EXPLICIT_BRAKE_PCT_ALIASES,
    )

    brake_source_column = (
        explicit_brake
        or audit.brake_column
    )

    brake_pct = percentage_from_source(
        native[
            brake_source_column
        ].iloc[
            interval.start_position:
            interval.end_position + 1
        ].reset_index(drop=True)
    )

    explicit_steer = find_column(
        native,
        EXPLICIT_STEER_NORM_ALIASES,
    )

    steer_source_column = (
        explicit_steer
        or audit.steer_column
    )

    steer_normalized = steering_from_source(
        native[
            steer_source_column
        ].iloc[
            interval.start_position:
            interval.end_position + 1
        ].reset_index(drop=True)
    )

    gear = numeric_series(
        native,
        audit.gear_column,
    ).iloc[
        interval.start_position:
        interval.end_position + 1
    ].reset_index(drop=True)

    lateral_g = resolve_g_channel(
        native,
        direct_aliases=LAT_G_ALIASES,
        acceleration_aliases=LAT_ACCEL_ALIASES,
    ).iloc[
        interval.start_position:
        interval.end_position + 1
    ].reset_index(drop=True)

    longitudinal_g = resolve_g_channel(
        native,
        direct_aliases=LON_G_ALIASES,
        acceleration_aliases=LON_ACCEL_ALIASES,
    ).iloc[
        interval.start_position:
        interval.end_position + 1
    ].reset_index(drop=True)

    lap_id = (
        f"{audit.session_id}_lap_{lap_index:02d}"
    )

    canonical = pd.DataFrame(
        {
            "session_id": audit.session_id,
            "capture_id": capture_id,
            "lap_id": lap_id,
            "extracted_lap_index": lap_index,
            "source_lap_number": interval.lap_number,
            "source_start_row": interval.start_row,
            "source_end_row": interval.end_row,
            "lap_elapsed_time_s": lap_elapsed_time,
            "lap_distance_m": lap_distance,
            "speed_mps": speed_mps,
            "speed_kph": speed_kph,
            "throttle_pct": throttle_pct,
            "brake_pct": brake_pct,
            "steer_normalized": steer_normalized,
            "gear": gear,
            "longitudinal_g": longitudinal_g,
            "lateral_g": lateral_g,
        }
    )

    raw = source.reset_index(
        drop=True
    ).copy()

    raw.columns = [
        f"raw__{column}"
        for column in raw.columns
    ]

    result = pd.concat(
        [
            canonical.reset_index(drop=True),
            raw,
        ],
        axis=1,
    )

    validate_monotonic(
        result["lap_elapsed_time_s"],
        "lap_elapsed_time_s",
    )

    validate_monotonic(
        result["lap_distance_m"],
        "lap_distance_m",
    )

    return result


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
    """Write text through a temporary file."""

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


def build_report(
    *,
    session_id: str,
    capture_id: str,
    artifacts: Sequence[ExtractionArtifact],
    source_audit: DetailedSessionAudit,
    summary_file: Path,
) -> str:
    """Build the human-readable extraction report."""

    audit = source_audit.summary

    lines = [
        "FH6 COMPLETE-LAP EXTRACTION REPORT",
        "=" * 72,
        "",
        "SOURCE",
        "-" * 72,
        f"Capture ID: {capture_id}",
        f"Native session: {session_id}",
        f"Boundary-aware audit status: {audit.readiness_status}",
        f"Detected lap boundaries: {audit.detected_lap_boundaries}",
        (
            "Clean qualified complete laps: "
            f"{audit.qualified_complete_lap_candidates}"
        ),
        "",
        "EXTRACTION POLICY",
        "-" * 72,
        "The validated Candidate Auditor V2 boundaries were reused directly.",
        "Only clean qualified complete intervals were exported.",
        "Native telemetry was not modified.",
        "Raw source fields are retained with the raw__ prefix.",
        "",
        "EXTRACTED LAPS",
        "-" * 72,
    ]

    for artifact in artifacts:
        lines.extend(
            [
                "",
                f"Lap artifact {artifact.lap_index:02d}",
                f"  Lap ID: {artifact.lap_id}",
                (
                    "  Source lap number: "
                    f"{artifact.source_lap_number}"
                ),
                (
                    "  Source rows: "
                    f"{artifact.start_row}-{artifact.end_row}"
                ),
                f"  Rows: {artifact.row_count}",
                (
                    "  Audit duration: "
                    f"{artifact.audit_duration_s:.3f} s"
                ),
                (
                    "  Extracted duration: "
                    f"{artifact.extracted_duration_s:.3f} s"
                ),
                (
                    "  Duration error: "
                    f"{artifact.duration_error_s:.6f} s"
                ),
                (
                    "  Audit distance: "
                    f"{artifact.audit_distance_m:.2f} m"
                ),
                (
                    "  Extracted distance: "
                    f"{artifact.extracted_distance_m:.2f} m"
                ),
                (
                    "  Distance error: "
                    f"{artifact.distance_error_m:.3f} m"
                ),
                (
                    "  Minimum required-channel coverage: "
                    f"{artifact.required_channel_coverage_pct:.3f}%"
                ),
                (
                    "  Validation: "
                    f"{artifact.validation_status}"
                ),
                (
                    "  Output: "
                    f"{artifact.output_file}"
                ),
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
            f"Extraction summary: {summary_file}",
            "",
            "FINAL STATUS",
            "-" * 72,
            f"Complete laps extracted: {len(artifacts)}",
            f"Extraction validation: {overall}",
            "",
            "No normalization or lap-comparison analysis was performed.",
        ]
    )

    return "\n".join(lines)


def run_extraction(
    args: argparse.Namespace,
) -> ExtractionResult:
    """Execute the complete extraction stage."""

    native_file = resolve_project_path(
        args.input
    )

    manifest_file = resolve_project_path(
        args.manifest
    )

    if not native_file.exists():
        raise FileNotFoundError(
            f"Native telemetry file does not exist: {native_file}"
        )

    if native_file.suffix.lower() != ".csv":
        raise ValueError(
            "Native telemetry input must be a CSV."
        )

    manifest = read_manifest(
        manifest_file
    )

    validate_manifest_capture(
        manifest=manifest,
        capture_id=args.capture_id,
        native_file=native_file,
    )

    detailed = rerun_boundary_audit(
        native_file=native_file,
        capture_id=args.capture_id,
        minimum_clean_laps=args.minimum_clean_laps,
    )

    clean_intervals = tuple(
        interval
        for interval in detailed.intervals
        if interval.qualified
    )

    if len(clean_intervals) < args.minimum_clean_laps:
        raise ValueError(
            "Not enough clean intervals remain after V2 audit."
        )

    native = pd.read_csv(
        native_file
    )

    output_directory = (
        resolve_project_path(args.output_dir)
        if args.output_dir is not None
        else (
            DEFAULT_LAP_ROOT
            / native_file.stem
        )
    )

    report_file = (
        resolve_project_path(args.report)
        if args.report is not None
        else (
            DEFAULT_REPORT_ROOT
            / (
                native_file.stem
                + "_lap_extraction_report.txt"
            )
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    built_laps: list[
        tuple[LapIntervalAudit, pd.DataFrame, Path]
    ] = []

    for lap_index, interval in enumerate(
        clean_intervals,
        start=1,
    ):
        dataframe = build_canonical_lap(
            native=native,
            detailed_audit=detailed,
            interval=interval,
            lap_index=lap_index,
            capture_id=args.capture_id,
        )

        output_file = (
            output_directory
            / f"lap_{lap_index:02d}.csv"
        )

        built_laps.append(
            (
                interval,
                dataframe,
                output_file,
            )
        )

    artifacts: list[ExtractionArtifact] = []

    for lap_index, (
        interval,
        dataframe,
        output_file,
    ) in enumerate(
        built_laps,
        start=1,
    ):
        extracted_duration = float(
            dataframe[
                "lap_elapsed_time_s"
            ].iloc[-1]
        )

        extracted_distance = float(
            dataframe[
                "lap_distance_m"
            ].iloc[-1]
        )

        duration_error = abs(
            extracted_duration
            - interval.duration_s
        )

        distance_error = abs(
            extracted_distance
            - interval.distance_m
        )

        coverage = required_coverage_pct(
            dataframe
        )

        validation_status = (
            "PASS"
            if (
                duration_error
                <= EXTRACTION_TIME_TOLERANCE_S
                and distance_error
                <= EXTRACTION_DISTANCE_TOLERANCE_M
                and coverage >= 99.0
                and interval.integrity_pass
                and interval.qualified
            )
            else "FAIL"
        )

        lap_id = str(
            dataframe["lap_id"].iloc[0]
        )

        artifacts.append(
            ExtractionArtifact(
                lap_index=lap_index,
                lap_id=lap_id,
                source_lap_number=(
                    interval.lap_number
                ),
                output_file=output_file,
                start_row=interval.start_row,
                end_row=interval.end_row,
                row_count=len(dataframe),
                audit_duration_s=(
                    interval.duration_s
                ),
                extracted_duration_s=(
                    extracted_duration
                ),
                duration_error_s=(
                    duration_error
                ),
                audit_distance_m=(
                    interval.distance_m
                ),
                extracted_distance_m=(
                    extracted_distance
                ),
                distance_error_m=(
                    distance_error
                ),
                required_channel_coverage_pct=(
                    coverage
                ),
                validation_status=(
                    validation_status
                ),
            )
        )

    if not all(
        artifact.validation_status == "PASS"
        for artifact in artifacts
    ):
        failed = [
            artifact.lap_index
            for artifact in artifacts
            if artifact.validation_status != "PASS"
        ]

        raise ValueError(
            "Extraction validation failed before files were committed. "
            f"Failed lap artifacts: {failed}"
        )

    for _, dataframe, output_file in built_laps:
        atomic_csv_write(
            dataframe,
            output_file,
        )

    summary_file = (
        output_directory
        / "lap_extraction_summary.csv"
    )

    summary_dataframe = pd.DataFrame(
        [
            {
                "session_id": native_file.stem,
                "capture_id": args.capture_id,
                "lap_id": artifact.lap_id,
                "extracted_lap_index": artifact.lap_index,
                "source_lap_number": artifact.source_lap_number,
                "source_start_row": artifact.start_row,
                "source_end_row": artifact.end_row,
                "row_count": artifact.row_count,
                "audit_duration_s": artifact.audit_duration_s,
                "extracted_duration_s": (
                    artifact.extracted_duration_s
                ),
                "duration_error_s": artifact.duration_error_s,
                "audit_distance_m": artifact.audit_distance_m,
                "extracted_distance_m": (
                    artifact.extracted_distance_m
                ),
                "distance_error_m": artifact.distance_error_m,
                "required_channel_coverage_pct": (
                    artifact.required_channel_coverage_pct
                ),
                "integrity_pass": True,
                "validation_status": artifact.validation_status,
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
        session_id=native_file.stem,
        capture_id=args.capture_id,
        artifacts=artifacts,
        source_audit=detailed,
        summary_file=summary_file,
    )

    atomic_text_write(
        report,
        report_file,
    )

    return ExtractionResult(
        session_id=native_file.stem,
        capture_id=args.capture_id,
        output_directory=output_directory,
        summary_file=summary_file,
        report_file=report_file,
        artifacts=tuple(artifacts),
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Extract boundary-validated complete laps from "
            "an FH6 formal-lap capture."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Validated native FH6 race telemetry CSV.",
    )

    parser.add_argument(
        "--capture-id",
        type=str,
        required=True,
        help="PASS capture ID from the formal-lap manifest.",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
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
        "--minimum-clean-laps",
        type=int,
        default=3,
    )

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""

    args = parse_arguments()

    if args.minimum_clean_laps < 1:
        raise ValueError(
            "--minimum-clean-laps must be positive."
        )

    print("=" * 72)
    print("FH6 Complete-Lap Extractor")
    print("=" * 72)

    result = run_extraction(
        args
    )

    print(
        f"\nSource session: {result.session_id}"
    )

    print(
        f"Capture ID: {result.capture_id}"
    )

    print(
        f"Complete laps extracted: {len(result.artifacts)}"
    )

    for artifact in result.artifacts:
        print(
            f"  Lap {artifact.lap_index:02d}: "
            f"{artifact.extracted_duration_s:.3f} s | "
            f"{artifact.extracted_distance_m:.2f} m | "
            f"{artifact.validation_status}"
        )

    print(
        "\nExtraction validation: PASS"
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
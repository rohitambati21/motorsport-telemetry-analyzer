"""
forza_session_logger.py

Native session capture and CSV logging for Forza Horizon 6 Data Out telemetry.

Responsibilities:
- Receive UDP datagrams.
- Decode valid FH6 packets.
- Show periodic live telemetry previews.
- Preserve decoded native telemetry in CSV form.
- Track basic session statistics.
- Report invalid packets and plausibility warnings.
- Shut down and flush data safely.

This module intentionally writes native telemetry rather than the normalized
schema used by the existing analyzer.
"""

from __future__ import annotations

import argparse
import csv
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, TextIO

from forza_packet_decoder import (
    EXPECTED_PACKET_SIZE,
    PACKET_FIELD_NAMES,
    ForzaPacketDecodeError,
    ForzaTelemetryPacket,
    decode_packet,
    format_live_preview,
    packet_definition_summary,
    validate_packet_plausibility,
)
from forza_udp_receiver import (
    DEFAULT_BIND_HOST,
    DEFAULT_PORT,
    ForzaUDPReceiver,
    ReceivedDatagram,
)


DEFAULT_CAPTURE_DURATION_S: Final[float] = 60.0
DEFAULT_PREVIEW_INTERVAL_S: Final[float] = 1.0
DEFAULT_STATUS_INTERVAL_S: Final[float] = 5.0
DEFAULT_FLUSH_EVERY_ROWS: Final[int] = 120


RECEIVER_METADATA_FIELDS: Final[
    tuple[str, ...]
] = (
    "sequence_number",
    "local_receive_time_ns",
    "sender_ip",
    "sender_port",
    "packet_length",
)


CSV_FIELD_NAMES: Final[
    tuple[str, ...]
] = (
    RECEIVER_METADATA_FIELDS
    + PACKET_FIELD_NAMES
)


@dataclass(slots=True)
class SessionStatistics:
    """
    Incremental session-quality and signal-range statistics.
    """

    valid_packets: int = 0
    invalid_packets: int = 0

    packets_with_plausibility_warnings: int = 0

    first_receive_time_ns: int | None = None
    last_receive_time_ns: int | None = None

    minimum_speed_mps: float | None = None
    maximum_speed_mps: float | None = None

    minimum_rpm: float | None = None
    maximum_rpm: float | None = None

    minimum_accel_raw: int | None = None
    maximum_accel_raw: int | None = None

    minimum_brake_raw: int | None = None
    maximum_brake_raw: int | None = None

    minimum_steer_raw: int | None = None
    maximum_steer_raw: int | None = None

    observed_gears: set[int] = field(
        default_factory=set
    )

    observed_laps: set[int] = field(
        default_factory=set
    )

    def update_valid_packet(
        self,
        packet: ForzaTelemetryPacket,
        datagram: ReceivedDatagram,
        warnings: tuple[str, ...],
    ) -> None:
        """
        Update statistics after one valid packet.
        """

        self.valid_packets += 1

        if warnings:
            self.packets_with_plausibility_warnings += 1

        if self.first_receive_time_ns is None:
            self.first_receive_time_ns = (
                datagram.received_time_ns
            )

        self.last_receive_time_ns = (
            datagram.received_time_ns
        )

        self.minimum_speed_mps = update_minimum(
            self.minimum_speed_mps,
            packet.speed_mps,
        )

        self.maximum_speed_mps = update_maximum(
            self.maximum_speed_mps,
            packet.speed_mps,
        )

        self.minimum_rpm = update_minimum(
            self.minimum_rpm,
            packet.current_engine_rpm,
        )

        self.maximum_rpm = update_maximum(
            self.maximum_rpm,
            packet.current_engine_rpm,
        )

        self.minimum_accel_raw = update_minimum(
            self.minimum_accel_raw,
            packet.accel_raw,
        )

        self.maximum_accel_raw = update_maximum(
            self.maximum_accel_raw,
            packet.accel_raw,
        )

        self.minimum_brake_raw = update_minimum(
            self.minimum_brake_raw,
            packet.brake_raw,
        )

        self.maximum_brake_raw = update_maximum(
            self.maximum_brake_raw,
            packet.brake_raw,
        )

        self.minimum_steer_raw = update_minimum(
            self.minimum_steer_raw,
            packet.steer_raw,
        )

        self.maximum_steer_raw = update_maximum(
            self.maximum_steer_raw,
            packet.steer_raw,
        )

        self.observed_gears.add(
            packet.gear
        )

        self.observed_laps.add(
            packet.lap_number
        )

    def mark_invalid_packet(
        self,
    ) -> None:
        """
        Count one packet that failed decoding.
        """

        self.invalid_packets += 1

    @property
    def local_capture_duration_s(
        self,
    ) -> float:
        """
        Duration between first and last valid locally received packets.
        """

        if (
            self.first_receive_time_ns is None
            or self.last_receive_time_ns is None
        ):
            return 0.0

        return (
            self.last_receive_time_ns
            - self.first_receive_time_ns
        ) / 1_000_000_000.0

    @property
    def average_valid_packet_rate_hz(
        self,
    ) -> float:
        """
        Approximate average valid-packet rate from local receipt timing.
        """

        duration_s = (
            self.local_capture_duration_s
        )

        if (
            duration_s <= 0.0
            or self.valid_packets < 2
        ):
            return 0.0

        return (
            self.valid_packets - 1
        ) / duration_s


class NativeSessionCSVLogger:
    """
    Buffered CSV writer for decoded native FH6 telemetry.
    """

    def __init__(
        self,
        output_file: Path,
        flush_every_rows: int =
            DEFAULT_FLUSH_EVERY_ROWS,
    ) -> None:
        if flush_every_rows <= 0:
            raise ValueError(
                "flush_every_rows must be positive."
            )

        self.output_file = output_file
        self.flush_every_rows = flush_every_rows

        self._file_handle: TextIO | None = None
        self._writer: csv.DictWriter | None = None

        self._rows_written = 0
        self._rows_since_flush = 0

    @property
    def rows_written(
        self,
    ) -> int:
        return self._rows_written

    def open(
        self,
    ) -> None:
        """
        Create output directory and initialize CSV writer.
        """

        if self._file_handle is not None:
            raise RuntimeError(
                "Session logger is already open."
            )

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_handle = self.output_file.open(
            mode="w",
            newline="",
            encoding="utf-8",
        )

        writer = csv.DictWriter(
            file_handle,
            fieldnames=CSV_FIELD_NAMES,
            extrasaction="raise",
        )

        writer.writeheader()
        file_handle.flush()

        self._file_handle = file_handle
        self._writer = writer

        self._rows_written = 0
        self._rows_since_flush = 0

    def write_packet(
        self,
        datagram: ReceivedDatagram,
        packet: ForzaTelemetryPacket,
    ) -> None:
        """
        Write one received and decoded packet to CSV.
        """

        if (
            self._file_handle is None
            or self._writer is None
        ):
            raise RuntimeError(
                "Session logger is not open."
            )

        row: dict[
            str,
            int | float | str,
        ] = {
            "sequence_number":
                datagram.sequence_number,
            "local_receive_time_ns":
                datagram.received_time_ns,
            "sender_ip":
                datagram.sender_ip,
            "sender_port":
                datagram.sender_port,
            "packet_length":
                datagram.packet_length,
        }

        row.update(
            packet.as_dict()
        )

        self._writer.writerow(
            row
        )

        self._rows_written += 1
        self._rows_since_flush += 1

        if (
            self._rows_since_flush
            >= self.flush_every_rows
        ):
            self.flush()

    def flush(
        self,
    ) -> None:
        """
        Flush buffered CSV content to disk.
        """

        if self._file_handle is not None:
            self._file_handle.flush()
            self._rows_since_flush = 0

    def close(
        self,
    ) -> None:
        """
        Flush and close the CSV file safely.
        """

        if self._file_handle is not None:
            self.flush()
            self._file_handle.close()

            self._file_handle = None
            self._writer = None

    def __enter__(
        self,
    ) -> NativeSessionCSVLogger:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()


def update_minimum(
    current_value: int | float | None,
    new_value: int | float,
) -> int | float:
    """
    Update a running minimum.
    """

    if current_value is None:
        return new_value

    return min(
        current_value,
        new_value,
    )


def update_maximum(
    current_value: int | float | None,
    new_value: int | float,
) -> int | float:
    """
    Update a running maximum.
    """

    if current_value is None:
        return new_value

    return max(
        current_value,
        new_value,
    )


def build_default_output_directory() -> Path:
    """
    Resolve the default native-session output folder.
    """

    project_root = Path(
        __file__
    ).resolve().parents[1]

    return (
        project_root
        / "data"
        / "forza"
        / "native"
    )


def build_session_output_path(
    output_directory: Path,
    file_prefix: str = "fh6_session",
) -> Path:
    """
    Build a unique timestamped native-session filename.
    """

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d_%H%M%S_UTC"
    )

    base_path = (
        output_directory
        / f"{file_prefix}_{timestamp}.csv"
    )

    if not base_path.exists():
        return base_path

    duplicate_index = 2

    while True:
        candidate_path = (
            output_directory
            / (
                f"{file_prefix}_{timestamp}_"
                f"{duplicate_index:02d}.csv"
            )
        )

        if not candidate_path.exists():
            return candidate_path

        duplicate_index += 1


def capture_native_session(
    host: str,
    port: int,
    duration_s: float | None,
    output_file: Path,
    preview_interval_s: float,
    status_interval_s: float,
    flush_every_rows: int,
) -> SessionStatistics:
    """
    Capture and save one native FH6 telemetry session.

    The duration timer begins after the first valid packet is received.
    A duration of None records until Ctrl+C.
    """

    if (
        duration_s is not None
        and duration_s <= 0.0
    ):
        raise ValueError(
            "duration_s must be positive or None."
        )

    if preview_interval_s <= 0.0:
        raise ValueError(
            "preview_interval_s must be positive."
        )

    if status_interval_s <= 0.0:
        raise ValueError(
            "status_interval_s must be positive."
        )

    receiver = ForzaUDPReceiver(
        host=host,
        port=port,
    )

    statistics = SessionStatistics()

    first_valid_monotonic_s: float | None = None

    last_preview_monotonic_s = 0.0
    last_status_monotonic_s = time.monotonic()

    invalid_error_print_count = 0
    warning_print_count = 0

    print("=" * 72)
    print(
        "Forza Horizon 6 Native Telemetry Session Logger"
    )
    print("=" * 72)

    print(
        f"\nListening on UDP {host}:{port}"
    )

    print(
        packet_definition_summary()
    )

    print(
        f"\nOutput file:\n{output_file}"
    )

    if duration_s is None:
        print(
            "\nCapture mode: run until Ctrl+C"
        )
    else:
        print(
            f"\nCapture duration: "
            f"{duration_s:.1f} s after first valid packet"
        )

    print(
        "\nWaiting for first valid FH6 telemetry packet..."
    )

    try:
        with receiver, NativeSessionCSVLogger(
            output_file=output_file,
            flush_every_rows=
                flush_every_rows,
        ) as session_logger:

            while True:
                try:
                    datagram = (
                        receiver.receive()
                    )
                except socket.timeout:
                    now = time.monotonic()

                    if (
                        now
                        - last_status_monotonic_s
                        >= status_interval_s
                    ):
                        print(
                            "\nStatus: waiting for telemetry..."
                        )

                        last_status_monotonic_s = now

                    continue

                try:
                    packet = decode_packet(
                        datagram.payload
                    )
                except ForzaPacketDecodeError as exc:
                    statistics.mark_invalid_packet()

                    if invalid_error_print_count < 5:
                        print(
                            "\nPacket skipped:"
                        )
                        print(
                            f"  {exc}"
                        )

                        invalid_error_print_count += 1

                    continue

                now = time.monotonic()

                if first_valid_monotonic_s is None:
                    first_valid_monotonic_s = now
                    last_preview_monotonic_s = now
                    last_status_monotonic_s = now

                    print(
                        "\nFirst valid FH6 packet received."
                    )

                    print(
                        "Capture timer started."
                    )

                warnings = (
                    validate_packet_plausibility(
                        packet
                    )
                )

                if (
                    warnings
                    and warning_print_count < 5
                ):
                    print(
                        "\nPlausibility warning:"
                    )

                    for warning in warnings:
                        print(
                            f"  - {warning}"
                        )

                    warning_print_count += 1

                session_logger.write_packet(
                    datagram,
                    packet,
                )

                statistics.update_valid_packet(
                    packet,
                    datagram,
                    warnings,
                )

                if (
                    now
                    - last_preview_monotonic_s
                    >= preview_interval_s
                ):
                    print(
                        format_live_preview(
                            packet
                        )
                    )

                    last_preview_monotonic_s = now

                if (
                    now
                    - last_status_monotonic_s
                    >= status_interval_s
                ):
                    print(
                        "\nCapture Status:"
                    )

                    print(
                        f"  Valid rows written: "
                        f"{session_logger.rows_written}"
                    )

                    print(
                        f"  Invalid packets: "
                        f"{statistics.invalid_packets}"
                    )

                    print(
                        f"  Average valid packet rate: "
                        f"{statistics.average_valid_packet_rate_hz:.1f} Hz"
                    )

                    last_status_monotonic_s = now

                if (
                    duration_s is not None
                    and first_valid_monotonic_s
                    is not None
                    and (
                        now
                        - first_valid_monotonic_s
                        >= duration_s
                    )
                ):
                    print(
                        "\nRequested capture duration reached."
                    )
                    break

    except KeyboardInterrupt:
        print(
            "\n\nCapture stopped by user."
        )

    return statistics


def print_session_summary(
    statistics: SessionStatistics,
    output_file: Path,
) -> None:
    """
    Print final capture-quality and signal-range summary.
    """

    print("\nNative Session Summary:")
    print(
        f"Valid packets saved: "
        f"{statistics.valid_packets}"
    )

    print(
        f"Invalid packets skipped: "
        f"{statistics.invalid_packets}"
    )

    print(
        "Packets with plausibility warnings: "
        f"{statistics.packets_with_plausibility_warnings}"
    )

    print(
        f"Local capture duration: "
        f"{statistics.local_capture_duration_s:.2f} s"
    )

    print(
        f"Average valid packet rate: "
        f"{statistics.average_valid_packet_rate_hz:.1f} Hz"
    )

    if statistics.valid_packets > 0:
        print(
            "\nObserved Signal Ranges:"
        )

        print(
            "Speed: "
            f"{statistics.minimum_speed_mps:.3f} to "
            f"{statistics.maximum_speed_mps:.3f} m/s"
        )

        print(
            "Speed: "
            f"{statistics.minimum_speed_mps * 3.6:.2f} to "
            f"{statistics.maximum_speed_mps * 3.6:.2f} km/h"
        )

        print(
            "RPM: "
            f"{statistics.minimum_rpm:.0f} to "
            f"{statistics.maximum_rpm:.0f}"
        )

        print(
            "Throttle raw: "
            f"{statistics.minimum_accel_raw} to "
            f"{statistics.maximum_accel_raw}"
        )

        print(
            "Brake raw: "
            f"{statistics.minimum_brake_raw} to "
            f"{statistics.maximum_brake_raw}"
        )

        print(
            "Steering raw: "
            f"{statistics.minimum_steer_raw} to "
            f"{statistics.maximum_steer_raw}"
        )

        print(
            "Gears observed: "
            f"{sorted(statistics.observed_gears)}"
        )

        print(
            "Lap numbers observed: "
            f"{sorted(statistics.observed_laps)}"
        )

    print(
        f"\nNative telemetry CSV:\n{output_file}"
    )

    print("=" * 72)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Capture decoded native Forza Horizon 6 "
            "Data Out telemetry to CSV."
        )
    )

    parser.add_argument(
        "--host",
        default=DEFAULT_BIND_HOST,
        help=(
            "Local bind address. "
            "Default: 0.0.0.0"
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=(
            "UDP port to listen on. "
            "Default: 53000"
        ),
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_CAPTURE_DURATION_S,
        help=(
            "Capture duration in seconds after the first valid packet. "
            "Use 0 to record until Ctrl+C. "
            "Default: 60"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=build_default_output_directory(),
        help=(
            "Directory for native session CSV files."
        ),
    )

    parser.add_argument(
        "--preview-interval",
        type=float,
        default=DEFAULT_PREVIEW_INTERVAL_S,
        help=(
            "Seconds between live telemetry previews. "
            "Default: 1.0"
        ),
    )

    parser.add_argument(
        "--status-interval",
        type=float,
        default=DEFAULT_STATUS_INTERVAL_S,
        help=(
            "Seconds between capture status summaries. "
            "Default: 5.0"
        ),
    )

    parser.add_argument(
        "--flush-every",
        type=int,
        default=DEFAULT_FLUSH_EVERY_ROWS,
        help=(
            "Rows written between CSV flushes. "
            "Default: 120"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """

    args = parse_arguments()

    duration_s: float | None = (
        None
        if args.duration == 0
        else args.duration
    )

    output_file = build_session_output_path(
        args.output_dir
    )

    statistics = capture_native_session(
        host=args.host,
        port=args.port,
        duration_s=duration_s,
        output_file=output_file,
        preview_interval_s=
            args.preview_interval,
        status_interval_s=
            args.status_interval,
        flush_every_rows=
            args.flush_every,
    )

    print_session_summary(
        statistics,
        output_file,
    )


if __name__ == "__main__":
    main()
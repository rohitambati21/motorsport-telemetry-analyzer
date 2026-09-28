"""
forza_udp_receiver.py

Network-only UDP receiver for Forza Horizon 6 Data Out telemetry.

Responsibilities:
- Bind a UDP socket.
- Receive datagrams.
- Timestamp local receipt.
- Record sender information.
- Count packets.
- Report packet lengths and approximate receive rate.
- Shut down cleanly.

This module intentionally does not decode telemetry fields.
That separation makes network problems distinguishable from decoder problems.
"""

from __future__ import annotations

import argparse
import socket
import time
from collections import Counter
from dataclasses import dataclass
from typing import Final


DEFAULT_BIND_HOST: Final[str] = "0.0.0.0"
DEFAULT_PORT: Final[int] = 53000
DEFAULT_TIMEOUT_S: Final[float] = 1.0
DEFAULT_STATUS_INTERVAL_S: Final[float] = 2.0
DEFAULT_MAX_DATAGRAM_SIZE: Final[int] = 4096
DEFAULT_RECEIVE_BUFFER_BYTES: Final[int] = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ReceivedDatagram:
    """
    One received UDP datagram plus local network metadata.
    """

    sequence_number: int
    payload: bytes

    sender_ip: str
    sender_port: int

    received_time_ns: int
    received_monotonic_s: float

    @property
    def packet_length(
        self,
    ) -> int:
        return len(
            self.payload
        )


class ForzaUDPReceiver:
    """
    Reusable FH6 UDP socket receiver.

    The receiver listens on all interfaces by default and returns raw
    datagrams without interpreting their binary contents.
    """

    def __init__(
        self,
        host: str = DEFAULT_BIND_HOST,
        port: int = DEFAULT_PORT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_datagram_size: int = DEFAULT_MAX_DATAGRAM_SIZE,
        receive_buffer_bytes: int = DEFAULT_RECEIVE_BUFFER_BYTES,
    ) -> None:
        if not host:
            raise ValueError(
                "host cannot be empty."
            )

        if not 1 <= port <= 65535:
            raise ValueError(
                "port must be between 1 and 65535."
            )

        if timeout_s <= 0.0:
            raise ValueError(
                "timeout_s must be positive."
            )

        if max_datagram_size <= 0:
            raise ValueError(
                "max_datagram_size must be positive."
            )

        if receive_buffer_bytes <= 0:
            raise ValueError(
                "receive_buffer_bytes must be positive."
            )

        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.max_datagram_size = max_datagram_size
        self.receive_buffer_bytes = receive_buffer_bytes

        self._socket: socket.socket | None = None
        self._sequence_number = 0

    @property
    def is_open(
        self,
    ) -> bool:
        return self._socket is not None

    def open(
        self,
    ) -> None:
        """
        Create, configure, and bind the UDP socket.
        """

        if self.is_open:
            raise RuntimeError(
                "UDP receiver is already open."
            )

        udp_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        try:
            udp_socket.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_RCVBUF,
                self.receive_buffer_bytes,
            )

            udp_socket.settimeout(
                self.timeout_s
            )

            udp_socket.bind(
                (
                    self.host,
                    self.port,
                )
            )
        except Exception:
            udp_socket.close()
            raise

        self._socket = udp_socket
        self._sequence_number = 0

    def receive(
        self,
    ) -> ReceivedDatagram:
        """
        Wait for and return one UDP datagram.

        Raises
        ------
        socket.timeout
            If no packet arrives during the configured timeout.
        """

        if self._socket is None:
            raise RuntimeError(
                "UDP receiver is not open."
            )

        payload, sender_address = (
            self._socket.recvfrom(
                self.max_datagram_size
            )
        )

        received_time_ns = time.time_ns()
        received_monotonic_s = (
            time.monotonic()
        )

        self._sequence_number += 1

        sender_ip = str(
            sender_address[0]
        )

        sender_port = int(
            sender_address[1]
        )

        return ReceivedDatagram(
            sequence_number=
                self._sequence_number,
            payload=payload,
            sender_ip=sender_ip,
            sender_port=sender_port,
            received_time_ns=
                received_time_ns,
            received_monotonic_s=
                received_monotonic_s,
        )

    def close(
        self,
    ) -> None:
        """
        Close the socket safely.
        """

        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def __enter__(
        self,
    ) -> ForzaUDPReceiver:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()


def run_network_monitor(
    host: str,
    port: int,
    expected_packet_size: int,
    status_interval_s: float,
    first_packet_details: int,
) -> None:
    """
    Run a network-only packet monitor.

    This is the first diagnostic tool to use before decoder or logger testing.
    """

    receiver = ForzaUDPReceiver(
        host=host,
        port=port,
    )

    total_packets = 0

    size_counts: Counter[int] = (
        Counter()
    )

    interval_packets = 0

    monitor_start = time.monotonic()
    interval_start = monitor_start

    last_sender: tuple[
        str,
        int,
    ] | None = None

    print("=" * 72)
    print(
        "Forza Horizon 6 UDP Network Monitor"
    )
    print("=" * 72)

    print(
        f"\nListening on UDP {host}:{port}"
    )

    print(
        f"Expected FH6 packet size: "
        f"{expected_packet_size} bytes"
    )

    print(
        "\nWaiting for telemetry..."
    )

    print(
        "Drive actively in FH6 after Data Out is configured."
    )

    print(
        "Press Ctrl+C to stop."
    )

    try:
        with receiver:
            while True:
                try:
                    datagram = (
                        receiver.receive()
                    )
                except socket.timeout:
                    now = time.monotonic()

                    if (
                        now - interval_start
                        >= status_interval_s
                    ):
                        print(
                            "\nStatus: still listening; "
                            "no packets received during the latest interval."
                        )

                        interval_start = now
                        interval_packets = 0

                    continue

                total_packets += 1
                interval_packets += 1

                size_counts[
                    datagram.packet_length
                ] += 1

                last_sender = (
                    datagram.sender_ip,
                    datagram.sender_port,
                )

                if (
                    total_packets
                    <= first_packet_details
                ):
                    size_status = (
                        "EXPECTED"
                        if datagram.packet_length
                        == expected_packet_size
                        else "UNEXPECTED"
                    )

                    print(
                        "\nPacket received:"
                    )

                    print(
                        f"  Sequence: "
                        f"{datagram.sequence_number}"
                    )

                    print(
                        f"  Sender: "
                        f"{datagram.sender_ip}:"
                        f"{datagram.sender_port}"
                    )

                    print(
                        f"  Length: "
                        f"{datagram.packet_length} bytes "
                        f"[{size_status}]"
                    )

                now = time.monotonic()

                interval_elapsed_s = (
                    now
                    - interval_start
                )

                if (
                    interval_elapsed_s
                    >= status_interval_s
                ):
                    interval_rate_hz = (
                        interval_packets
                        / interval_elapsed_s
                    )

                    total_elapsed_s = (
                        now
                        - monitor_start
                    )

                    total_average_rate_hz = (
                        total_packets
                        / total_elapsed_s
                        if total_elapsed_s > 0.0
                        else 0.0
                    )

                    print(
                        "\nNetwork Status:"
                    )

                    print(
                        f"  Total packets: "
                        f"{total_packets}"
                    )

                    print(
                        f"  Latest interval rate: "
                        f"{interval_rate_hz:.1f} packets/s"
                    )

                    print(
                        f"  Overall average rate: "
                        f"{total_average_rate_hz:.1f} packets/s"
                    )

                    print(
                        f"  Packet sizes observed: "
                        f"{dict(size_counts)}"
                    )

                    if last_sender is not None:
                        print(
                            f"  Last sender: "
                            f"{last_sender[0]}:"
                            f"{last_sender[1]}"
                        )

                    interval_start = now
                    interval_packets = 0

    except KeyboardInterrupt:
        print(
            "\n\nNetwork monitor stopped by user."
        )

    total_elapsed_s = (
        time.monotonic()
        - monitor_start
    )

    average_rate_hz = (
        total_packets
        / total_elapsed_s
        if total_elapsed_s > 0.0
        else 0.0
    )

    print("\nFinal Network Summary:")
    print(
        f"Total packets received: "
        f"{total_packets}"
    )

    print(
        f"Monitoring duration: "
        f"{total_elapsed_s:.2f} s"
    )

    print(
        f"Average receive rate: "
        f"{average_rate_hz:.1f} packets/s"
    )

    print(
        f"Packet sizes observed: "
        f"{dict(size_counts)}"
    )

    if (
        size_counts
        and set(size_counts)
        == {expected_packet_size}
    ):
        print(
            "Packet-size result: "
            "all received packets matched the expected size."
        )
    elif size_counts:
        print(
            "Packet-size result: "
            "one or more unexpected packet sizes were observed."
        )
    else:
        print(
            "Packet-size result: "
            "no packets were received."
        )

    print("=" * 72)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Network-only UDP monitor for "
            "Forza Horizon 6 Data Out telemetry."
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
        "--expected-size",
        type=int,
        default=324,
        help=(
            "Expected packet size in bytes. "
            "Default: 324"
        ),
    )

    parser.add_argument(
        "--status-interval",
        type=float,
        default=DEFAULT_STATUS_INTERVAL_S,
        help=(
            "Seconds between network status updates. "
            "Default: 2.0"
        ),
    )

    parser.add_argument(
        "--first-packet-details",
        type=int,
        default=5,
        help=(
            "Number of initial packets to print individually. "
            "Default: 5"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """

    args = parse_arguments()

    run_network_monitor(
        host=args.host,
        port=args.port,
        expected_packet_size=
            args.expected_size,
        status_interval_s=
            args.status_interval,
        first_packet_details=
            args.first_packet_details,
    )


if __name__ == "__main__":
    main()
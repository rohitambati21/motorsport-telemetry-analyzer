"""
forza_packet_decoder.py

Binary packet decoder for Forza Horizon 6 Data Out telemetry.

Responsibilities:
- Validate the fixed FH6 packet size.
- Decode the packet using the documented binary field order.
- Preserve native field meanings and units.
- Return an immutable structured telemetry record.
- Provide lightweight plausibility checks for live validation.

This module intentionally does not:
- normalize telemetry into the analyzer schema
- reconstruct laps
- calculate lap-relative distance
- generate figures
- write CSV files

Those responsibilities belong to later layers of the project.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping, TypeAlias


NativeScalar: TypeAlias = int | float

EXPECTED_PACKET_SIZE: Final[int] = 324


# Each tuple stores:
#   (project-friendly field name, Python struct format code)
#
# Format code meanings:
#   i = signed 32-bit integer
#   I = unsigned 32-bit integer
#   f = 32-bit floating point
#   H = unsigned 16-bit integer
#   B = unsigned 8-bit integer
#   b = signed 8-bit integer
FIELD_SPECS: Final[tuple[tuple[str, str], ...]] = (
    ("is_race_on", "i"),
    ("timestamp_ms", "I"),

    ("engine_max_rpm", "f"),
    ("engine_idle_rpm", "f"),
    ("current_engine_rpm", "f"),

    ("acceleration_x_mps2", "f"),
    ("acceleration_y_mps2", "f"),
    ("acceleration_z_mps2", "f"),

    ("velocity_x_mps", "f"),
    ("velocity_y_mps", "f"),
    ("velocity_z_mps", "f"),

    ("angular_velocity_x_rps", "f"),
    ("angular_velocity_y_rps", "f"),
    ("angular_velocity_z_rps", "f"),

    ("yaw_rad", "f"),
    ("pitch_rad", "f"),
    ("roll_rad", "f"),

    (
        "normalized_suspension_travel_front_left",
        "f",
    ),
    (
        "normalized_suspension_travel_front_right",
        "f",
    ),
    (
        "normalized_suspension_travel_rear_left",
        "f",
    ),
    (
        "normalized_suspension_travel_rear_right",
        "f",
    ),

    ("tire_slip_ratio_front_left", "f"),
    ("tire_slip_ratio_front_right", "f"),
    ("tire_slip_ratio_rear_left", "f"),
    ("tire_slip_ratio_rear_right", "f"),

    (
        "wheel_rotation_speed_front_left_rps",
        "f",
    ),
    (
        "wheel_rotation_speed_front_right_rps",
        "f",
    ),
    (
        "wheel_rotation_speed_rear_left_rps",
        "f",
    ),
    (
        "wheel_rotation_speed_rear_right_rps",
        "f",
    ),

    (
        "wheel_on_rumble_strip_front_left",
        "i",
    ),
    (
        "wheel_on_rumble_strip_front_right",
        "i",
    ),
    (
        "wheel_on_rumble_strip_rear_left",
        "i",
    ),
    (
        "wheel_on_rumble_strip_rear_right",
        "i",
    ),

    (
        "wheel_in_puddle_front_left",
        "i",
    ),
    (
        "wheel_in_puddle_front_right",
        "i",
    ),
    (
        "wheel_in_puddle_rear_left",
        "i",
    ),
    (
        "wheel_in_puddle_rear_right",
        "i",
    ),

    ("surface_rumble_front_left", "f"),
    ("surface_rumble_front_right", "f"),
    ("surface_rumble_rear_left", "f"),
    ("surface_rumble_rear_right", "f"),

    ("tire_slip_angle_front_left", "f"),
    ("tire_slip_angle_front_right", "f"),
    ("tire_slip_angle_rear_left", "f"),
    ("tire_slip_angle_rear_right", "f"),

    ("tire_combined_slip_front_left", "f"),
    ("tire_combined_slip_front_right", "f"),
    ("tire_combined_slip_rear_left", "f"),
    ("tire_combined_slip_rear_right", "f"),

    (
        "suspension_travel_front_left_m",
        "f",
    ),
    (
        "suspension_travel_front_right_m",
        "f",
    ),
    (
        "suspension_travel_rear_left_m",
        "f",
    ),
    (
        "suspension_travel_rear_right_m",
        "f",
    ),

    ("car_ordinal", "i"),
    ("car_class", "i"),
    ("car_performance_index", "i"),
    ("drivetrain_type", "i"),
    ("num_cylinders", "i"),
    ("car_group", "I"),

    (
        "smashable_velocity_difference_mps",
        "f",
    ),
    ("smashable_mass_kg", "f"),

    ("position_x_m", "f"),
    ("position_y_m", "f"),
    ("position_z_m", "f"),

    ("speed_mps", "f"),
    ("power_w", "f"),
    ("torque_nm", "f"),

    ("tire_temp_front_left", "f"),
    ("tire_temp_front_right", "f"),
    ("tire_temp_rear_left", "f"),
    ("tire_temp_rear_right", "f"),

    ("boost_psi", "f"),
    ("fuel_fraction", "f"),
    ("distance_traveled_m", "f"),

    ("best_lap_s", "f"),
    ("last_lap_s", "f"),
    ("current_lap_s", "f"),
    ("current_race_time_s", "f"),

    ("lap_number", "H"),
    ("race_position", "B"),

    ("accel_raw", "B"),
    ("brake_raw", "B"),
    ("clutch_raw", "B"),
    ("handbrake_raw", "B"),

    ("gear", "B"),

    ("steer_raw", "b"),
    ("normalized_driving_line_raw", "b"),
    ("normalized_ai_brake_difference_raw", "b"),
)


PACKET_FIELD_NAMES: Final[tuple[str, ...]] = tuple(
    field_name
    for field_name, _ in FIELD_SPECS
)


# FH6 telemetry is decoded little-endian.
#
# The explicit final padding byte reconciles the documented field list
# with the documented fixed 324-byte packet size. The padding byte is
# ignored and does not appear in PACKET_FIELD_NAMES.
_PACKET_STRUCT: Final[struct.Struct] = struct.Struct(
    "<"
    + "".join(
        format_code
        for _, format_code in FIELD_SPECS
    )
    + "x"
)


if _PACKET_STRUCT.size != EXPECTED_PACKET_SIZE:
    raise RuntimeError(
        "Internal FH6 packet definition is invalid. "
        f"Expected {EXPECTED_PACKET_SIZE} bytes, "
        f"but struct size is {_PACKET_STRUCT.size} bytes."
    )


class ForzaPacketDecodeError(ValueError):
    """
    Raised when a UDP payload cannot be decoded as an FH6 packet.
    """


@dataclass(frozen=True, slots=True)
class ForzaTelemetryPacket:
    """
    Immutable decoded FH6 telemetry packet.

    Native values are stored using source-accurate project field names
    while preserving the original FH6 units and scales.
    """

    values: Mapping[str, NativeScalar]

    def __post_init__(self) -> None:
        """
        Freeze the underlying dictionary so decoded records cannot be
        accidentally modified after creation.
        """

        object.__setattr__(
            self,
            "values",
            MappingProxyType(
                dict(self.values)
            ),
        )

    def as_dict(
        self,
    ) -> dict[str, NativeScalar]:
        """
        Return a normal dictionary for CSV writing or inspection.
        """

        return dict(
            self.values
        )

    def value(
        self,
        field_name: str,
    ) -> NativeScalar:
        """
        Return one field by name.
        """

        try:
            return self.values[
                field_name
            ]
        except KeyError as exc:
            raise KeyError(
                f"Unknown FH6 telemetry field: {field_name}"
            ) from exc

    @property
    def is_race_on(
        self,
    ) -> int:
        return int(
            self.values["is_race_on"]
        )

    @property
    def timestamp_ms(
        self,
    ) -> int:
        return int(
            self.values["timestamp_ms"]
        )

    @property
    def speed_mps(
        self,
    ) -> float:
        return float(
            self.values["speed_mps"]
        )

    @property
    def current_engine_rpm(
        self,
    ) -> float:
        return float(
            self.values[
                "current_engine_rpm"
            ]
        )

    @property
    def distance_traveled_m(
        self,
    ) -> float:
        return float(
            self.values[
                "distance_traveled_m"
            ]
        )

    @property
    def current_race_time_s(
        self,
    ) -> float:
        return float(
            self.values[
                "current_race_time_s"
            ]
        )

    @property
    def current_lap_s(
        self,
    ) -> float:
        return float(
            self.values["current_lap_s"]
        )

    @property
    def lap_number(
        self,
    ) -> int:
        return int(
            self.values["lap_number"]
        )

    @property
    def accel_raw(
        self,
    ) -> int:
        return int(
            self.values["accel_raw"]
        )

    @property
    def brake_raw(
        self,
    ) -> int:
        return int(
            self.values["brake_raw"]
        )

    @property
    def gear(
        self,
    ) -> int:
        return int(
            self.values["gear"]
        )

    @property
    def steer_raw(
        self,
    ) -> int:
        return int(
            self.values["steer_raw"]
        )


def decode_packet(
    payload: bytes | bytearray | memoryview,
) -> ForzaTelemetryPacket:
    """
    Decode one FH6 Data Out packet.

    Raises
    ------
    ForzaPacketDecodeError
        If payload length is not exactly 324 bytes.
    """

    payload_view = memoryview(
        payload
    )

    packet_size = payload_view.nbytes

    if packet_size != EXPECTED_PACKET_SIZE:
        raise ForzaPacketDecodeError(
            "Unsupported FH6 telemetry packet length. "
            f"Received {packet_size} bytes; "
            f"expected {EXPECTED_PACKET_SIZE} bytes."
        )

    try:
        unpacked_values = _PACKET_STRUCT.unpack(
            payload_view
        )
    except struct.error as exc:
        raise ForzaPacketDecodeError(
            "FH6 packet unpacking failed."
        ) from exc

    if len(unpacked_values) != len(
        PACKET_FIELD_NAMES
    ):
        raise ForzaPacketDecodeError(
            "Decoded value count does not match "
            "the FH6 field-name definition."
        )

    packet_values = dict(
        zip(
            PACKET_FIELD_NAMES,
            unpacked_values,
            strict=True,
        )
    )

    return ForzaTelemetryPacket(
        packet_values
    )


def validate_packet_plausibility(
    packet: ForzaTelemetryPacket,
) -> tuple[str, ...]:
    """
    Perform lightweight physical and structural plausibility checks.

    These checks return warnings rather than rejecting the packet.
    The decoder's job is source preservation, not aggressive filtering.
    """

    warnings: list[str] = []

    if packet.is_race_on not in {
        0,
        1,
    }:
        warnings.append(
            "is_race_on is outside the expected 0/1 range."
        )

    for field_name, value in packet.values.items():
        if (
            isinstance(value, float)
            and not math.isfinite(value)
        ):
            warnings.append(
                f"{field_name} is not finite."
            )

    if packet.speed_mps < 0.0:
        warnings.append(
            "speed_mps is negative."
        )

    if packet.current_engine_rpm < 0.0:
        warnings.append(
            "current_engine_rpm is negative."
        )

    fuel_fraction = float(
        packet.values["fuel_fraction"]
    )

    if (
        math.isfinite(fuel_fraction)
        and not 0.0
        <= fuel_fraction
        <= 1.05
    ):
        warnings.append(
            "fuel_fraction is outside the expected approximate 0–1 range."
        )

    for timing_field in (
        "best_lap_s",
        "last_lap_s",
        "current_lap_s",
        "current_race_time_s",
    ):
        timing_value = float(
            packet.values[
                timing_field
            ]
        )

        if (
            math.isfinite(timing_value)
            and timing_value < 0.0
        ):
            warnings.append(
                f"{timing_field} is negative."
            )

    return tuple(
        warnings
    )


def format_live_preview(
    packet: ForzaTelemetryPacket,
) -> str:
    """
    Format the most useful first-day channels for terminal validation.
    """

    speed_kph = (
        packet.speed_mps
        * 3.6
    )

    return (
        f"timestamp={packet.timestamp_ms:>10d} ms | "
        f"speed={speed_kph:>7.2f} km/h | "
        f"rpm={packet.current_engine_rpm:>8.0f} | "
        f"throttle={packet.accel_raw:>3d}/255 | "
        f"brake={packet.brake_raw:>3d}/255 | "
        f"gear={packet.gear:>2d} | "
        f"steer={packet.steer_raw:>4d} | "
        f"lap={packet.lap_number:>3d}"
    )


def packet_definition_summary() -> str:
    """
    Return a compact structural description for diagnostics.
    """

    return (
        f"FH6 packet size: {_PACKET_STRUCT.size} bytes | "
        f"decoded fields: {len(PACKET_FIELD_NAMES)}"
    )
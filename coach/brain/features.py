"""Fast, local geometric features derived from :class:`ClimberState`."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from math import atan2, degrees, hypot
from statistics import median

from coach.contracts import ClimberState, Limb


def height_above(lower_y: float, upper_y: float) -> float:
    """Return vertical height from ``lower_y`` to ``upper_y`` (y increases down)."""
    return lower_y - upper_y


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _keypoint(state: ClimberState, name: str) -> tuple[float, float] | None:
    value = state.keypoints.get(name)
    if value is None or len(value) < 3 or value[2] < 0.5:
        return None
    return value[0], value[1]


def _chain_length(state: ClimberState, names: tuple[str, ...]) -> float | None:
    points = [_keypoint(state, name) for name in names]
    if any(point is None for point in points):
        return None
    return sum(_distance(a, b) for a, b in zip(points, points[1:], strict=False))  # type: ignore[arg-type]


def _elbow_angle(state: ClimberState, side: str) -> float | None:
    points = [_keypoint(state, f"{side}_{part}") for part in ("shoulder", "elbow", "wrist")]
    if any(point is None for point in points):
        return None
    shoulder, elbow, wrist = points  # type: ignore[misc]
    first = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
    second = (wrist[0] - elbow[0], wrist[1] - elbow[1])
    return abs(
        degrees(
            atan2(
                first[0] * second[1] - first[1] * second[0],
                first[0] * second[0] + first[1] * second[1],
            )
        )
    )


@dataclass
class Features:
    limb_speed: dict[Limb, float] = field(default_factory=dict)
    elbow_angle: dict[str, float | None] = field(default_factory=dict)
    hip_velocity_y: float = 0.0
    hip_speed: float = 0.0
    hip_static_s: float = 0.0
    arm_length: float = 0.0
    leg_length: float = 0.0
    highest_hold_id: int | None = None
    hand_moving_no_contact_s: dict[Limb, float] = field(default_factory=dict)
    # Net movement over the last static_window_s, per second. Frame-to-frame speed is mostly
    # pose jitter when someone hangs still; over a window the jitter cancels and real movement
    # does not. None when the tracker did not fill it in (features built by hand in tests).
    hip_drift: float | None = None
    limb_drift: dict[Limb, float] | None = None
    _calibrate: Callable[[ClimberState], None] | None = field(default=None, repr=False)

    @property
    def hip_still_speed(self) -> float:
        """The hip speed to judge stillness by: windowed when known, else frame-to-frame."""
        return self.hip_speed if self.hip_drift is None else self.hip_drift

    @property
    def limb_still_speed(self) -> dict[Limb, float]:
        return self.limb_speed if self.limb_drift is None else self.limb_drift

    def add_calibration_sample(self, state: ClimberState) -> None:
        """Forward a standing sample to the tracker that produced this snapshot."""
        if self._calibrate is not None:
            self._calibrate(state)


class FeatureTracker:
    """Keep the small amount of history needed by the attempt and cue logic."""

    def __init__(self, move_speed_eps: float = 0.02, static_window_s: float = 0.5) -> None:
        self.move_speed_eps = float(move_speed_eps)
        self.static_window_s = float(static_window_s)
        self._hip_trail: deque[tuple[float, tuple[float, float]]] = deque()
        self._limb_trail: dict[Limb, deque[tuple[float, tuple[float, float]]]] = {
            limb: deque() for limb in Limb
        }
        self._previous: ClimberState | None = None
        self._hip_static_s = 0.0
        self._hand_search_s = {Limb.LH: 0.0, Limb.RH: 0.0}
        self._highest_hold_id: int | None = None
        self._arm_samples: list[float] = []
        self._leg_samples: list[float] = []

    def add_calibration_sample(self, state: ClimberState) -> None:
        arm = _chain_length(state, ("left_shoulder", "left_elbow", "left_wrist"))
        if arm is None:
            arm = _chain_length(state, ("right_shoulder", "right_elbow", "right_wrist"))
        leg = _chain_length(state, ("left_hip", "left_knee", "left_ankle"))
        if leg is None:
            leg = _chain_length(state, ("right_hip", "right_knee", "right_ankle"))
        if arm is not None:
            self._arm_samples.append(arm)
        if leg is not None:
            self._leg_samples.append(leg)

    def _drift(
        self,
        trail: deque[tuple[float, tuple[float, float]]],
        t: float,
        point: tuple[float, float],
    ) -> float:
        """Net speed of a point since the oldest sample inside the window (always at least
        the previous sample, so widely spaced frames give plain frame-to-frame speed)."""
        while len(trail) > 1 and t - trail[1][0] >= self.static_window_s:
            trail.popleft()
        speed = 0.0
        if trail and t > trail[0][0]:
            speed = _distance(point, trail[0][1]) / (t - trail[0][0])
        trail.append((t, point))
        return speed

    def reset_highest_hold(self) -> None:
        """Reset the highest hold tracker at the start of a new attempt."""
        self._highest_hold_id = None

    def update(self, state: ClimberState) -> Features:
        previous = self._previous
        dt = max(0.0, state.t - previous.t) if previous is not None else 0.0
        speeds = {limb: 0.0 for limb in Limb}
        if previous is not None and dt > 0:
            for limb, point in state.limbs.items():
                old = previous.limbs.get(limb)
                if old is not None:
                    speeds[limb] = _distance(point, old) / dt

        # A person reappearing after a dropout (person_found False -> True) is not
        # continuous motion: pose.py freezes the hip at its last known point while
        # nobody is found, so the next real detection (possibly a different person
        # entirely) can jump far away. Treating that jump as hip velocity/speed has
        # mistaken a person just entering the frame for a fall.
        continuous = previous is not None and previous.person_found and state.person_found
        hip_velocity = 0.0
        if continuous and dt > 0:
            hip_velocity = (state.hip[1] - previous.hip[1]) / dt
        hip_speed = 0.0
        if continuous and dt > 0:
            hip_speed = _distance(state.hip, previous.hip) / dt
        if not continuous:  # a dropout: the trail before it says nothing about movement now
            self._hip_trail.clear()
            for trail in self._limb_trail.values():
                trail.clear()
        hip_drift = self._drift(self._hip_trail, state.t, state.hip)
        limb_drift = {limb: 0.0 for limb in Limb}
        for limb, trail in self._limb_trail.items():
            point = state.limbs.get(limb)
            if point is None:
                trail.clear()
            else:
                limb_drift[limb] = self._drift(trail, state.t, point)
        if previous is not None and dt > 0 and hip_drift < self.move_speed_eps:
            self._hip_static_s += dt
        elif hip_drift >= self.move_speed_eps:
            self._hip_static_s = 0.0

        for hand in (Limb.LH, Limb.RH):
            if previous is not None and dt > 0 and speeds[hand] > self.move_speed_eps:
                if state.contacts.get(hand) is None:
                    self._hand_search_s[hand] += dt
                else:
                    self._hand_search_s[hand] = 0.0
            else:
                self._hand_search_s[hand] = 0.0

        for hold_id in state.contacts.values():
            if hold_id is not None and (
                self._highest_hold_id is None or hold_id > self._highest_hold_id
            ):
                self._highest_hold_id = hold_id

        current_arm = _chain_length(state, ("left_shoulder", "left_elbow", "left_wrist"))
        if current_arm is None:
            current_arm = _chain_length(state, ("right_shoulder", "right_elbow", "right_wrist"))
        current_leg = _chain_length(state, ("left_hip", "left_knee", "left_ankle"))
        if current_leg is None:
            current_leg = _chain_length(state, ("right_hip", "right_knee", "right_ankle"))
        self._previous = state
        return Features(
            limb_speed=speeds,
            elbow_angle={
                "left": _elbow_angle(state, "left"),
                "right": _elbow_angle(state, "right"),
            },
            hip_velocity_y=hip_velocity,
            hip_speed=hip_speed,
            hip_static_s=self._hip_static_s,
            arm_length=median(self._arm_samples) if self._arm_samples else (current_arm or 0.0),
            leg_length=median(self._leg_samples) if self._leg_samples else (current_leg or 0.0),
            highest_hold_id=self._highest_hold_id,
            hand_moving_no_contact_s=dict(self._hand_search_s),
            hip_drift=hip_drift,
            limb_drift=limb_drift,
            _calibrate=self.add_calibration_sample,
        )


if __name__ == "__main__":
    print("FeatureTracker ready; provide ClimberState frames to update().")

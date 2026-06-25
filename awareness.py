"""Situational awareness helpers for Second Sight."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


PRIORITY_RANKS = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Critical": 3,
}

DISTANCE_WEIGHTS = {
    "Far": 0,
    "Medium": 1,
    "Near": 2,
    "Very Close": 3,
}

POSITION_WEIGHTS = {
    "Left": 1,
    "Center": 2,
    "Right": 1,
}

# Future versions can add class-specific weights here. For example, stairs or
# vehicles could be treated as more important than small household objects.
OBJECT_CLASS_WEIGHTS: dict[str, int] = {}


@dataclass(frozen=True)
class Alert:
    """A warning that can later be sent to speech or haptic feedback."""

    message: str
    priority: str
    detection: dict[str, Any]
    escalated: bool


@dataclass
class _AlertState:
    """Remember the last alert sent for one tracked or approximated object."""

    last_alert_time: float
    priority_rank: int


def calculate_priority_score(class_name: str, position: str, distance: str) -> int:
    """Calculate a simple obstacle priority score from known detection details."""
    distance_score = DISTANCE_WEIGHTS.get(distance, 0)
    position_score = POSITION_WEIGHTS.get(position, 0)
    class_score = OBJECT_CLASS_WEIGHTS.get(class_name.lower(), 0)
    return distance_score + position_score + class_score


def priority_from_score(score: int, position: str, distance: str) -> str:
    """Convert a numeric priority score into a user-friendly category."""
    if position == "Center" and distance == "Very Close":
        return "Critical"
    if score >= 5:
        return "Critical"
    if score >= 4:
        return "High"
    if score >= 2:
        return "Medium"
    return "Low"


def calculate_priority(class_name: str, position: str, distance: str) -> str:
    """Return Low, Medium, High, or Critical for a detected object."""
    score = calculate_priority_score(class_name, position, distance)
    return priority_from_score(score, position, distance)


def get_detection_alert_key(detection: dict[str, Any]) -> str:
    """Return a stable-enough key for cooldown tracking.

    If a tracker ID exists in future phases, it is preferred. Otherwise, class
    and position provide a lightweight fallback without requiring tracking.
    """
    track_id = detection.get("track_id")
    if track_id is not None:
        return f"track:{track_id}"

    class_name = detection.get("class_name", "object")
    position = detection.get("position", "unknown")
    return f"{class_name}:{position}".lower()


class AlertManager:
    """Decide when important detections should generate alerts."""

    def __init__(self, cooldowns: dict[str, float] | None = None) -> None:
        self.cooldowns = cooldowns or {
            "High": 5.0,
            "Critical": 2.0,
        }
        self._last_alerts: dict[str, _AlertState] = {}

    def process_detections(self, detections: list[dict[str, Any]]) -> list[Alert]:
        """Return alerts for the most meaningful detections in this frame."""
        alerts: list[Alert] = []
        current_time = time.monotonic()

        # Higher-priority objects are evaluated first so future consumers can
        # choose to announce only the first alert without extra sorting.
        sorted_detections = sorted(
            detections,
            key=lambda detection: PRIORITY_RANKS.get(detection.get("priority", "Low"), 0),
            reverse=True,
        )

        for detection in sorted_detections:
            if not self.should_alert(detection, current_time):
                continue

            alert = self._build_alert(detection, current_time)
            self._remember_alert(detection, current_time)
            alerts.append(alert)

        return alerts

    def should_alert(
        self,
        detection: dict[str, Any],
        current_time: float | None = None,
    ) -> bool:
        """Return True when a detection is important and not in cooldown."""
        priority = detection.get("priority", "Low")
        priority_rank = PRIORITY_RANKS.get(priority, 0)

        if priority_rank < PRIORITY_RANKS["High"]:
            return False

        now = current_time if current_time is not None else time.monotonic()
        alert_key = get_detection_alert_key(detection)
        previous_alert = self._last_alerts.get(alert_key)

        if previous_alert is None:
            return True

        # Escalation is immediate: a High object becoming Critical should not
        # wait for the normal cooldown.
        if priority_rank > previous_alert.priority_rank:
            return True

        cooldown = self.cooldowns.get(priority, 5.0)
        return now - previous_alert.last_alert_time >= cooldown

    def _build_alert(self, detection: dict[str, Any], current_time: float) -> Alert:
        priority = detection.get("priority", "Low")
        alert_key = get_detection_alert_key(detection)
        previous_alert = self._last_alerts.get(alert_key)
        priority_rank = PRIORITY_RANKS.get(priority, 0)
        escalated = previous_alert is not None and priority_rank > previous_alert.priority_rank

        class_name = detection.get("class_name", "Object")
        position = detection.get("position", "Unknown")
        distance = detection.get("distance", "Unknown")
        message = f"{class_name} - {position} - {distance} - {priority}"

        return Alert(
            message=message,
            priority=priority,
            detection=detection,
            escalated=escalated,
        )

    def _remember_alert(self, detection: dict[str, Any], current_time: float) -> None:
        priority = detection.get("priority", "Low")
        alert_key = get_detection_alert_key(detection)
        self._last_alerts[alert_key] = _AlertState(
            last_alert_time=current_time,
            priority_rank=PRIORITY_RANKS.get(priority, 0),
        )

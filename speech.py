"""Offline speech guidance for Second Sight alerts."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from awareness import Alert, PRIORITY_RANKS, get_detection_alert_key


@dataclass(frozen=True)
class SpeechConfig:
    """Configuration values for speech queueing and duplicate suppression."""

    enabled: bool = True
    speech_cooldown_seconds: float = 8.0
    duplicate_suppression_seconds: float = 3.0
    max_queue_size: int = 5
    max_history_entries: int = 100
    history_ttl_seconds: float = 120.0
    minimum_priority: str = "High"
    interrupt_priority: str = "Critical"
    voice_rate: int = 175
    voice_volume: float = 1.0


@dataclass(frozen=True)
class SpeechEvent:
    """A single message approved for spoken guidance."""

    message: str
    priority: str
    alert_key: str
    priority_rank: int
    created_at: float = field(default_factory=time.monotonic)
    escalated: bool = False


@dataclass
class _SpeechHistory:
    """Remember recently accepted speech so duplicates can be suppressed."""

    accepted_at: float
    priority_rank: int


class SpeechManager:
    """Queue and speak smart-alert messages without blocking the camera loop."""

    def __init__(self, config: SpeechConfig | None = None) -> None:
        self.config = config or SpeechConfig()
        self._queue: deque[SpeechEvent] = deque()
        self._condition = threading.Condition()
        self._last_by_key: dict[str, _SpeechHistory] = {}
        self._last_by_message: dict[str, float] = {}
        self._current_message: str | None = None
        self._current_priority_rank: int | None = None
        self._last_speech_duration_seconds: float | None = None
        self._last_error: str | None = None
        self._engine_ready = False
        self._stop_requested = False
        self._engine: Any | None = None
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="SecondSightSpeech",
            daemon=True,
        )

        if self.config.enabled:
            self._worker.start()

    def submit_alerts(self, alerts: list[Alert]) -> None:
        """Queue alerts that have already been approved by AlertManager."""
        for alert in alerts:
            self.submit_alert(alert)

    def submit_message(
        self,
        message: str,
        priority: str = "High",
        alert_key: str = "system",
    ) -> bool:
        """Queue a non-object message through the same speech infrastructure."""
        if not self.config.enabled:
            return False

        event = SpeechEvent(
            message=message,
            priority=priority,
            alert_key=alert_key,
            priority_rank=PRIORITY_RANKS.get(priority, 0),
        )
        return self._submit_event(event)

    def submit_alert(self, alert: Alert) -> bool:
        """Queue one alert for speech when cooldown rules allow it."""
        if not self.config.enabled:
            return False

        event = self._create_event(alert)
        return self._submit_event(event)

    def _submit_event(self, event: SpeechEvent) -> bool:
        """Queue one speech event when cooldown rules allow it."""
        if event.priority_rank < PRIORITY_RANKS[self.config.minimum_priority]:
            return False

        now = time.monotonic()

        with self._condition:
            if self._last_error:
                return False

            self._prune_history(now)

            if self._should_suppress(event, now):
                return False

            self._remember_event(event, now)
            self._enqueue_event(event)
            self._condition.notify()
            return True

    def get_status_text(self) -> str:
        """Return a short speech status string for optional debug overlay."""
        with self._condition:
            if not self.config.enabled:
                return "Speech disabled"
            if self._last_error:
                return "Speech unavailable"
            if self._current_message:
                return f"Speaking: {self._current_message}"
            if self._queue:
                return f"Speech queue: {len(self._queue)}"
            if not self._engine_ready:
                return "Speech starting"
            return "Speech ready"

    def shutdown(self) -> None:
        """Stop the speech worker cleanly."""
        if not self.config.enabled:
            return

        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()

        self._worker.join(timeout=2.0)
        if self._worker.is_alive():
            logging.warning("Speech worker did not stop within timeout")

    def _create_event(self, alert: Alert) -> SpeechEvent:
        message = format_voice_message(alert)
        priority_rank = PRIORITY_RANKS.get(alert.priority, 0)
        return SpeechEvent(
            message=message,
            priority=alert.priority,
            alert_key=get_detection_alert_key(alert.detection),
            priority_rank=priority_rank,
            escalated=alert.escalated,
        )

    def _should_suppress(self, event: SpeechEvent, now: float) -> bool:
        previous_key = self._last_by_key.get(event.alert_key)
        previous_message_time = self._last_by_message.get(event.message)

        if event.priority != self.config.interrupt_priority and previous_message_time is not None:
            if now - previous_message_time < self.config.duplicate_suppression_seconds:
                return True

        if previous_key is None:
            return False

        # Escalation must be allowed through even inside the normal cooldown.
        if event.priority_rank > previous_key.priority_rank:
            return False

        return now - previous_key.accepted_at < self.config.speech_cooldown_seconds

    def _remember_event(self, event: SpeechEvent, now: float) -> None:
        self._last_by_key[event.alert_key] = _SpeechHistory(
            accepted_at=now,
            priority_rank=event.priority_rank,
        )
        self._last_by_message[event.message] = now

    def _prune_history(self, now: float) -> None:
        """Keep cooldown history bounded during long sessions."""
        ttl = self.config.history_ttl_seconds
        self._last_by_key = {
            key: history
            for key, history in self._last_by_key.items()
            if now - history.accepted_at <= ttl
        }
        self._last_by_message = {
            message: accepted_at
            for message, accepted_at in self._last_by_message.items()
            if now - accepted_at <= ttl
        }

        while len(self._last_by_key) > self.config.max_history_entries:
            oldest_key = min(
                self._last_by_key,
                key=lambda key: self._last_by_key[key].accepted_at,
            )
            del self._last_by_key[oldest_key]

        while len(self._last_by_message) > self.config.max_history_entries:
            oldest_message = min(self._last_by_message, key=self._last_by_message.get)
            del self._last_by_message[oldest_message]

    def _enqueue_event(self, event: SpeechEvent) -> None:
        if event.priority == self.config.interrupt_priority:
            # Critical messages jump ahead of lower-priority queued speech. This
            # keeps urgent hazards responsive without changing alert decisions.
            self._queue = deque(
                queued_event
                for queued_event in self._queue
                if queued_event.priority_rank >= event.priority_rank
            )
            self._queue.appendleft(event)
            while len(self._queue) > self.config.max_queue_size:
                self._queue.pop()
            return

        if len(self._queue) >= self.config.max_queue_size:
            self._drop_lowest_priority_event()

        self._queue.append(event)

    def _drop_lowest_priority_event(self) -> None:
        if not self._queue:
            return

        lowest_index = min(
            range(len(self._queue)),
            key=lambda index: self._queue[index].priority_rank,
        )
        del self._queue[lowest_index]

    def _worker_loop(self) -> None:
        if not self._initialize_engine():
            return

        while True:
            with self._condition:
                while not self._queue and not self._stop_requested:
                    self._condition.wait()

                if self._stop_requested:
                    return

                event = self._queue.popleft()
                self._current_message = event.message
                self._current_priority_rank = event.priority_rank

            self._speak(event.message)

            with self._condition:
                self._current_message = None
                self._current_priority_rank = None

    def _initialize_engine(self) -> bool:
        try:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.config.voice_rate)
            self._engine.setProperty("volume", self.config.voice_volume)
            with self._condition:
                self._engine_ready = True
            logging.info("Offline speech engine initialized")
            return True
        except Exception as exc:
            with self._condition:
                self._last_error = str(exc)
            logging.warning("Speech guidance disabled: %s", exc)
            return False

    def _speak(self, message: str) -> None:
        if self._engine is None:
            return

        try:
            logging.info("Speaking: %s", message)
            start_time = time.monotonic()
            self._engine.say(message)
            self._engine.runAndWait()
            duration = time.monotonic() - start_time
            with self._condition:
                self._last_speech_duration_seconds = duration
            logging.info("Speech completed in %.2fs", duration)
        except Exception as exc:
            with self._condition:
                self._last_error = str(exc)
            logging.warning("Speech failed: %s", exc)


def format_voice_message(alert: Alert) -> str:
    """Create a concise spoken phrase from an approved smart alert."""
    detection = alert.detection
    class_name = str(detection.get("class_name", "Object")).capitalize()
    position = detection.get("position", "Center")
    distance = detection.get("distance", "")

    if position == "Center":
        location_phrase = "ahead"
    elif position == "Left":
        location_phrase = "on your left"
    elif position == "Right":
        location_phrase = "on your right"
    else:
        location_phrase = ""

    message_parts = [f"{class_name} {location_phrase}".strip()]

    if distance == "Very Close":
        message_parts.append("Very close")
    elif distance in {"Near", "Medium"}:
        message_parts.append(str(distance))

    return ". ".join(message_parts) + "."

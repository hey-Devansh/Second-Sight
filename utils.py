"""Shared helpers for Second Sight."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

import cv2
from cv2.typing import MatLike


def configure_logging() -> None:
    """Configure readable console logging for the prototype."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


class FPSCounter:
    """Measure frames per second using elapsed wall-clock time."""

    def __init__(self) -> None:
        self._previous_time = time.perf_counter()
        self.fps = 0.0

    def update(self) -> float:
        """Update and return the latest FPS estimate."""
        current_time = time.perf_counter()
        elapsed = current_time - self._previous_time
        self._previous_time = current_time

        if elapsed > 0:
            self.fps = 1.0 / elapsed

        return self.fps


class PipelineProfiler:
    """Collect lightweight timing data for each stage of the video pipeline."""

    def __init__(self, report_interval: int = 60) -> None:
        self.report_interval = report_interval
        self.frame_count = 0
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Time a named section and include it in periodic reports."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self._totals[name] += elapsed_ms
            self._counts[name] += 1

    def finish_frame(self) -> None:
        """Record a completed frame and log averages at the report interval."""
        self.frame_count += 1
        if self.frame_count % self.report_interval == 0:
            self.log_report()

    def average_ms(self, name: str) -> float:
        """Return the average milliseconds spent in a named section."""
        count = self._counts[name]
        if count == 0:
            return 0.0
        return self._totals[name] / count

    def log_report(self) -> None:
        """Log a compact performance report for active and future stages."""
        active_stages = (
            "webcam_capture",
            "yolo_inference",
            "tracking",
            "drawing",
            "ocr",
            "audio",
            "display_wait",
            "background_threads",
        )
        stage_times = {stage: self.average_ms(stage) for stage in active_stages}
        total_ms = sum(stage_times.values())
        estimated_fps = 1000.0 / total_ms if total_ms > 0 else 0.0

        logging.info(
            "Performance avg over %s frames | FPS %.1f | capture %.1fms | "
            "YOLO %.1fms | tracking %.1fms | drawing %.1fms | OCR %.1fms | "
            "audio %.1fms | display/wait %.1fms | background %.1fms",
            self.report_interval,
            estimated_fps,
            stage_times["webcam_capture"],
            stage_times["yolo_inference"],
            stage_times["tracking"],
            stage_times["drawing"],
            stage_times["ocr"],
            stage_times["audio"],
            stage_times["display_wait"],
            stage_times["background_threads"],
        )


def draw_fps(frame: MatLike, fps: float) -> None:
    """Draw the current FPS value in the top-left corner of the frame."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def get_object_position(center_x: float, frame_width: int) -> str:
    """Classify an object's horizontal center as Left, Center, or Right."""
    if frame_width <= 0:
        raise ValueError("frame_width must be greater than zero.")

    left_boundary = frame_width / 3
    right_boundary = (frame_width * 2) / 3

    if center_x < left_boundary:
        return "Left"
    if center_x < right_boundary:
        return "Center"
    return "Right"


def estimate_distance(
    box_width: float,
    box_height: float,
    frame_width: int,
    frame_height: int,
) -> str:
    """Estimate object proximity from bounding-box size.

    This is a lightweight heuristic, not true depth. The assumption is that an
    object taking up more of the camera frame is generally closer to the user.
    A future depth model can replace this function without changing callers.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be greater than zero.")

    safe_box_width = max(0.0, box_width)
    safe_box_height = max(0.0, box_height)
    box_area = safe_box_width * safe_box_height
    frame_area = frame_width * frame_height
    area_ratio = box_area / frame_area

    if area_ratio >= 0.45:
        return "Very Close"
    if area_ratio >= 0.18:
        return "Near"
    if area_ratio >= 0.05:
        return "Medium"
    return "Far"


def should_quit(key_code: int) -> bool:
    """Return True when the pressed key should close the application."""
    return key_code & 0xFF == ord("q")

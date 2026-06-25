"""User-triggered OCR support for Second Sight."""

from __future__ import annotations

import logging
import re
import threading
from collections import deque
from dataclasses import dataclass

import cv2
from cv2.typing import MatLike


@dataclass(frozen=True)
class OCRConfig:
    """Configuration for one-shot OCR requests."""

    min_line_length: int = 2
    max_spoken_chars: int = 220
    tesseract_config: str = "--psm 6"


@dataclass(frozen=True)
class OCRResult:
    """Clean OCR text plus the message that should be spoken."""

    text: str
    speech_message: str
    success: bool


class OCRManager:
    """Run OCR only when requested, without blocking the video loop."""

    def __init__(self, config: OCRConfig | None = None) -> None:
        self.config = config or OCRConfig()
        self._condition = threading.Condition()
        self._pending_frame: MatLike | None = None
        self._completed_results: deque[OCRResult] = deque()
        self._latest_text = ""
        self._status_text = "OCR ready"
        self._busy = False
        self._stop_requested = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="SecondSightOCR",
            daemon=True,
        )
        self._worker.start()

    def request_read(self, frame: MatLike) -> bool:
        """Queue the current frame for OCR if no OCR request is active."""
        with self._condition:
            if self._busy or self._pending_frame is not None:
                self._status_text = "OCR busy"
                return False

            self._pending_frame = frame.copy()
            self._status_text = "OCR queued"
            self._condition.notify()
            return True

    def get_completed_results(self) -> list[OCRResult]:
        """Return completed OCR results and clear the delivery queue."""
        with self._condition:
            results = list(self._completed_results)
            self._completed_results.clear()
            return results

    def get_status_text(self) -> str:
        """Return the current OCR status for lightweight overlay display."""
        with self._condition:
            return self._status_text

    def get_latest_text(self) -> str:
        """Return the most recent cleaned OCR text."""
        with self._condition:
            return self._latest_text

    def shutdown(self) -> None:
        """Stop the OCR worker cleanly."""
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()

        self._worker.join(timeout=2.0)
        if self._worker.is_alive():
            logging.warning("OCR worker did not stop within timeout")

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while self._pending_frame is None and not self._stop_requested:
                    self._condition.wait()

                if self._stop_requested:
                    return

                frame = self._pending_frame
                self._pending_frame = None
                self._busy = True
                self._status_text = "OCR reading"

            result = self._process_frame(frame)

            with self._condition:
                self._busy = False
                self._latest_text = result.text
                self._status_text = "OCR complete" if result.success else "OCR no text"
                self._completed_results.append(result)

    def _process_frame(self, frame: MatLike | None) -> OCRResult:
        if frame is None:
            return OCRResult(
                text="",
                speech_message="No readable text detected.",
                success=False,
            )

        try:
            import pytesseract

            prepared_frame = self._prepare_frame(frame)
            raw_text = pytesseract.image_to_string(
                prepared_frame,
                config=self.config.tesseract_config,
            )
            cleaned_text = clean_ocr_text(raw_text, self.config.min_line_length)

            if not cleaned_text:
                return OCRResult(
                    text="",
                    speech_message="No readable text detected.",
                    success=False,
                )

            spoken_text = cleaned_text[: self.config.max_spoken_chars].strip()
            return OCRResult(
                text=cleaned_text,
                speech_message=f"Text detected. {spoken_text}.",
                success=True,
            )
        except Exception as exc:
            logging.warning("OCR failed: %s", exc)
            return OCRResult(
                text="",
                speech_message="OCR is unavailable.",
                success=False,
            )

    @staticmethod
    def _prepare_frame(frame: MatLike) -> MatLike:
        """Prepare a camera frame for Tesseract without changing live video."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.threshold(
            gray_frame,
            0,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )[1]


def clean_ocr_text(raw_text: str, min_line_length: int = 2) -> str:
    """Remove duplicate, empty, and obviously noisy OCR lines."""
    cleaned_lines: list[str] = []
    seen_lines: set[str] = set()

    for raw_line in raw_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if len(line) < min_line_length:
            continue
        if _is_noise_line(line):
            continue

        normalized_line = line.casefold()
        if normalized_line in seen_lines:
            continue

        seen_lines.add(normalized_line)
        cleaned_lines.append(line)

    return " ".join(cleaned_lines)


def _is_noise_line(line: str) -> bool:
    """Return True for lines that look like OCR artifacts, not readable text."""
    alpha_numeric_count = sum(character.isalnum() for character in line)
    if alpha_numeric_count == 0:
        return True

    readable_ratio = alpha_numeric_count / len(line)
    return readable_ratio < 0.45

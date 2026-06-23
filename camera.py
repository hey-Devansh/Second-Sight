"""Camera handling for Second Sight."""

from __future__ import annotations

import logging

import cv2
from cv2.typing import MatLike


class Camera:
    """Small wrapper around OpenCV webcam capture."""

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open the webcam and raise a clear error if it is unavailable."""
        logging.info("Opening webcam at index %s", self.camera_index)
        self._capture = cv2.VideoCapture(self.camera_index)

        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise RuntimeError(
                "Could not open webcam. Check that it is connected and not used by another app."
            )

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Keep only the newest camera frame when the backend supports it.
        # This reduces perceived lag if detection takes longer than capture.
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logging.info("Webcam opened successfully")

    def read(self) -> MatLike:
        """Read one frame from the webcam."""
        if self._capture is None:
            raise RuntimeError("Camera has not been opened yet.")

        success, frame = self._capture.read()
        if not success or frame is None:
            raise RuntimeError("Could not read a frame from the webcam.")

        return frame

    def release(self) -> None:
        """Release the webcam if it is open."""
        if self._capture is not None:
            logging.info("Releasing webcam")
            self._capture.release()
            self._capture = None

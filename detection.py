"""YOLOv8 object detection for Second Sight."""

from __future__ import annotations

import logging

import cv2
from cv2.typing import MatLike
from ultralytics import YOLO


class ObjectDetector:
    """Load a YOLOv8 model and run object detection on webcam frames."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.35,
        image_size: int = 416,
        iou: float = 0.45,
        max_detections: int = 30,
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.image_size = image_size
        self.iou = iou
        self.max_detections = max_detections
        logging.info("Loading YOLOv8 model: %s", self.model_path)
        self.model = YOLO(self.model_path)

        # Fuse convolution and batch-normalization layers when available.
        # This is a one-time CPU optimization and does not change detections.
        try:
            self.model.fuse()
        except Exception as exc:  # pragma: no cover - depends on model/backend
            logging.debug("Model fusion skipped: %s", exc)

        logging.info(
            "YOLOv8 model loaded with confidence=%.2f, image_size=%s, max_detections=%s",
            self.confidence,
            self.image_size,
            self.max_detections,
        )

    def detect(self, frame: MatLike):
        """Run object detection and return Ultralytics results."""
        return self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.image_size,
            iou=self.iou,
            max_det=self.max_detections,
            verbose=False,
        )

    @staticmethod
    def draw_detections(results) -> MatLike:
        """Draw bounding boxes, class names, and confidence scores on a frame.

        Ultralytics' built-in plot method is convenient but does extra work for
        many result types. A small OpenCV drawer is faster for Phase 1 boxes.
        """
        result = results[0]
        annotated_frame = result.orig_img.copy()
        boxes = result.boxes

        if boxes is None:
            return annotated_frame

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = f"{result.names[class_id]} {confidence:.2f}"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text_size, baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2,
            )
            label_y = max(y1, text_size[1] + baseline + 4)
            cv2.rectangle(
                annotated_frame,
                (x1, label_y - text_size[1] - baseline - 4),
                (x1 + text_size[0] + 6, label_y + baseline - 2),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 3, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        return annotated_frame

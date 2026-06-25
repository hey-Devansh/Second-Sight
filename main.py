"""Second Sight Phase 1 prototype.

This script opens a webcam, runs YOLOv8 object detection, draws detections and
FPS, and exits cleanly when the user presses Q.
"""

from __future__ import annotations

import logging

import cv2

from awareness import AlertManager
from camera import Camera
from detection import ObjectDetector
from ocr import OCRManager
from speech import SpeechManager
from utils import (
    FPSCounter,
    PipelineProfiler,
    configure_logging,
    draw_alerts,
    draw_fps,
    draw_ocr_status,
    draw_speech_status,
    should_quit,
)


WINDOW_NAME = "Second Sight - Phase 1"
PROFILE_REPORT_INTERVAL = 60


def run() -> None:
    """Run the real-time object detection loop."""
    configure_logging()
    logging.info("Starting Second Sight Phase 1")
    logging.info("Press Q in the video window to quit")

    camera = Camera(camera_index=0)
    detector = ObjectDetector(model_path="yolov8n.pt")
    alert_manager = AlertManager()
    speech_manager = SpeechManager()
    ocr_manager = OCRManager()
    fps_counter = FPSCounter()
    profiler = PipelineProfiler(report_interval=PROFILE_REPORT_INTERVAL)

    try:
        camera.open()

        while True:
            try:
                with profiler.section("webcam_capture"):
                    frame = camera.read()
            except RuntimeError as exc:
                logging.warning("%s", exc)
                continue

            with profiler.section("yolo_inference"):
                results = detector.detect(frame)

            with profiler.section("alert_generation"):
                detections = getattr(results[0], "second_sight_detections", [])
                alerts = alert_manager.process_detections(detections)
                for alert in alerts:
                    log_method = logging.error if alert.priority == "Critical" else logging.warning
                    log_method("Alert: %s", alert.message)

            with profiler.section("speech_queue"):
                speech_manager.submit_alerts(alerts)
                for ocr_result in ocr_manager.get_completed_results():
                    speech_manager.submit_message(
                        ocr_result.speech_message,
                        priority="High",
                        alert_key=f"ocr:{ocr_result.speech_message}",
                    )

            with profiler.section("drawing"):
                annotated_frame = detector.draw_detections(results)
                fps = fps_counter.update()
                draw_fps(annotated_frame, fps)
                draw_alerts(annotated_frame, alerts)
                draw_ocr_status(
                    annotated_frame,
                    ocr_manager.get_status_text(),
                    ocr_manager.get_latest_text(),
                )
                draw_speech_status(annotated_frame, speech_manager.get_status_text())

            # Speech runs downstream from alerts on its own worker thread. The
            # frame profiler still keeps future OCR/background sections visible
            # so later phases can be compared against this baseline.

            with profiler.section("display_wait"):
                cv2.imshow(WINDOW_NAME, annotated_frame)
                key_code = cv2.waitKey(1)

            if key_code & 0xFF == ord("r"):
                if ocr_manager.request_read(frame):
                    logging.info("OCR requested")
                else:
                    logging.info("OCR request ignored because OCR is busy")

            profiler.finish_frame()

            if should_quit(key_code):
                logging.info("Quit requested")
                break

    except RuntimeError as exc:
        logging.error("%s", exc)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        ocr_manager.shutdown()
        speech_manager.shutdown()
        camera.release()
        cv2.destroyAllWindows()
        logging.info("Second Sight stopped")


if __name__ == "__main__":
    run()

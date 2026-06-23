# Second Sight

Second Sight is a Python accessibility assistant prototype for visually impaired users. Phase 1 opens a webcam, runs YOLOv8 object detection in real time, draws bounding boxes with labels and confidence scores, shows FPS, and exits cleanly when `Q` is pressed.

## Current Phase

Phase 1 is implemented.

- Opens the default webcam.
- Runs Ultralytics YOLOv8 object detection.
- Draws bounding boxes, object names, and confidence scores.
- Shows FPS on screen.
- Logs startup, camera, model, and shutdown events.
- Handles camera open/read failures gracefully.

## Project Structure

```text
camera.py       # Webcam setup, frame reading, and release
detection.py    # YOLOv8 model loading and detection
main.py         # Application loop
utils.py        # Logging, FPS, and keyboard helpers
requirements.txt
README.md
```

## Linux Setup

Use Python 3.10 or newer.

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On some Linux systems, OpenCV needs system webcam/display libraries:

```bash
sudo apt update
sudo apt install python3-opencv v4l-utils
```

The first run may download the YOLOv8 nano model file, `yolov8n.pt`, if it is not already present.

## Run

```bash
python main.py
```

Click the video window and press `Q` to quit.

## Notes

- The default camera index is `0`. If your webcam uses a different index, update `Camera(camera_index=0)` in `main.py`.
- `yolov8n.pt` is used because it is the smallest YOLOv8 model and is a good starting point for CPU performance.
- Future phases should add position awareness, offline speech, distance estimation, OCR mode, scene description, and voice commands without turning `main.py` into a large monolithic file.

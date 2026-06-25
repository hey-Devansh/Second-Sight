# Second Sight

Second Sight is a real-time computer vision accessibility assistant designed to help visually impaired users better understand their surroundings.

Using a webcam and on-device AI models, the system can detect objects, estimate their relative position and distance, prioritize potential obstacles, provide spoken alerts, and read visible text aloud through OCR.

The project focuses on low-cost, local-first accessibility using commonly available hardware.

---

## Features

### Real-Time Object Detection

* Detects objects using YOLOv8.
* Processes a live webcam feed.
* Draws bounding boxes and labels on detected objects.

### Position Awareness

Determines where an object is located relative to the user:

* Left
* Center
* Right

Example:

```text
Chair - Left
Person - Center
```

### Distance Estimation

Approximates object distance using bounding box size.

Distance categories:

* Far
* Medium
* Near
* Very Close

Example:

```text
Person - Center - Very Close
```

### Obstacle Prioritization

Objects are ranked according to their potential importance.

Factors include:

* Position
* Distance

Priority levels:

* Low
* Medium
* High
* Critical

### Smart Alert System

The application does not treat every object equally.

It prioritizes meaningful hazards and generates alerts when necessary.

Examples:

```text
Chair - Center - Very Close - Critical
Person - Left - Near - High
```

### Speech Guidance

Provides spoken feedback using offline text-to-speech.

Examples:

```text
"Person ahead. Very close."

"Chair on your left."
```

### Speech Queue Management

Prevents audio spam by:

* Suppressing duplicate messages
* Managing speech cooldowns
* Prioritizing critical alerts
* Preventing overlapping speech

### OCR Text Reading

Reads visible text from the camera feed on demand.

Press:

```text
R
```

to perform OCR on the current frame.

Examples:

```text
Emergency Exit

Welcome to Library

Second Sight
```

OCR results can also be spoken aloud.

---

## Tech Stack

### Computer Vision

* OpenCV
* Ultralytics YOLOv8

### OCR

* Tesseract OCR
* pytesseract

### Speech

* pyttsx3

### Language

* Python 3

---

## Installation

Clone the repository:

```bash
git clone https://github.com/hey-Devansh/Second-Sight.git
cd Second-Sight
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install Tesseract OCR:

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install tesseract-ocr
```

Verify installation:

```bash
tesseract --version
```

---

## Usage

Run:

```bash
python main.py
```

Controls:

| Key | Action                      |
| --- | --------------------------- |
| Q   | Quit application            |
| R   | Read visible text using OCR |

---

## Project Structure

```text
Second-Sight/
│
├── main.py
├── camera.py
├── detection.py
├── awareness.py
├── speech.py
├── ocr.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## Future Improvements

Planned ideas include:

* Scene understanding
* Navigation guidance
* Spatial audio feedback
* Improved distance estimation
* Mobile and Raspberry Pi support
* Multi-language OCR and speech

---

## Disclaimer

This project is currently a prototype and should not be relied upon as a primary safety device.

Distance estimation and object recognition are approximations and may occasionally produce incorrect results.

---

## Author

Developed by Devansh Sharma as a computer vision and accessibility project.


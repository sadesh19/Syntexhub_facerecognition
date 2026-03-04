# Face Recognition Python

This is a Python-based facial recognition system using OpenCV's Deep Neural Network (DNN) face detector and the `face_recognition` library.

## Features

- **Real-Time Detection and Recognition**: Processes live video from your webcam to detect and continuously recognize faces frame by frame.
- **OpenCV DNN Backend**: Uses a pre-trained Caffe SSD model (`res10_300x300_ssd_iter_140000.caffemodel` and `deploy.prototxt`) for fast and robust face detection.
- **On-the-Fly Registration**: Detects "Unknown" faces. While the program is running, simply press the `r` key on your keyboard to pause and register the new face into the database immediately. The image is saved to the `known_faces/` folder.
- **Offline Face Registration**: Use the `register_face.py` script to manually register new users offline.

## Requirements

Ensure you have a Python environment set up (e.g., using `venv`) and install the required dependencies:

```bash
pip install -r requirements.txt
```

You will need CMake installed to build `dlib` (a dependency of the `face_recognition` package).

## Usage

### 1. Main Recognition Module

Run the main script to start your webcam and begin face detection and recognition:

```bash
python main.py
```

- When an unknown face appears, an "Unknown" bounding box will be drawn in red.
- Press `r` on the keyboard to capture the face.
- Go to your terminal window where the script is running, type the name of the person, and hit Enter. The face will now be recognized and bounded in green.
- Press `q` on the webcam interface to quit the application.

### 2. Manual Face Registration

If you want to quickly add a bunch of photos manually, you can place images inside the `known_faces/` folder. The filename (without the extension) will be used as the person's name (e.g., `John_Doe.jpg`).

Alternatively, run the dedicated registration script:

```bash
python register_face.py
```

## Directory Structure

- `main.py`: The entry point for real-time webcam face recognition and dynamic registering.
- `register_face.py`: Auxiliary script for standalone face registration logic.
- `known_faces/`: Automatically created. Where cropped faces are persisted.
- `deploy.prototxt` & `res10_300x300...caffemodel`: Required models for OpenCV DNN based face detection.
- `requirements.txt`: Project dependencies.

# my_project
ML Project - Online Cheating Detection System

**Description:**

-Developed a real-time intelligent monitoring system to detect suspicious behaviors during online exams or meetings using a webcam feed.

-Implemented detection modules for:

-Head movement using Optical Flow (Lucas-Kanade).

-Eye gaze tracking and look-away detection with MediaPipe facial landmarks.

-Hand-over-eye gestures using hand-to-eye Euclidean distance.

-Mobile phone detection using a custom-trained YOLOv8 object detection model (8,400+ images).

-Designed a Tkinter-based GUI to allow user interaction and control.

-Alert system integrated using PyAutoGUI to notify when cheating/distraction events are triggered.

-Engineered for continuous video frame analysis with cooldown periods for realistic, low-false-positive detection.

**Achievements:**

-Successfully detects distractions like using phones, looking away, or covering eyes in real-time.

-Applicable to e-learning, remote work, and driver monitoring systems.

-Provides a base for future features like emotion detection, multi-user tracking, and edge device deployment.


Link for used image dataset in yolo model: https://www.kaggle.com/datasets/a165079/cellphoneobjectdetectionusingyolov7

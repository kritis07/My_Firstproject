

**Project Title:** Real-Time AI-Powered Cheating Detection System for Online Interviews & Examinations

**Team Members:** Kriti Singh Behl, Simranjit Kaur

**Description:**
Developed a comprehensive, real-time intelligent monitoring system to detect and prevent cheating behaviors during online interviews and examinations using computer vision and deep learning. The system analyzes webcam feeds to identify suspicious activities with high accuracy, ensuring the integrity of remote assessment processes.

**Key Features & Technical Implementation:**

- **Multi-Modal Cheating Detection:** Engineered a robust system capable of detecting four primary cheating scenarios:
  - **Mobile Phone Usage:** Implemented a custom-trained YOLOv8 object detection model, trained on a curated dataset of 8,400+ images for precise device identification.
  - **Head Movement Tracking:** Utilized Optical Flow algorithms (Lucas-Kanade) to monitor and classify unusual head rotations (up, down, left, right).
  - **Eye Gaze Tracking:** Integrated MediaPipe facial landmarks for real-time gaze estimation and look-away detection.
  - **Hand-over-Eye Gestures:** Detected eye-blocking attempts using Euclidean distance calculations between hand and eye landmarks.

- **End-to-End System Design:** Built a complete pipeline from data collection and preprocessing (including data augmentation with Albumentations to enhance dataset diversity) to real-time inference and alerting.

- **Intelligent Alert System:** Designed a Tkinter-based GUI for user-friendly interaction and integrated PyAutoGUI for immediate notifications when suspicious behaviors are detected, featuring cooldown periods to minimize false positives.

- **Advanced Model Training:** Employed time-series clustering algorithms (Time Series K-Means) for behavior classification without manual labeling, achieving an overall accuracy of 83.6% across all cheating scenarios.

**Achievements & Impact:**

- Successfully created a reliable system that addresses critical challenges in remote proctoring, enhancing fairness and credibility in online evaluations.
- Demonstrated strong performance in real-time environments, making it applicable for e-learning platforms, remote hiring processes, and secure testing environments.
- Provided a scalable foundation for future enhancements, including multi-user tracking, emotion detection, and deployment on edge devices for broader accessibility.

**Dataset:** Utilized a combination of custom-collected video data (200+ samples) and the publicly available [Cell Phone Detection Dataset](https://www.kaggle.com/datasets/a165079/cellphoneobjectdetectionusingyolov7) from Kaggle for model training and validation.

**Technologies Used:** Python, YOLOv8, OpenCV, MediaPipe, NumPy, Tkinter, PyAutoGUI, Albumentations, Scikit-learn

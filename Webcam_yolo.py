import cv2
from ultralytics import YOLO

# Load your custom-trained YOLO model from the specified path
model = YOLO("C:/Users/hp/runs/detect/train3/weights/best.pt")  # Replace with the path to your trained model

# Open the default webcam (0) for real-time video captured
cap = cv2.VideoCapture(0)  # 0 means default webcam, or replace with a video file path if needed

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")  # If webcam cannot be accessed, print error
    exit()  # Exit the program if webcam is not available

# Start an infinite loop to continuously capture frames from the webcam
while True:
    ret, frame = cap.read()  # Capture a single frame from the webcam
    if not ret:
        print("Error: Could not read frame.")  # If frame could not be read, print error
        break  # Exit the loop

    # Perform object detection on the current frame using the trained YOLO model
    results = model(frame)  # Pass the frame to the model for prediction

    # Annotate the frame with the detected objects (bounding boxes, labels)
    annotated_frame = results[0].plot()  # Annotates the frame by plotting boxes and class names

    # Check if a mobile phone (or any other object) is detected
    for obj in results[0].boxes:  # Loop through each detected object in the frame
        class_id = int(obj.cls)  # Get the class ID of the detected object (each ID corresponds to a specific class)

        if class_id == 0:  # Check if the class ID corresponds to a mobile phone (adjust the ID as per your model)
            # If mobile phone detected, overlay a warning text on the frame
            cv2.putText(annotated_frame, "Warning: Mobile Phone Detected!", (50, 50),  # Position (50, 50) on the frame
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  # Text color (red) and size

    # Display the annotated frame with bounding boxes and labels
    cv2.imshow("YOLO Object Detection", annotated_frame)  # Show the window with the annotated frame

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Wait for key press; 'q' key to quit
        break

# Release the webcam and close the OpenCV window when done
cap.release()  # Release the webcam to free resources
cv2.destroyAllWindows()  # Close all OpenCV windows

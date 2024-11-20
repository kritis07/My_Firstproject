import cv2       #for handling video capture, image processing, and displaying the output
import mediapipe as mp   # Provides pre-trained models for face and hand tracking
import numpy as np      #for handling arrays and numerical operations
import pyautogui       #Used for creating pop-up alerts (i.e., when suspicious behavior is detected).
import time            # used to track the time when an alert was last triggered to implement a cooldown period.
from tkinter import Tk, Label, Button, messagebox    #A GUI toolkit to create a simple user interface to control the program


# Detection function
def start_detection():
# Initialize MediaPipe solutions for face and hand tracking

#This class is a part of the MediaPipe Face Mesh solution
    mp_face_mesh = mp.solutions.face_mesh
    # Face mesh for detecting facial landmarks
    #Landmarks refer to specific points on the face, such as the corners of the eyes, the tip of the nose, and the contours of the mouth.
    # The FaceMesh model can detect up to 468 facial landmarks for each face in real-time.

#This class is part of the MediaPipe Hands solution,
    mp_hands = mp.solutions.hands  # Hands solution for detecting hand landmarks
   #The model can detect the positions of 21 landmarks per hand, including the key points for each finger, the palm, and the wrist.


# Instantiate face mesh and hand tracking objects

    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    #This argument improves the accuracy of the landmarks.
    # When set to True, the model not only detects the facial landmarks but also refines them, ensuring that the points are more precise.

    hands = mp_hands.Hands()
   #these landmarks are used to detect if a hand is covering the eyes, which could indicate cheating behavior.


    # Parameters for Lucas-Kanade Optical Flow (used for tracking head/eye movement)
#the Lucas-Kanade Optical Flow algorithm is being used to track the movement of features between consecutive frames.
#Optical flow refers to the pattern of apparent motion of objects between two consecutive frames of video caused by the movement of the object or camera.
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
   #winSize stands for the size of the window (in pixels) used to compute the optical flow for each feature point.
       # means that the algorithm will consider a 15x15 pixel window around each feature point to estimate its movement.

   #maxLevel refers to the number of pyramid levels to use for computing the optical flow.
       #means the algorithm will work with 3 pyramid levels (levels 0, 1, and 2).

   #criteria defines termination criteria for the optical flow algorithm, which specifies when the algorithm should stop iterating.
       #the algorithm will stop either after a certain number of iterations or when the change in the flow vectors between iterations is below a given threshold.
        #with two stopping criteria;
                       #1)Stop after a specified number of iterations (10 in this case)
                       #2)Stop if the change in the flow between iterations is less than a small value (0.03 in this case)




    # Shi-Tomasi corner detection parameters (used for tracking features between frames)
                 # used to identify feature points (corners) in an image that can be tracked across consecutive frames.
                     #These feature points are then used in combination with Lucas-Kanade Optical Flow

    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
     #maxCorners specifies the maximum number of corners (or feature points) to detect in the image;100 in this case

     #qualityLevel specifies the quality threshold for selecting corners.
        #value between 0 and 1 that indicates how strong or "good" a feature point must be for it to be considered a valid corner
           # here, the algorithm will retain feature points that have a goodness value greater than 30% of the strongest corner

     # minDistance defines the minimum distance between the detected corners.
         #here, the algorithm will ensure that the corners are at least 7 pixels apart from each other

    #blockSize defines the size of the neighborhood around each pixel to be considered for corner detection
       # here, the corner detection algorithm will use a 7x7 pixel window



    # Optical flow tracking variables
    prev_gray = None  # Previous grayscale frame for optical flow comparison
    prev_points = None  # Previous feature points for tracking movement
   #Initially set to None because the first frame doesn't have any previous frame to compare against.

    # Detection thresholds
    #help determine when to alert the user about potential suspicious behavior based on head movement, eye movement, and hand proximity to the eyes.
    HEAD_MOVEMENT_THRESHOLD = 0.5  # Threshold for detecting significant head movement
    EYE_LOOK_AWAY_THRESHOLD = 0.03  # Threshold for detecting eye look-away movement
    EYE_COVERING_HAND_THRESHOLD = 40  # Threshold for detecting hand proximity to the eye

    # Initialize the webcam feed
    cam = cv2.VideoCapture(0)

    # Cooldown period in seconds to avoid continuous alerts
    alert_cooldown = 5     #The value 5 means that once an alert is triggered, the system will wait for 5 seconds before it can trigger another alert
    last_alert_time = time.time()   #When the first alert is triggered, it records the current time and stores it in last_alert_time
     #By comparing time.time() with last_alert_time, the system can determine if enough time has passed (based on the value of alert_cooldown) before triggering a new alert


    # Function to extract facial landmarks from MediaPipe output and scale them to the frame size
    def extract_landmarks(landmarks, frame_w, frame_h):
        #The function uses a list comprehension to iterate over all the landmarks in the landmarks list
        #The expression lm.x * frame_w scales the x coordinate of each landmark to match the actual width of the image.
        #similarly for y coordinates
        #After calculating the scaled coordinates for each landmark, the list comprehension creates a list of lists
        #This list is converted to a NumPy array
        return np.array([[lm.x * frame_w, lm.y * frame_h] for lm in landmarks])

#therefore ,using this function to get the scaled landmarks of the face
    while True:      # an infinite loop ; to continuously capture frames from the webcam, process them, and perform the necessary actions
        # Read each frame from the webcam
        ret, frame = cam.read()      #ret : A boolean that indicates whether the frame was successfully captured
        if not ret:    #checks if the value of ret is False;there was a problem with capturing the frame
            messagebox.showerror("Error", "Unable to access the webcam.")  # Error handling if webcam fails
            break

        # Flip the frame horizontally (for mirror-like view)
        frame = cv2.flip(frame, 1)       # to flip images or frames ; Flip horizontally (left to right)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for MediaPipe
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale for optical flow

        # Process frame using MediaPipe face mesh and hand tracking
        face_mesh_output = face_mesh.process(rgb_frame)   #detects facial landmarks in the frame
        hand_output = hands.process(rgb_frame)      #detects hand landmarks in the frame

        # Get frame height and width
        frame_h, frame_w, _ = frame.shape

        # Flags for detecting different cheating actions
        eye_detected = False
        hand_covering_eyes = False
        head_movement_detected = False
        #f any of the flags are set to True, and the alert cooldown period has passed, an alert is triggered, warning the user of the suspicious behavior


        # Optical Flow Tracking (detects head movement)
        if prev_gray is not None and prev_points is not None:     #checks whether two important variables, prev_gray and prev_points, are not None
            #A valid previous frame (prev_gray) to compare against
            #A set of feature points (prev_points) from the previous frame that can be tracked to the new frame

            #the function tracks the movement of feature points (stored in prev_points) from a previous frame (prev_gray) to a new frame (gray_frame)
            new_points, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, prev_points, None, **lk_params)
                                  #Tracks the movement of feature points (like corners) from one frame to another using the Lucas-Kanade Optical Flow algorithm
                                          #prev_gray: The previous frame in grayscale
                                          #gray_frame: The current frame in grayscale
                                          #prev_points: The feature points in the previous frame you want to track
                                          #None: A mask (optional, None means no mask)
                                          #**lk_params: Additional parameters (window size, number of pyramid levels, criteria)

                                           #new_points: The new locations of the tracked points
                                           #st: A status vector showing whether each point was successfully tracked
                                           #err: The error of tracking for each point

            # Select the good points (points that were successfully tracked)
            good_new = new_points[st == 1]
            good_old = prev_points[st == 1]
            #Example in tracking eyes;
                    #prev_points would be the positions of the eyes in the previous frame
                    #new_points would be the positions of those same points (the eyes) in the current frame, after the optical flow algorithm has been applied
                    #st indicates whether each of the points (the eyes, in this case) was successfully tracked or not
            #The reason for selecting only the good points (st == 1) is to ensure that the tracking process only works with points that were successfully matched between the two frames


            # Detect major head movement by calculating displacement between frames
            movement = np.linalg.norm(good_new - good_old, axis=1).mean()
                        #This part calculates the Euclidean distance (or displacement) between the corresponding points in good_new and good_old for each tracked point
                             # good_new and good_old represent the successfully tracked points in the current and previous frames respectively
                             #The axis=1 argument ensures the calculation is done for each pair of points (across the x and y coordinates)
                              #.mean() calculates the average displacement of all the tracked points

        #checks if the amount of head movement detected in the current frame (movement) exceeds a predefined threshold (HEAD_MOVEMENT_THRESHOLD)
            if movement > HEAD_MOVEMENT_THRESHOLD and (time.time() - last_alert_time > alert_cooldown):
                                                       #ensures there is a cooldown period between consecutive alerts
                                                       # The purpose is to prevent the system from spamming alerts for every small movement
                                                       #time.time() gives the current time in seconds since the epoch
                                                       # By subtracting last_alert_time from it, you get the time that has passed since the last alert
                                                       #if alert_cooldown = 5, it will not trigger another alert for 5 seconds after the last one.

                head_movement_detected = True    #indicating that significant head movement has been detected in the video feed
                last_alert_time = time.time()    #updates the last_alert_time variable with the current time, recorded using the time.time() function
                pyautogui.alert("Alert: Significant head movement detected! Possible cheating.")
                #notifying the user that significant head movement has been detected

        # Face and Eye Detection with MediaPipe
        if face_mesh_output.multi_face_landmarks:        #checks if face landmarks have been detected in the current frame
            # Extract facial landmarks from the first detected face
            landmarks = face_mesh_output.multi_face_landmarks[0].landmark    # extracts the landmarks from the first detected face in the frame
                        #If there are multiple faces in the frame, you could iterate through all faces, but here we are just using the first face

            current_landmarks = extract_landmarks(landmarks, frame_w, frame_h)
            #function (extract_landmarks) to scale the facial landmarks so they align with the actual frame size of the video
            #converts those normalized coordinates to pixel coordinates based on the actual size of the frame

            # Extract points for left and right eyes
            #current_landmarks is an array of facial landmark points for the detected face
            left_eye = current_landmarks[[159, 145, 144, 133, 153, 154]]    #The indices correspond to key points around the left eye
            right_eye = current_landmarks[[386, 374, 373, 362, 382, 383]]    ##The indices correspond to key points around the right eye

            # Calculate the center of each eye for distance checks with hands
            #The np.mean() function computes the mean (average) of the provided points
            #axis=0 specifies that the mean should be calculated across the rows; the average is taken separately for the x and y coordinates of the points
            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)
            #By taking the average x and y coordinates of the six points around each eye, you can get the center of the eye


            # Hand Detection with MediaPipe
            #a list of detected hands in the current frame
            if hand_output.multi_hand_landmarks:  # Each hand in the list has a set of landmarks (the key points representing the position of the hand and fingers)
                for hand_landmarks in hand_output.multi_hand_landmarks:    #iterates through all detected hands in the current frame
                    hand_coords = extract_landmarks(hand_landmarks.landmark, frame_w, frame_h)
                                  #extracts and scales the coordinates of the hand landmarks based on the width and height of the video frame
                                  #It returns a list of 2D coordinates for each hand's landmark


                    # Check if any hand is covering either of the eyes
                    for hand_coord in hand_coords:      #terates through all the individual hand landmarks
                        distance_to_left_eye = np.linalg.norm(left_eye_center - hand_coord)
                                               #calculates the Euclidean distance between the hand landmark (hand_coord) and the center of the left eye
                        distance_to_right_eye = np.linalg.norm(right_eye_center - hand_coord)    #Similarly, the distance to the right eye is calculated

                    #This condition checks if the hand is close enough to either the left eye or the right eye by comparing it with a threshold distance
                        if distance_to_left_eye < EYE_COVERING_HAND_THRESHOLD or distance_to_right_eye < EYE_COVERING_HAND_THRESHOLD:
                            hand_covering_eyes = True      #signaling that a hand is covering the eye
                            if (time.time() - last_alert_time > alert_cooldown):
                                   #ensures that the alert is not triggered repeatedly in a short amount of time
                                last_alert_time = time.time()   #gives the current time in seconds
                                pyautogui.alert("Alert: Hand covering eye detected! Possible cheating.")
                                     #pop up a message on the screen to warn the user of suspicious behavior

            # Detect eye look-away (if no hand is covering the eyes)
            if prev_points is not None and not hand_covering_eyes:
                #ensures that the previous feature points exist (;there is a valid frame to compare with the current frame)
                #ensures that the detection of eye look-away only occurs when no hand is covering the eyes
                eye_movement = np.linalg.norm(good_new - good_old, axis=1).mean()
                #good_new and good_old are the points that were successfully tracked using Optical Flow; represent the feature points in the current and previous frames, respectively
                                #calculates the Euclidean distance between the current points (good_new) and the previous points (good_old) for each feature
                                #The axis=1 argument ensures the calculation is done for each pair of points (across the x and y coordinates)
                                #.mean() calculates the average displacement of all the tracked points

                if eye_movement > EYE_LOOK_AWAY_THRESHOLD and (time.time() - last_alert_time > alert_cooldown):
                    # checks if the calculated eye movement exceeds a certain threshold
                    eye_detected = True
                    last_alert_time = time.time()
                    pyautogui.alert("Alert: Eye look-away detected! Possible cheating.")

        # Update the previous frame and points for the next iteration
        prev_gray = gray_frame.copy()
           #the current grayscale frame (gray_frame) is copied and stored in prev_gray
           #This will be used as the previous frame in the next iteration of the optical flow detection

        prev_points = cv2.goodFeaturesToTrack(gray_frame, mask=None, **feature_params)
        #This function detects good feature points to track in the next frame using the Shi-Tomasi corner detector
        #these points are used for optical flow tracking in the next iteration of the loop
               #If mask=None, feature points are detected throughout the entire image
               #feature_params holds parameters that control the detection of good features,like the maximum number of corners,the quality of the corners,and the minimum distance between corners.

        #This displays the current processed frame in an OpenCV window with the title "Optical Flow Cheating Detection"
        cv2.imshow('Optical Flow Cheating Detection', frame)

        # Exit loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #waits for a key event for 1 millisecond. If a key is pressed during that time, the key(q) ASCII value is returned
            break

    # Release the webcam and close OpenCV windows after the loop ends
    cam.release()
    cv2.destroyAllWindows()


# Tkinter GUI for the detection system
def main():
    # Create the Tkinter root window
    root = Tk()
    root.title("Cheating Detection System")  #Sets the Window title
    root.geometry("400x200")  #Sets the Window size
    root.configure(bg="black")  # Set background color to black

    # Add label text to the GUI
    Label(root, text="Cheating Detection System", font=("Arial", 16), bg="black", fg="white").pack(pady=10)   #Adds a label to the window with the title of the system
    Label(root, text="Press the button below to start the detection.", bg="black", fg="white").pack(pady=5)   #Adds another label with instructions on how to start the detection

    # Add a button to start the detection process
    Button(root, text="Start Detection", command=start_detection, font=("Arial", 14), bg="green", fg="white").pack(pady=10)

    # Add an exit button to close the program
    Button(root, text="Exit", command=root.destroy, font=("Arial", 14), bg="red", fg="white").pack(pady=10)

    # Start the Tkinter main loop
    root.mainloop()


# Run the GUI and start the program
if __name__ == "__main__":
    main()

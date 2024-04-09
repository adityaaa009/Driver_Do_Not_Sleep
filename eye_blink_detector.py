import cv2
import numpy as np
import time

# Classifier paths (replace with your actual paths)
face_cascade_path = 'haarcascade_frontalface_default.xml'
eye_cascade_path = 'haarcascade_eye_tree_eyeglasses.xml'

# Eye detection parameters
consecutive_no_eyes_frames = 0
blink_threshold = 3  # Adjust this value based on blink sensitivity
alert_display_duration = 1  # Time (in seconds) to display alert message

# Initialize variables for eye closure detection
eyes_closed_start_time = None
alert_start_time = None

# Start video capture
cap = cv2.VideoCapture(0)
ret, img = cap.read()

alert_text = "BE ALERT!"
font = cv2.FONT_HERSHEY_SIMPLEX  # Font for alert text

while ret:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 1, 1)

    # Load the face cascade classifier before the loop
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_face = gray[y:y + h, x:x + w]
            roi_face_clr = img[y:y + h, x:x + w]

            # Load the eye cascade classifier within the loop
            eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

            # Detect eyes within the face region
            eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))

            # Handle eye detection and blinking
            if len(eyes) >= 0.5:  # At least one eye detected
                consecutive_no_eyes_frames = 0
                eyes_closed_start_time = None  # Reset timer if eyes are open
                alert_start_time = None  # Reset alert display timer
            else:  # No eyes detected
                consecutive_no_eyes_frames += 1

                # Check for eyes closed for more than 5 seconds
                if consecutive_no_eyes_frames >= blink_threshold and eyes_closed_start_time is None:
                    eyes_closed_start_time = time.time()  # Start timer
                elif consecutive_no_eyes_frames >= blink_threshold and time.time() - eyes_closed_start_time >= 5:
                    print(alert_text)  # Print alert message to console
                    alert_start_time = time.time()  # Start alert display timer

            # Display alert message for 1 second even after eyes are open
            if alert_start_time is not None and time.time() - alert_start_time < alert_display_duration:
                (text_width, text_height) = cv2.getTextSize(alert_text, font, 1, 2)[0]
                text_offset_x = int((img.shape[1] - text_width) / 2)
                text_offset_y = int(img.shape[0] / 2)
                cv2.putText(img, alert_text, (text_offset_x, text_offset_y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    else:
        cv2.putText(img, "No face detected", (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    cv2.imshow('img', img)
    a = cv2.waitKey(10)
    if a == ord('r'):
        break
    elif a == ord('s'):
        pass  # No need to change

# Import required libraries
import cv2
import numpy as np
import dlib

# Connects to your computer's default camera
cap = cv2.VideoCapture(0)

# Detect the coordinates
detector = dlib.get_frontal_face_detector()

# Capture frames continuously
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # RGB to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        # Get the coordinates of faces
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.putText(frame, 'OK', (x-10, y-10),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        filename = 'faces' + '.jpg'
        cv2.imwrite(filename, frame)
        print("Image Saved -", filename)
        break

cap.release()
cv2.destroyAllWindows()

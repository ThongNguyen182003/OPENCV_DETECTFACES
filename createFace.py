import cv2
import face_recognition
import numpy as np
import os 

# Name input
name = input("Enter name: ")

cap = cv2.VideoCapture(0)

while True:
    # Read img from cap
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    # Display Img
    cv2.imshow("IMG", img)
    
    # save img when 'c' key is pressed
    if cv2.waitKey(1) == ord('c'):
        filename = 'faces/' + name + '.jpg'
        cv2.imwrite(filename, img)
        print("Image Saved -", filename)
        break

# Freeing up resources
cap.release()
cv2.destroyAllWindows()

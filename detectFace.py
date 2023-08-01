import cv2
import face_recognition
import numpy as np
import os 

# Define the path for training
path = 'faces'
#read image from folder
images = []
classnames = []

for img in os.listdir(path):
    image = cv2.imread(f'{path}/{img}')
    images.append(image)
    classnames.append(os.path.splitext(img)[0])
print(classnames)

# Encoded data of the input
def findEncodings(images):
    encodeList = []
    for img in images:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError:
            # If face_recognition.face_encodings cannot find a face in the image, an IndexError will occur
            print(f"Error encoding face in image {img}. Skipping...")
            continue
    return encodeList

knownEncodes = findEncodings(images)
print('Encoding Complete')

# Read the image "faces.jpg"
img = cv2.imread("faces.jpg")

Current_image = img  # No need to resize for a single image
Current_image = cv2.cvtColor(Current_image, cv2.COLOR_BGR2RGB)
    
# Find the face location and encoding it
face_locations = face_recognition.face_locations(Current_image, model='cnn')
face_encodes = face_recognition.face_encodings(Current_image, face_locations)

for encodeFace, faceLocation in zip(face_encodes, face_locations):
    matches = face_recognition.compare_faces(knownEncodes, encodeFace, tolerance=0.6)
    faceDis = face_recognition.face_distance(knownEncodes, encodeFace)
    matchIndex = np.argmin(faceDis)

    # if match found
    if matches[matchIndex]:
        name = classnames[matchIndex].upper()  # display name in videocapture
    else:
        name = 'Unknown'
    # draw and detection and show the identity of the person
    y1, x2, y2, x1 = faceLocation

    # draw rectangle around detected face
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(img, (x1, y2 - 20), (x2, y2), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)

cv2.imshow("IMG", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# OPENCV_DETECTFACES
# Install
```sh
pip install face_recognition
pip install dlib
pip install opencv-python
```
## Program
**createFace.py** is a file used to generate facical images from source dataset, which are then utilized for comparison with new facical. The face will be save in the "faces/" directory.
Then compile file, we get:
```sh
Enter name: 
```
For example, in createFace.py, I can set the name as "Ronaldo" for the generated facial image. Then, I simply press the 'q' key to save the image from Videocapture and exit the program
<div style="text-align:center">
    <img src="faces/Ronaldo.jpg" alt="Image" />
</div>

In the "faces" folder, I have an image named "Ronaldo.jpg". It will be used for comparison with new facial images.  

**Videocapture.py** is a file used to detect and save new faces. When you run it, you will get :
<div style="text-align:center">
    <img src="ok2.png" alt="Image" />
</div>
If the labels attached alongside each face are "ok," then you have successfully completed the process of facial recognition within the frame.  You simply press the 'q' key to save and exit program. The images will be saved in the "faces.jpg" file.<br><br>

**detectFace.py** is a file used to detect, encode, and compare images with "Ronaldo.jpg" and "faces.jpg". Result:

<div style="text-align:center">
    <img src="result.png" alt="Image" />
</div>

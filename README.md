# X-INF573-heart-rater

This project is for measuring the heart rate from video capturing.

## If you want to detect the real-time heart rate, run detector.py
The user-defined parameter **framerate** is the number of frames that the camera takes per second.



## If you want to detect the heart rate from a recorded video, run detector_disk.py
Our recorded video is saved in folder ./video and the GrabCut results for each frame is saved in folder ./segmentation_theo
You can also use your own video but remember to cut the video into frames and change the parameters **framerate** and **nr_frames**(the number of captured frames) that fit your video.


## Pipeline
1) First open webcam and fetch the video feed
2) input the video in the python code
3) detact the face, draw the bounding box for the face
4) detact the forehead or segment the face
5) measure the frequency of color change
6) show the video feed with bounding box and with plotted heart rate curve and freqency.


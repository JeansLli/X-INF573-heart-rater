import numpy as np
import cv2


def main(framerate = 10, scale=1.5):
    """
    Input: The framerate in FPS and the scale of the video window
    Output: The video feed from the webcam
    """
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        #here detact the face
        
        #here manipulate the frame
        cv2.imshow('Input', frame)

        c = cv2.waitKey(int(1000//framerate))
        

    cap.release()
    cv2.destroyAllWindows()


main(framerate = 20)
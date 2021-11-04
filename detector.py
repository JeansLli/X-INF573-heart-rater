import numpy as np
import cv2
import  face_detection


def main(framerate = 10, scale=1.5):
    """
    Input: The framerate in FPS and the scale of the video window
    Output: The video feed from the webcam
    """

    #here later add the feature to use a different webcam, not just the default one
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        #here detact the face
        
        coordinates = face_detection.detact_and_draw_box(frame)
        face_detection.forehead_detection(frame, coordinates)
        #here manipulate the frame
        cv2.imshow('Input', frame)

        c = cv2.waitKey(int(1000//framerate))
        

    cap.release()
    cv2.destroyAllWindows()


main(framerate = 50, scale = 1.5)

import pdb

import numpy as np
import cv2
import  face_detection
import  rate_compute
import pdb

def main(framerate = 20, scale=1.5):
    """
    Input: The framerate in FPS and the scale of the video window
    Output: The video feed from the webcam
    """
    frame_buffer_object = []
    #here later add the feature to use a different webcam, not just the default one
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    counter = 0
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        #here detact the face
        
        coordinates = face_detection.detact_and_draw_box(frame)
        fr, forehead = face_detection.forehead_detection(frame, coordinates)
        #here manipulate the frame
        #cut out the face here

        if forehead.shape[0]==0:
            continue

        frame_buffer_object.append(forehead)
        #how many frames do we want to pass?
        if(len(frame_buffer_object)==(framerate*5)):
            #remove the first frame from the buffer
            print("frame buffer object",len(frame_buffer_object))
            frame_buffer_object.pop(0)
            rate_compute.detect_change(frame_buffer_object)
        #here call t
        #pass the pictures to the function to compute the buffer
        '''
        if(counter<500):
            cv2.imwrite(
                "/Users/jingyili/Documents/ip-paris/courses_taken/INF573/project/data_face/pic_%i.png" % counter,
                forehead)
        counter += 1
        '''
        #tuning of the parameters of the buffer size and the framerate to see when it is good.

        cv2.imshow('Input', fr)

        c = cv2.waitKey(int(1000//framerate))


    cap.release()
    cv2.destroyAllWindows()


main(framerate = 10, scale = 1.5)

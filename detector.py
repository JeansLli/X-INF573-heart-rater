import pdb

import numpy as np
import cv2
import  face_detection
import  rate_compute
import pdb
from scipy import ndimage

def main(framerate = 20, scale=0.1):
    """
    Input: The framerate in FPS and the scale of the video window
    Output: The video feed from the webcam
    """
    frame_buffer_object = []
    #here later add the feature to use a different webcam, not just the default one
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('./video/testing.mp4')

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    counter = 0
    while True:
        ret, frame = cap.read()
        #frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
        
        #here detact the face
        #rotate the frame if necessary
        #frame = ndimage.rotate(frame, 180+90)
        coordinates = face_detection.detact_and_draw_box(frame, False)
        #pdb.set_trace()
        if np.array(coordinates).any()==0:
            continue
        else:
            #fr, forehead = face_detection.forehead_detection(frame, coordinates)
            fr, forehead = face_detection.face_segmentation(frame, coordinates)
            #here manipulate the frame
            #cut out the face here
            frame_buffer_object.append(forehead)
            #how many frames do we want to pass?
            #print("len(frame_buffer_object)=",len(frame_buffer_object))

            #only if we add the forehead we want to augment the counter
            counter += 1
        if(len(frame_buffer_object)==(framerate)*30):
            #remove the first frame from the buffer
            #print("frame buffer object",len(frame_buffer_object))
            frame_buffer_object.pop(0)
            rate_compute.detect_change(frame_buffer_object, 1/framerate, counter)
        else:
            print('counter', len(frame_buffer_object))
        #here call t
        #pass the pictures to the function to compute the buffer
        '''
        if(counter<500):
            cv2.imwrite(
                "/Users/jingyili/Documents/ip-paris/courses_taken/INF573/project/data_face/pic_%i.png" % counter,
                forehead)
        '''
        
        
        #tuning of the parameters of the buffer size and the framerate to see when it is good.
        cv2.namedWindow('Input',cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('Input', 500,500)
        cv2.imshow('Input', forehead)

        c = cv2.waitKey(int(1000//framerate))


    cap.release()
    cv2.destroyAllWindows()


main(framerate = 3, scale = 1.5)

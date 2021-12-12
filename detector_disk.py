import pdb

import numpy as np
import cv2
import  face_detection
import  rate_compute
import pdb
from scipy import ndimage

import  matplotlib.pyplot as plt




def determine_hr(framerate, nr_frames):
    """
    Input: The framerate of the video, the number of captured frames
    """

    frame_buffer=[]
    cap = cv2.VideoCapture('video/theo_video_test.mp4')
    for j in range(1,nr_frames):
        
        ret, frame = cap.read()

        forehead = cv2.imread("/home/theo/Documents/ecole_polytechnique/computer_vision/project/segmentation_theo/img_%i.png"%j)
        frame_buffer.append(forehead)
        if(len(frame_buffer)==5*framerate):
            frame_buffer.pop(0)
            rate_compute.detect_change(frame_buffer, 1/framerate, j)
        else:
            #rate_compute.detect_change(frame_buffer, 1/framerate, j)
            print('counter', len(frame_buffer))

        cv2.namedWindow('segmentation face',cv2.WINDOW_NORMAL)
        cv2.imshow('segmentation face', forehead)
        cv2.imshow('full face', frame)
        #how many frames are there
        cv2.waitKey(1000//framerate)
    #pdb.set_trace()


determine_hr(30,1206)
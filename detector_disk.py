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
    for j in range(1,nr_frames):

        forehead = cv2.imread("/home/theo/Documents/ecole_polytechnique/computer_vision/project/X-INF573-heart-rater/data_face/img_%i.png"%j)
        frame_buffer.append(forehead)
        if(len(frame_buffer)==10*framerate):
            frame_buffer.pop(0)
            rate_compute.detect_change(frame_buffer, 1/framerate, j)
        else:
            print('counter', len(frame_buffer))
        cv2.namedWindow('segmentation face',cv2.WINDOW_NORMAL)
        cv2.imshow('segmentation face', forehead)
        #how many frames are there
        cv2.waitKey(1000//framerate)
    #pdb.set_trace()


determine_hr(30,2016)
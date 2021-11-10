import numpy as np
import cv2
import matplotlib.pyplot as plt

#create the data to test the function
"""
buffer = []
for j in range(20):
    im = cv2.imread('/home/theo/Documents/ecole_polytechnique/computer_vision/project/X-INF573-heart-rater/test_forhead/pic_%i.png'%j)
    buffer.append(im)
"""

def detect_change(buffer_object):
    """Input: buffer object, a sequence of frames
    """
    min_y = buffer_object[0].shape[0]
    min_x = buffer_object[0].shape[1]
    for j in range(len(buffer_object)):
        if(buffer_object[j].shape[1]<min_x):
            min_x = buffer_object[j].shape[1]
        if(buffer_object[j].shape[0]<min_y):
            min_y = buffer_object[j].shape[0]
        

    for idx in range(len(buffer_object)):
        #cut the pictures in the right size
        buffer_object[idx] = buffer_object[idx][:min_y,:min_x,::]
    
    

#detect_change(buffer)
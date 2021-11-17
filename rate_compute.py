import numpy as np
import cv2
import matplotlib.pyplot as plt

#create the data to test the function

buffer = []
for j in range(150):
    im = cv2.imread('/home/theo/Documents/ecole_polytechnique/computer_vision/project/X-INF573-heart-rater/test_forhead/pic_%i.png'%j)
    buffer.append(im)


def detect_change(buffer_object):
    """
    Input: buffer object, a sequence of frames
    Output: 
    """
    
    min_y = buffer_object[0].shape[0]
    min_x = buffer_object[0].shape[1]
    for j in range(len(buffer_object)):
        #if the programm failed to generate a 
        if(buffer_object[j].shape[0]==0):
            buffer_object[j] = buffer_object[j-1]
        else:
            if(buffer_object[j].shape[1]<min_x):
                min_x = buffer_object[j].shape[1]
            if(buffer_object[j].shape[0]<min_y):
                min_y = buffer_object[j].shape[0]
        

    for idx in range(len(buffer_object)):
        #cut the pictures in the right size
        buffer_object[idx] = buffer_object[idx][:min_y,:min_x,::]
    
    #for idx in range(len(buffer_object)):
    #    print(buffer_object[j].shape)
    #calculate the maximum red green and blue value, plot it over the frame
    
    #maybe better change to numpy objects.. see whether this is a problem.
    
    #red_channel = buffer_object[:,:,:,0]
    buffer_np = np.array(buffer_object)
    red_channel = buffer_np[:,:,:,0]
    green_channel = buffer_np[:,:,:,1]
    blue_channel = buffer_np[:,:,:,2]
    
    #compute the mean per image to get an array that corresponds to the 
    x_red = red_channel.mean(axis=(1,2))
    x_green = green_channel.mean(axis=(1,2))
    x_blue = blue_channel.mean(axis=(1,2))

    #normalize the vector over the whole time
    x_red =  x_red/np.linalg.norm(x_red)
    x_green =  x_green/np.linalg.norm(x_green)
    x_blue =  x_blue/np.linalg.norm(x_blue)
    
    
    
    

detect_change(buffer)
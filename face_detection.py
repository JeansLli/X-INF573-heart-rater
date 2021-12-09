import cv2 as cv2
import numpy as np
from skimage import io
import pdb
from matplotlib import pyplot as plt

#only works well for people without masks
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')




def detact_and_draw_box(frame, drawing):
    """
    Input: A frame 
    Output: The frame with bounding box on the face, only the largest one.
    """

    #convert to grayscale
    #cv2.imshow('frame',frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    max = 0
    coords = (0,0,0,0)
    for (x, y, w, h) in faces:
        #only draw the one with maximal area. the others are probably noise.
        if(w*h>max):
            coords = (x, y, w, h)
            max = w*h


    (x, y, w, h) = coords

    ########## This adaption is for graphcut
    #h = int(h*1.35)
    #y = int(y*0.5)
    #h=h-30
    #y=y+20
    ##############
    coords = (x, y, w, h)
    if drawing==True:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    return coords
    


#img = cv2.imread("Input_screenshot_03.11.2021.png")
#detact_and_draw_box(img)

def forehead_detection(frame, coordinates):
    """
    Input: A Frame and coordinates
    Output: A sliced frame, that only contains the forehead.
    """
    start_x = coordinates[0] + int(1/4*coordinates[2])
    stop_x = coordinates[0] + int(3/4*coordinates[2])
    start_y = coordinates[1] + int(1/20*coordinates[3])
    stop_y= coordinates[1] + int(1/6*coordinates[3])
    cv2.rectangle(frame, (start_x, start_y), (stop_x, stop_y), (0, 255, 0), thickness=2)

    #cut this part out then, then return it to analyse further.
    frame_mod = frame[start_y:stop_y,start_x:stop_x]
    #white_bg = 255*np.ones_like(frame)
    #white_bg[start_y:stop_y,start_x:stop_x] = frame_mod
    return (frame,frame_mod)

    #use the empiric value of skin proportions to cut out the forehead.



def face_segmentation(frame, coordinates):
    mask = np.zeros(frame.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    rect = np.array(coordinates)

    #this is necessary to augment the size of the bounding box in order to fill in the whole face in a 
    #rectangle, that we can pass to the graph cut segmentation function.
    rect[1] =int(0.5*rect[1])
    rect[3] = int(rect[3]*1.6)
    print("rect=",rect)
    #pdb.set_trace()
    cv2.grabCut(frame,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    
    graph_face = frame * mask2[:,:,np.newaxis]
    #graph_face = np.asarray(graph_face) 
    #plt.imshow(graph_face)
    #plt.colorbar()
    #plt.show()
    return(frame, graph_face)

if __name__=="__main__":
    frame = cv2.imread("./video/jingyi.jpg")
    coordinates = detact_and_draw_box(frame)
    #pdb.set_trace()
    #fr, forehead = face_detection.forehead_detection(frame, coordinates)
    fr, graphcut_face = face_segmentation(frame, coordinates)
    #cv2.imwrite('./video/jingyi_cut.jpg', graphcut_face)
    #cv2.imshow('graphcut_face',graphcut_face)
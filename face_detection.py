import cv2 as cv2
from skimage import io

#only works well for people without masks
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detact_and_draw_box(frame):
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
    

    #use the empiric value of skin proportions to cut out the forehead.

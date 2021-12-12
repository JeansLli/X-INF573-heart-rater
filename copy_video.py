import cv2
import time
import face_detection
import numpy as np



# define a video capture object
cap = cv2.VideoCapture('video/theo_video_test.mp4')
counter=0
seconds = time.time()
while(True):
	
	# Capture the video frame
	# by frame
	ret, frame = cap.read()

	# Display the resulting frame
	#cv2.imshow('frame', frame)
	coordinates = face_detection.detact_and_draw_box(frame, False)
	if np.array(coordinates).any()==0:
		continue
	else:
		#fr, forehead = face_detection.forehead_detection(frame, coordinates)
		fr, forehead = face_detection.face_segmentation(frame, coordinates)
		counter+=1
		cv2.imwrite('/home/theo/Documents/ecole_polytechnique/computer_vision/project/segmentation_theo/img_%i.png' % counter, forehead)
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	print('counter', counter)
	

	

# After the loop release the cap object
# Destroy all the windows
cv2.destroyAllWindows()
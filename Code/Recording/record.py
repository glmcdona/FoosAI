import cv2
import sys
import os


if( len(sys.argv) == 2 ):
	result = sys.argv[1]
	print("Recording to %s" % result)
	
	camera = 2 # First webcam attached to PC
	
	cap = cv2.VideoCapture(camera)
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(result,fourcc, 20.0, (640,480))
	
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:
			out.write(frame)

			# Display the resulting frame
			cv2.imshow('frame', frame)
			
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			
	cap.release()
	out.release()
	cv2.destroyAllWindows()
	
else:
	print("record.py <output file.avi>")
	


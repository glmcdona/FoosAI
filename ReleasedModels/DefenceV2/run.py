import cv2
import keras
import pprint
import sys
import os
import numpy as np
from PIL import Image
import imageio
import itertools as it
pp = pprint.PrettyPrinter(depth=6)
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import serial # conda install pyserial
import struct

class Viewpoint(object):
	def __init__(self, cam_x, cam_y, cam_w, cam_h, resize_w, resize_h, num_frames):
		self.cam_x = cam_x
		self.cam_y = cam_y
		self.cam_w = cam_w
		self.cam_h = cam_h

		self.resize_w = resize_w
		self.resize_h = resize_h
		self.num_frames = num_frames
		self.data = np.zeros(shape=(1, num_frames, self.resize_h, self.resize_w, 3), dtype=np.float32)

	def get_cam(self):
		return (self.cam_x, self.cam_y, self.cam_w, self.cam_h)
	
	def set_top_left(self, x, y):
		self.cam_x = x
		self.cam_y = y
	
	def top_left(self):
		return (self.cam_x, self.cam_y)
	
	def bottom_right(self):
		return (self.cam_x + self.cam_w, self.cam_y + self.cam_h)
	
	def set_width(self, width):
		self.cam_w = width
		self.cam_h = round(width * (self.resize_h / self.resize_w))
		
	def process_frame(self, frame):
		# Crop the frame, resample, and normalize
		frame_resized = cv2.resize(frame[self.cam_y:(self.cam_y+self.cam_h), self.cam_x:(self.cam_x+self.cam_w)], (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)

		batch_data = np.zeros(shape=(1, 1, self.resize_h, self.resize_w, 3), dtype=np.float32)
		image = Image.fromarray(frame_resized)
		norm_image = np.array(image, dtype=np.float32)
		norm_image -= 128.0
		norm_image /= 128.0
		
		# Update our current batch of frames
		
		# Shift the old frames
		self.data[0,0,:,:,:] = self.data[0,1,:,:,:]
		self.data[0,1,:,:,:] = self.data[0,2,:,:,:]
		
		# Add the new frame
		self.data[0,self.num_frames-1,:,:,:] = np.ascontiguousarray(norm_image)
		
		return self.data
		
	def process_frame_no_resize(self, frame):
		# Crop the frame, resample, and normalize
		frame_resized = frame[self.cam_y:(self.cam_y+self.cam_h), self.cam_x:(self.cam_x+self.cam_w)]
		return frame_resized


moving = False
def click_callback(event, x, y, flags, param):
	global refPt
	global moving
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt[2] = refPt[2] + x - refPt[0]
		refPt[3] = refPt[3] + y - refPt[1]
		refPt[0] = x
		refPt[1] = y
		moving = True
		
	if event == cv2.EVENT_LBUTTONUP or moving == True:
		if x - refPt[0] > 50:
			refPt[2] = x
			refPt[3] = y
		
		if event == cv2.EVENT_LBUTTONUP:
			moving = False

class Foosbot(object):
	def __init__(self, viewpoint, model_dpos_file = None, model_pos_file = None, video_file = 0, ser = None):
		self.viewpoint = viewpoint
		self.video = cv2.VideoCapture(video_file)
		self.ser = ser
		self.crop = False

		# Load the models
		self.model_dpos_file = None
		self.model_pos_file = None
		if model_dpos_file is not None:
			self.model_dpos_file = keras.models.load_model(model_dpos_file)
		#model_pos_file = None
		if model_pos_file is not None:
			self.model_pos_file = keras.models.load_model(model_pos_file)

	def _process_frame(self, frame):
		# Evaluate the frame and output the resulting frames
		dpos = None
		pos = None

		if self.model_pos_file is not None:
			# Evalaute difference in position
			pos = self.model_pos_file.predict(frame)[0,:]
			#pos = (0.0,0.0,0.0)

		if self.model_dpos_file is not None:
			# Evalaute difference in position
			dpos = self.model_dpos_file.predict(frame)[0,:]
			#dpos = (0.0,0.0,0.0)

		return (pos, dpos)

	def run(self, display=True):
		if display:
			global refPt
			cv2.namedWindow("FoosBot")
			cv2.setMouseCallback("FoosBot", click_callback)

		count = 0
		while(self.video.isOpened()):
			if display:
				self.viewpoint.set_top_left(refPt[0], refPt[1])
				self.viewpoint.set_width(refPt[2]-refPt[0])
			
			#	cv2.setMouseCallback("FoosBot", click_callback)

			# Read in the next frame
			ret, frame = self.video.read()

			# Crop and stretch it
			frame_processed = self.viewpoint.process_frame(frame)

			# Evaluate the Foosbot ML models
			(pos, dpos) = self._process_frame(frame_processed)
			
			# Output the desired rod position deltas to the Arduino driving the robot
			if not self.ser is None:
				#self.ser.write( struct.pack('3f', *dpos) )
				self.ser.write( (str(dpos[1]) + "\n").encode() )
				# In the arduino code to read a single float:
				#float f;
				#...
				#if (Serial.readBytes((char*)&f, sizeof(f)) != sizeof(f)) {
			
				if ser.inWaiting() >= 0:
					line = self.ser.read(ser.inWaiting()) 
					print("Serial: " + str(line))
			
			# Display the result
			if display:
				# Only display the cropped region?
				if self.crop:
					frame = self.viewpoint.process_frame_no_resize(frame)
				
				# Draw the frame
				font = cv2.FONT_HERSHEY_SIMPLEX
				
				cv2.putText(frame,'%i' % (count),(10,50), font, 1,(255,255,255),1,cv2.LINE_AA)
				if not dpos is None:
					line_step = 20
					x = 10
					y = 90
					arrow_x = 120
					arrow_width = 200
					
					cv2.putText(frame,'FoosAI dPos:',(10,y), font, 1,(255,255,255),1,cv2.LINE_AA)
					y += line_step
					
					cv2.putText(frame,'%.2f' % (dpos[2]),(10,y), font, 1,(255,255,255),1,cv2.LINE_AA)
					cv2.arrowedLine(frame, (int(arrow_x+arrow_width/2),int(y-line_step/2)), (int(arrow_x+arrow_width/2 + dpos[2]*arrow_width*5),int(y-line_step/2)), (255,0,0), thickness=3)
					y += line_step
					
					cv2.putText(frame,'%.2f' % (dpos[1]),(10,y), font, 1,(255,255,255),1,cv2.LINE_AA)
					cv2.arrowedLine(frame, (int(arrow_x+arrow_width/2),int(y-line_step/2)), (int(arrow_x+arrow_width/2 + dpos[1]*arrow_width*5),int(y-line_step/2)), (255,0,0), thickness=3)
					y += line_step
					
					cv2.putText(frame,'%.2f' % (dpos[0]),(10,y), font, 1,(255,255,255),1,cv2.LINE_AA)
					cv2.arrowedLine(frame, (int(arrow_x+arrow_width/2),int(y-line_step/2)), (int(arrow_x+arrow_width/2 + dpos[0]*arrow_width*5),int(y-line_step/2)), (255,0,0), thickness=3)
					y += line_step
					
				if not pos is None:
					line_step = 20
					x = 10
					y = 200
					arrow_x = 120
					arrow_width = 200
					
					cv2.putText(frame,'FoosAI Pos:',(10,y), font, 1,(255,255,255),1,cv2.LINE_AA)
					y += line_step
					
					cv2.putText(frame,'%.2f' % (pos[2]),(10,y), font, 1,(255,255,255),1,cv2.LINE_AA)
					cv2.line(frame, (int(arrow_x),int(y-line_step/2)), (int(arrow_x+arrow_width),int(y-line_step/2)), (255,255,255), thickness=5)
					cv2.line(frame, (int(arrow_x),int(y-line_step/2)), (int(arrow_x+arrow_width*pos[2]),int(y-line_step/2)), (0,0,0), thickness=5)
					y += line_step
					
					cv2.putText(frame,'%.2f' % (pos[1]),(10,y), font, 1,(255,255,255),1,cv2.LINE_AA)
					cv2.line(frame, (int(arrow_x),int(y-line_step/2)), (int(arrow_x+arrow_width),int(y-line_step/2)), (255,255,255), thickness=5)
					cv2.line(frame, (int(arrow_x),int(y-line_step/2)), (int(arrow_x+arrow_width*pos[1]),int(y-line_step/2)), (0,0,0), thickness=5)
					y += line_step
					
					cv2.putText(frame,'%.2f' % (pos[0]),(10,y), font, 1,(255,255,255),1,cv2.LINE_AA)
					cv2.line(frame, (int(arrow_x),int(y-line_step/2)), (int(arrow_x+arrow_width),int(y-line_step/2)), (255,255,255), thickness=5)
					cv2.line(frame, (int(arrow_x),int(y-line_step/2)), (int(arrow_x+arrow_width*pos[0]),int(y-line_step/2)), (0,0,0), thickness=5)
					y += line_step
				
				if not self.crop:
					cv2.rectangle(frame, self.viewpoint.top_left(), self.viewpoint.bottom_right(), (255,0,0), 2 )
				
				cv2.imshow('FoosBot',frame)
				

			# Keystroke to quit or pause
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				break
			elif key == ord('c'):
				self.crop =  not self.crop
			elif key == ord(' '):
				cv2.waitKey()

			count += 1

		cv2.destroyAllWindows()


refPt = [5,5,600+5,330+5]
view = Viewpoint(cam_x = refPt[0], cam_y = refPt[1], cam_w = refPt[2], cam_h = refPt[3], resize_w = 100, resize_h = 54, num_frames = 3)


print("Note: If Python crashes, I've found that closing any other python apps using the GPU fixes the issue. Eg. close the Jupyter notebook used for training.")
if( len(sys.argv) == 2 ):
	#ser = serial.Serial('COM3', 115200) # Communcating to the arduino controller that runs to robot
	ser = None
	
	if sys.argv[1] == "simulate":
		#video_file = ".\\..\\..\\TrainingData\\Raw\\Pro1\\2017 Hall of Fame Classic 2.mp4"
		video_file = ".\\..\\..\\TrainingData\\Raw\\Am3\\out_fix2.avi"
		foosbot = Foosbot( ser = ser, viewpoint = view, model_dpos_file = "dpos_cnn_models_79.h5", model_pos_file = "pos_cnn_models_11.h5", video_file = video_file)
		foosbot.run()
	elif sys.argv[1] == "run":
		video_file = 2 # First webcam attached to PC
		foosbot = Foosbot( ser = ser, viewpoint = view, model_dpos_file = "dpos_cnn_models_79.h5", model_pos_file = "pos_cnn_models_11.h5", video_file = video_file)
		foosbot.run()
else:
	print("run.py <simulate OR run>")
	


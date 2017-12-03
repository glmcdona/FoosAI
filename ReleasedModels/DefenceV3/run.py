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
import scipy.misc

class ViewpointModel(object):
	def __init__(self, name, crop, resize_w, resize_h, num_frames, model, blackouts = None):
		self.cam_x = crop[0]
		self.cam_y = crop[1]
		self.cam_w = crop[2]
		self.cam_h = crop[3]
		self.model = model
		self.name = name
		self.blackouts = blackouts
		

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
	
	def draw_data(self, frame, text, arrow_numerics, range_numerics):
		line_step = 20
		x = self.cam_x + 10
		y = self.cam_y + line_step
		font = cv2.FONT_HERSHEY_SIMPLEX
		
		# Draw the box
		cv2.rectangle(frame, self.top_left(), self.bottom_right(), (255,0,0), 1 )
		
		# Draw the lines of text
		if text is not None:
			# Draw the text lines
			lines = text.split("\n")
			for line in lines:
				cv2.putText(frame,line,(10,y), font, 1,(255,255,255),1,cv2.LINE_AA)
				y += line_step
		
		# Draw the arrows
		arrow_x = 120
		arrow_width = 200
		if arrow_numerics is not None:
			for numeric in arrow_numerics:
				cv2.putText(frame,'%.2f' % (numeric),(10,y), font, 1,(255,255,255),1,cv2.LINE_AA)
				cv2.arrowedLine(frame, (int(arrow_x+arrow_width/2),int(y-line_step/2)), (int(arrow_x+arrow_width/2 + numeric*arrow_width),int(y-line_step/2)), (255,0,0), thickness=3)
				y += line_step
		
		# Draw the range numerics
		if range_numerics is not None:
			for numeric in range_numerics:
				cv2.putText(frame,'%.2f' % (numeric),(10,y), font, 1,(255,255,255),1,cv2.LINE_AA)
				cv2.line(frame, (int(arrow_x),int(y-line_step/2)), (int(arrow_x+arrow_width),int(y-line_step/2)), (255,255,255), thickness=5)
				cv2.line(frame, (int(arrow_x),int(y-line_step/2)), (int(arrow_x+arrow_width*numeric),int(y-line_step/2)), (0,0,0), thickness=5)
				y += line_step
		
		return frame
	

	def process_frame(self, frame):
		# Process any blackout regions
		if self.blackouts is not None:
			frame = frame.copy()
			for blackout in self.blackouts:
				frame[blackout[1]:(blackout[1] + blackout[3]), blackout[0]:(blackout[0] + blackout[2]),:] = 0
		
		# Crop the frame, resample, and normalize
		#frame_resized = scipy.misc.imresize(frame[self.cam_y:(self.cam_y+self.cam_h), self.cam_x:(self.cam_x+self.cam_w)], (self.resize_h, self.resize_w) )
		
		#frame_crop = frame[self.cam_y:(self.cam_y+self.cam_h), self.cam_x:(self.cam_x+self.cam_w)]
		frame_resized = cv2.resize(frame[self.cam_y:(self.cam_y+self.cam_h), self.cam_x:(self.cam_x+self.cam_w)], (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)
		
		image = Image.fromarray(frame_resized)
		norm_image = np.array(image, dtype=np.float32)
		norm_image -= 128.0
		norm_image /= 128.0
		
		# Update our current batch of frames
		
		# Shift the old frames
		for i in range(self.num_frames-1):
			self.data[0,i,:,:,:] = self.data[0,i+1,:,:,:]
		
		# Add the new frame
		self.data[0,self.num_frames-1,:,:,:] = norm_image
		
		#if self.name == "Rod1":
		#	pp.pprint(self.data[0,0,:,:,:])
		
		cv2.imshow(self.name,self.data[0,0,:,:,:])
		
		return self.model.predict(self.data)[0,:]
		
	def get_cropped_frame(self, frame):
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
	def __init__(self, rod_models = [], foosbot_model = None, video_file = 0, ser = None):
		self.video = cv2.VideoCapture(video_file)
		self.ser = ser
		self.crop = False
		self.pause_output = False
		
		# Set the models
		self.rod_models = rod_models
		self.foosbot_model = foosbot_model

	def run(self, display=True):
		if display:
			global refPt
			cv2.namedWindow("FoosBot")
			cv2.setMouseCallback("FoosBot", click_callback)

		count = 0
		while(self.video.isOpened()):
			if display:
				self.foosbot_model.set_top_left(refPt[0], refPt[1])
				self.foosbot_model.set_width(refPt[2]-refPt[0])
			
			# Read in the next frame
			ret, frame = self.video.read()

			# Evaluate the FoosAI rod position prediction model
			desired_rod2_pos = self.foosbot_model.process_frame(frame)
			controlled_rod = 1 # Controlling the twobar, rod 2
			
			# Evaluate all the rod models
			rods = []
			for rod in self.rod_models:
				rods.append(rod.process_frame(frame))
			
			# Output the desired rod position deltas to the Arduino driving the robot
			if not self.ser is None and not self.pause_output:
				#self.ser.write( struct.pack('3f', *dpos) )
				self.ser.write( (str(desired_rod2_pos-rods[controlled_rod]) + "\n").encode() )
				# In the arduino code to read a single float:
				#float f;
				#...
				#if (Serial.readBytes((char*)&f, sizeof(f)) != sizeof(f)) {
			
				if ser.inWaiting() >= 0:
					line = self.ser.read(ser.inWaiting()) 
					print("Serial: " + str(line))
			
			# Display the result
			if display:
				# Draw the frame
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(frame,'%i' % (count),(10,50), font, 1,(255,255,255),1,cv2.LINE_AA)
				
				# Draw the rod information
				for i, rod in enumerate(self.rod_models):
					if i == controlled_rod:
						frame = rod.draw_data(frame, "Controlled Rod", arrow_numerics = [desired_rod2_pos-rods[i]], range_numerics = [rods[i], desired_rod2_pos])
					else:
						frame = rod.draw_data(frame, None, arrow_numerics = None, range_numerics = [rods[i]])
				
				# Draw the foosbot region
				cv2.rectangle(frame, self.foosbot_model.top_left(), self.foosbot_model.bottom_right(), (255,0,0), 2 )
				
				# Only display the cropped region?
				if self.crop:
					frame = self.foosbot_model.get_cropped_frame(frame)
				cv2.imshow('FoosBot',frame)
				

			# Keystroke to quit or pause
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				break
			elif key == ord('c'):
				self.crop =  not self.crop
			elif key == ord(' '):
				cv2.waitKey()
			elif key == ord('p'):
				self.pause_output =  not self.pause_output

			count += 1

		cv2.destroyAllWindows()


<<<<<<< HEAD
rod_model = keras.models.load_model("rod_pos.h5")
Rod1 = [0,224,640,140]
Rod2 = [0,106,640,140]
Rod3 = [0,20,640,140]
model_pos_rod1 = ViewpointModel(name = "Rod1", crop = Rod1, resize_w = 160, resize_h = 30, num_frames = 1, model=rod_model)
model_pos_rod2 = ViewpointModel(name = "Rod2", crop = Rod2, resize_w = 160, resize_h = 30, num_frames = 1, model=rod_model)
model_pos_rod3 = ViewpointModel(name = "Rod3", crop = Rod3, resize_w = 160, resize_h = 30, num_frames = 1, model=rod_model)
=======
rod_model = keras.models.load_model("pos_cnn_models_0.h5")
Rod1 = [0,224,640,140]
Rod2 = [0,106,640,140]
Rod3 = [0,20,640,140]
model_pos_rod1 = ViewpointModel(name = "Rod1", crop = Rod1, resize_w = 160, resize_h = 35, num_frames = 1, model=rod_model)
model_pos_rod2 = ViewpointModel(name = "Rod2", crop = Rod2, resize_w = 160, resize_h = 35, num_frames = 1, model=rod_model)
model_pos_rod3 = ViewpointModel(name = "Rod3", crop = Rod3, resize_w = 160, resize_h = 35, num_frames = 1, model=rod_model)
>>>>>>> 6ba377c01e2b6784dfd3ab5223d4887802bc0734
rods = [model_pos_rod1, model_pos_rod2, model_pos_rod3]


foos_ai_model = keras.models.load_model("pos_cnn_models_43.h5")
Table = [0,0,640,360]
global refPt
refPt = Table
model_2bar = ViewpointModel(name = "FoosAI", crop = Table, resize_w = 100, resize_h = 54, num_frames = 3, model=foos_ai_model, blackouts=[Rod2])


print("Note: If Python crashes, I've found that closing any other python apps using the GPU fixes the issue. Eg. close the Jupyter notebook used for training.")
if( len(sys.argv) == 2 ):
	#ser = serial.Serial('COM3', 115200) # Communcating to the arduino controller that runs to robot
	ser = None
	
	if sys.argv[1] == "simulate":
		#video_file = ".\\..\\..\\TrainingData\\Raw\\Pro1\\2017 Hall of Fame Classic 2.mp4"
		video_file = ".\\..\\..\\TrainingData\\Raw\\Am1\\out.avi"
		foosbot = Foosbot( ser = ser, rod_models = rods, foosbot_model = model_2bar, video_file = video_file)
		foosbot.run()
	elif sys.argv[1] == "run":
		#ser = serial.Serial('COM3', 115200) # Communcating to the arduino controller that runs to robot
		ser = None
		video_file = 2 # Webcam attached to PC
		foosbot = Foosbot( ser = ser, rod_models = rods, foosbot_model = model_2bar, video_file = video_file)
		foosbot.run()
else:
	print("run.py <simulate OR run>")
	


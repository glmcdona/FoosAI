import cv2
import keras
from keras.models import Model
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
import numpy.ma as ma

def make_mosaic(imgs, nrows, ncols, border=0):
    """
	https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[-1]
    imshape = imgs.shape[0:2]
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[:,:,i]
    return mosaic


class ViewpointModel(object):
	def __init__(self, name, crop, resize_w, resize_h, num_frames, model, blackouts = None, crop_after_resize = None):
		self.cam_x = crop[0]
		self.cam_y = crop[1]
		self.cam_w = crop[2]
		self.cam_h = crop[3]
		self.model = model
		self.name = name
		self.blackouts = blackouts
		self.crop_after_resize = crop_after_resize
		

		self.resize_w = resize_w
		self.resize_h = resize_h
		self.num_frames = num_frames
		
		if self.crop_after_resize is None:
			self.data = np.zeros(shape=(1, num_frames, self.resize_h, self.resize_w, 3), dtype=np.float32)
		else:
			self.data = np.zeros(shape=(1, num_frames, self.crop_after_resize[2], self.crop_after_resize[3], 3), dtype=np.float32)

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
		#cv2.rectangle(frame, self.top_left(), self.bottom_right(), (255,0,0), 1 )
		
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
	
	def visualize(self, frame):
		# Visualize the result. The result is assumed to be a 3d or 4d volume, and is
		# shown as an image. This is intended to visualize intermediate CNN stages.
		data = self.process_frame(frame, False)
		
		mosaic = make_mosaic(data, 7, 6) # (7, 20, 40) to grid of (7 by 20 images)
		#pp.pprint(mosaic)
		mosaic = cv2.resize(mosaic*10,None,fx=7, fy=7, interpolation = cv2.INTER_CUBIC)
		
		cv2.imshow(self.name, mosaic) 
		

	def process_frame(self, frame, show = True):
		# Process any blackout regions
		if self.blackouts is not None:
			frame = frame.copy()
			for blackout in self.blackouts:
				frame[blackout[1]:(blackout[1] + blackout[3]), blackout[0]:(blackout[0] + blackout[2]),:] = 0
		
		# Crop the frame, resample, and normalize
		#frame_crop = frame[self.cam_y:(self.cam_y+self.cam_h), self.cam_x:(self.cam_x+self.cam_w)]
		frame_resized = cv2.resize(frame[self.cam_y:(self.cam_y+self.cam_h), self.cam_x:(self.cam_x+self.cam_w)], (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)
		
		if self.crop_after_resize is not None:
			# Apply the after-resize crop
			frame_resized = frame_resized[self.crop_after_resize[0]:self.crop_after_resize[0]+self.crop_after_resize[2],self.crop_after_resize[1]:self.crop_after_resize[1]+self.crop_after_resize[3],:]
		
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
		
		if show:
			cv2.imshow(self.name,self.data[0,0,:,:,:])
		
		return np.squeeze(self.model.predict(self.data))
		
	def get_cropped_frame(self, frame):
		# Crop the frame, resample, and normalize
		frame_resized = frame[self.cam_y:(self.cam_y+self.cam_h), self.cam_x:(self.cam_x+self.cam_w)]
		if self.crop_after_resize is not None:
			# Apply the after-resize crop
			frame_resized = frame_resized[self.crop_after_resize[0]:self.crop_after_resize[0]+self.crop_after_resize[2],self.crop_after_resize[1]:self.crop_after_resize[1]+self.crop_after_resize[3],:]
		
		return frame_resized

		import numpy.ma as ma


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
	def __init__(self, controlled_rod = 0, rod_models = [], foosbot_model = None, visualize_models = None, video_file = 0, ser = None, visualize_freq = 10):
		self.video = cv2.VideoCapture(video_file)
		self.ser = ser
		self.crop = False
		self.pause_output = False
		self.visualize = True
		self.visualize_freq = visualize_freq
		self.controlled_rod = controlled_rod
		
		# Set the models
		self.rod_models = rod_models
		self.foosbot_model = foosbot_model
		self.visualize_models = visualize_models

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
			
			# Evaluate all the rod models
			rods = []
			for i, rod in enumerate(self.rod_models):
				if i == self.controlled_rod or self.visualize:
					rods.append(rod.process_frame(frame))
				else:
					rods.append(0.0) # Don't calculate this rod position to save on compute time
			
			# Output the desired rod position deltas to the Arduino driving the robot
			if not self.ser is None and not self.pause_output:
				#self.ser.write( struct.pack('3f', *dpos) )
				self.ser.write( (str(desired_rod2_pos-rods[self.controlled_rod]) + "\n").encode() )
				# In the arduino code to read a single float:
				#float f;
				#...
				#if (Serial.readBytes((char*)&f, sizeof(f)) != sizeof(f)) {
			
				if ser.inWaiting() >= 0:
					line = self.ser.read(ser.inWaiting()) 
					print("Serial: " + str(line))
			
			# Visualize the intermediate layers
			if self.visualize:
				if self.visualize_models is not None:
					for visualize_model in self.visualize_models:
						visualize_model.visualize(frame)
			
			# Display the result
			if display:
				# Draw the frame
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(frame,'%i' % (count),(10,50), font, 1,(255,255,255),1,cv2.LINE_AA)
				
				# Draw the rod information
				for i, rod in enumerate(self.rod_models):
					if i == self.controlled_rod:
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
			elif key == ord('v'):
				self.visualize =  not self.visualize
			elif key == ord('p'):
				self.pause_output =  not self.pause_output

			count += 1

		cv2.destroyAllWindows()


rod_model = keras.models.load_model("rod_pos.h5")
Rod1 = [0,224,640,140]
Rod2 = [0,106,640,140]
Rod3 = [0,20,640,140]
model_pos_rod1 = ViewpointModel(name = "Rod1", crop = Rod1, resize_w = 160, resize_h = 35, num_frames = 1, model=rod_model)
model_pos_rod2 = ViewpointModel(name = "Rod2", crop = Rod2, resize_w = 160, resize_h = 35, num_frames = 1, model=rod_model)
model_pos_rod3 = ViewpointModel(name = "Rod3", crop = Rod3, resize_w = 160, resize_h = 35, num_frames = 1, model=rod_model)

rods = [model_pos_rod1, model_pos_rod2, model_pos_rod3]





foos_ai_model = keras.models.load_model("foosai_2frames.h5")
Table = [0,10,640,120] # Almost the same as Rod3, just cropped a bit smaller
global refPt
refPt = Table
model_2bar = ViewpointModel(name = "FoosAI", crop = Table, resize_w = 80, resize_h = 15, num_frames = 2, model=foos_ai_model, blackouts=None, crop_after_resize=None)


# Visualization
layer_name = 'conv3d_5' # (None, 1, 7, 20, 40)
intermediate_layer_model = Model(inputs=foos_ai_model.input,
                                 outputs=foos_ai_model.get_layer(layer_name).output)
#intermediate_output = intermediate_layer_model.predict(data)
model_visualize = ViewpointModel(name = "conv3 7x20x40", crop = Table, resize_w = 80, resize_h = 15, num_frames = 2, model=intermediate_layer_model, blackouts=None, crop_after_resize=None)

print("Note: If Python crashes, I've found that closing any other python apps using the GPU fixes the issue. Eg. close the Jupyter notebook used for training.")

if( len(sys.argv) == 2 ):
	if sys.argv[1] == "simulate":
		ser = None
		#video_file = ".\\..\\..\\TrainingData\\Raw\\Pro1\\2017 Hall of Fame Classic 2.mp4"
		video_file = ".\\..\\..\\TrainingData\\Raw\\Am2\\out.mp4"
		foosbot = Foosbot( controlled_rod = 0, ser = ser, rod_models = rods, foosbot_model = model_2bar, video_file = video_file, visualize_models = [model_visualize])
		foosbot.run()
	elif sys.argv[1] == "run":
		ser = serial.Serial('COM3', 115200) # Communcating to the arduino controller that runs to robot
		video_file = 2 # Webcam attached to PC
		foosbot = Foosbot( controlled_rod = 0, ser = ser, rod_models = rods, foosbot_model = model_2bar, video_file = video_file)
		foosbot.run()
else:
	print("run.py <simulate OR run>")
	


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

class Viewpoint(object):
	def __init__(self, cam_x, cam_y, cam_w, cam_h, resize_w, resize_h):
		self.cam_x = cam_x
		self.cam_y = cam_y
		self.cam_w = cam_w
		self.cam_h = cam_h

		self.resize_w = resize_w
		self.resize_h = resize_h

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
		batch_data[0,0,:,:,:] = np.ascontiguousarray(norm_image)

		return batch_data


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
	def __init__(self, viewpoint, model_dpos_file, model_pos_file, video_file = 0):
		self.viewpoint = viewpoint
		self.video = cv2.VideoCapture(video_file)

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
			#dpos = self.model_dpos_file.predict(frame)[0,:]
			dpos = (0.0,0.0,0.0)

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

			if display:
				# Draw the frame
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(frame,'%i: outputs: (%.2f,%.2f,%.2f)' % (count, pos[0], pos[1], pos[2]),(10,50), font, 1,(255,255,255),1,cv2.LINE_AA)
				
				cv2.rectangle(frame, self.viewpoint.top_left(), self.viewpoint.bottom_right(), (255,0,0), 2 )
				
				cv2.imshow('FoosBot',frame)
				

			# Keystroke to quit or pause
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				break
			elif key == ord(' '):
				cv2.waitKey()

			count += 1

		cv2.destroyAllWindows()


refPt = [5,5,600+5,330+5]
view = Viewpoint(cam_x = refPt[0], cam_y = refPt[1], cam_w = refPt[2], cam_h = refPt[3], resize_w = 100, resize_h = 54)


if( len(sys.argv) == 2 ):
    if sys.argv[1] == "simulate":
        video_file = ".\\..\\..\\TrainingData\\Raw\\Am1\\out.avi"
        foosbot = Foosbot( viewpoint = view, model_dpos_file = "config12_deltapos.h5", model_pos_file = "config12_model_posonly.hdf", video_file = video_file)
        foosbot.run()
    elif sys.argv[1] == "run":
        video_file = 0 # First webcam attached to PC
        foosbot = Foosbot( viewpoint = view, model_dpos_file = "config12_deltapos.h5", model_pos_file = "config12_model_posonly.hdf", video_file = video_file)
        foosbot.run()
else:
    print("run.py <simulate OR run>")
    


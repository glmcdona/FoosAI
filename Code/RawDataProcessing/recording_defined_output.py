import cv2
import xml.etree.ElementTree as ET
import os
from os import listdir
from os.path import isfile, join
from rod import *

import pprint
pp = pprint.PrettyPrinter(depth=6)

def xml_load_point(point):
	return (int(point.split(',')[0]), int(point.split(',')[1]))

def xml_load_rect(rect):
	return (int(rect.find("X").text), int(rect.find("Y").text), int(rect.find("W").text), int(rect.find("H").text))

def xml_load_rgb(text):
	tokens = text.split(",")
	return (int(tokens[0]), int(tokens[1]), int(tokens[2]))
	

global refPt
refPt = (0,0)

def click_callback(event, x, y, flags, param):
	global refPt
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = (x, y)
		cropping = True

class RecordingDefinedOutput(object):
	def __init__(self, recording_file, models):
		self.recording_file = recording_file
		self.tree = ET.parse(recording_file)
		self.root = self.tree.getroot()
		base_path = os.path.dirname(recording_file)
		
		# Files
		self.folder = os.path.join(base_path, self.root.find("RECORDING_FOLDER").find("FOLDER").text)
		self.video_files = [os.path.join(self.folder,f) for f in listdir(self.folder) if isfile(join(self.folder, f))]
		self.video_index = 0
		self.video_frame = 0
		self.video_frame_count = 0
		self.video_cap = None
		self.has_more = True
		self.num_frames = 0
		
		# Settings
		self.weight = int(self.root.find("RECORDING_FOLDER").find("WEIGHT").text)
		self.defined_output = [float(x) for x in self.root.find("RECORDING_FOLDER").find("OUTPUT").text.split(",")]
		
		# Apply the weighting by repeating the video files in order
		self.video_files = self.video_files * self.weight
		
		print("Loaded %i video files at a defined output of %s with a weight of %i" % (len(self.video_files), str(self.defined_output), self.weight))
		
	
	def __del__(self):
		if self.video_cap is not None:
			self.video_cap.release()
	
	def initialize(self):
		self.has_more = True
		self.video_index = 0
		self.video_frame = 0
		self.video_frame_count = 0
		self.video_cap = None
	
	def get_next_frame(self):
		# Returns:
		# (cropped frame, {'Output' = rod positions})
		
		if self.video_cap is None and self.video_index < len(self.video_files):
			# Open the current video
			self.video_cap = cv2.VideoCapture(self.video_files[self.video_index])
			self.has_more = True
			self.frame = 0
			self.frame_count = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
			print("Opened %s with %i frames" % (self.video_files[self.video_index], self.frame_count))
		
		if self.video_cap is None:
			self.has_more = False
			return (None, None)
		
		if( self.video_cap.isOpened() and self.has_more ):
			ret, frame = self.video_cap.read()
			if ret==True:
				# Calculate the output for this frame
				rate_through_cap = self.frame / (self.frame_count - 1)
				output = self.defined_output[0] + (self.defined_output[1] - self.defined_output[0]) * rate_through_cap
				
				self.frame += 1
				return (frame, {"Output": output})
			else:
				print("Finished capture file")
				self.video_cap = None
				self.video_index += 1
				return self.get_next_frame()
		
		return (None, None)
		
		
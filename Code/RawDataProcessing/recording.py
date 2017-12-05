import cv2
import xml.etree.ElementTree as ET
import os
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

class Recording(object):
	def __init__(self, recording_file, models, blackouts, crop_to_rod):
		self.recording_file = recording_file
		self.tree = ET.parse(recording_file)
		self.root = self.tree.getroot()
		base_path = os.path.dirname(recording_file)
		
		# Blackouts to hide any rods
		self.blackouts = blackouts # List of rod names to hide
				
		# Crops
		self.crop = None
		self.crop_to_rod = crop_to_rod
		if self.crop_to_rod is None:
			if self.root.find("CROP") is not None:
				self.crop = xml_load_rect(self.root.find("CROP").find("RECT"))
		
		# Files
		self.file_avi = os.path.join(base_path, self.root.find("RECORDING").find("AVI").text)
		self.frame_start = int(self.root.find("RECORDING").find("STARTFRAME").text)
		self.frame_end = int(self.root.find("RECORDING").find("ENDFRAME").text)
		
		# Rod tracking settings
		rod_tracking_alignment = None
		rod_tracking_gap_colour = None
		rod_tracking_gap_colour_distance = None
		rod_tracking_gap_min_size = None
		rod_tracking_rod_width = None
		self.line_detection_frequency = 1
		self.model = None
		
		if self.root.find("RODS").find("TRACKING").find("MODEL") is not None:
			# Load the corresponding tracking model
			self.model = models[self.root.find("RODS").find("TRACKING").find("MODEL").text]
		else:
			# Use standard line-detection based rod tracking
			rod_tracking_alignment = self.root.find("RODS").find("TRACKING").find("ALIGNMENT").text
			rod_tracking_gap_colour = xml_load_rgb(self.root.find("RODS").find("TRACKING").find("GAP_COLOUR").text)
			rod_tracking_gap_colour_distance = int(self.root.find("RODS").find("TRACKING").find("GAP_COLOUR_DISTANCE").text)
			rod_tracking_gap_min_size = int(self.root.find("RODS").find("TRACKING").find("GAP_MIN_SIZE").text)
			rod_tracking_rod_width = int(self.root.find("RODS").find("TRACKING").find("ROD_WIDTH").text)
			
			if self.root.find("RODS").find("TRACKING").find("LINE_DETECTION_FREQUENCY") is None:
				self.line_detection_frequency = 30
			else:
				self.line_detection_frequency = int(self.root.find("RODS").find("TRACKING").find("LINE_DETECTION_FREQUENCY").text)
		
		
		# Video file
		self.cap = cv2.VideoCapture(self.file_avi)
		self.num_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
		self.frame = 0
		self.has_more = False
		
		# Rods
		self.rods = {}
		for rod in self.root.find("RODS").iter("ROD"):
			rod_left = None
			rod_right = None
			if self.model is not None:
				if rod.find("TRACKING") is None or rod.find("TRACKING").find("LEFT") is None:
					rod_left = None
				else:
					rod_left = xml_load_point(rod.find("TRACKING").find("LEFT").text)
					
				if rod.find("TRACKING") is None or rod.find("TRACKING").find("RIGHT") is None:
					rod_right = None
				else:
					rod_right = xml_load_point(rod.find("TRACKING").find("RIGHT").text)
			
			rod_name = rod.find("NAME").text
			self.rods[rod_name] = Rod( xml_load_rect(rod.find("RECT")),rod_name, rod_tracking_gap_colour, rod_tracking_gap_colour_distance, rod_tracking_gap_min_size, rod_tracking_rod_width, self.line_detection_frequency, rod_left, rod_right, self.model)
			
			
			
	
	def __del__(self):
		self.cap.release()
	
	def process(self):
		# Extract the rod positions
		pass
	
	def initialize(self):
		self._initialize_rod_gaps()
		self.cap.set(1,self.frame_start)
		self.frame = self.frame_start
		self.has_more = True
	
	def get_next_frame(self):
		# Returns:
		# (cropped frame, {rod name = rod positions})
		if( self.cap.isOpened() and self.has_more ):
			font = cv2.FONT_HERSHEY_SIMPLEX
			ret, frame = self.cap.read()
			if ret==True and self.frame < self.frame_end:
				self.frame += 1
				
				frame_with_markup = frame.copy()
				
				# Only do edge-processing if we aren't using a ML model to extract position
				if self.model is None:
					gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
					edges = cv2.Canny(gray,50,150,apertureSize = 3)
				else:
					edges = None
				
				# Draw the crop box
				if self.crop is not None:
					box = self.crop
					cv2.rectangle(frame_with_markup, (box[0], box[1]),
										(box[0] + box[2], box[1] + box[3]), 
										(0,255,0) );
				
				# Process each rod
				rod_positions = {}
				failure_count = 0
				for rod_name, rod in self.rods.items():
					# Update rod tracking
					if rod.rod_left is None and self.model is None:
						# Update from the rod line
						if self.frame % self.line_detection_frequency == 0:
							rod.update_rod_line(edges)
					
					
					# Get this rod position
					if rod.rod_line is not None or rod.model is not None:
						(new_x, line, success) = rod.track_rod_position(frame)
						if( not success ):
							failure_count += 1
						rod_positions[rod_name] = new_x
						
					
					# Update the graphics
					box = rod.box
					if rod_name in self.blackouts:
						# Black box to indicate it is being cut out
						cv2.rectangle(frame_with_markup, (box[0], box[1]),
										(box[0] + box[2], box[1] + box[3]), 
										(0,0,0), 2 );
					else:
						# Show regular box
						cv2.rectangle(frame_with_markup, (box[0], box[1]),
										(box[0] + box[2], box[1] + box[3]), 
										(255,0,0), 3 );
					
					# Draw the rod line
					rod_line = rod.rod_line
					if rod_line is not None:
						cv2.line(frame_with_markup,rod_line[0],rod_line[1],(0,0,255),2)
						
						if line is not None:
							rod_offence_last_frame_x = new_x
							cv2.line(frame_with_markup, line[0], line[1], (255,0,255), thickness=4)
					
					# Draw this rod position
					cv2.putText(frame_with_markup,'Pos %.2f' % (new_x),(box[0]+30, box[1]+30), font, 1,(255,255,255),1,cv2.LINE_AA)
					
				# Black out the rods
				for blackout in self.blackouts:
					frame = self.rods[blackout].blackout(frame)
				
				# Crop the frame
				if self.crop is None:
					frame_cropped = frame
				else:
					frame_cropped = frame[self.crop[1]:(self.crop[1]+self.crop[3]), self.crop[0]:(self.crop[0]+self.crop[2])]
				
				# Crop to the specific rod
				if self.crop_to_rod is not None and self.crop_to_rod in self.rods:
					frame_cropped = self.rods[self.crop_to_rod].get_rod_region(frame_cropped)
				#cv2.imshow('image',frame_cropped)
				#key = cv2.waitKey(1)
				
				# Add text
				cv2.putText(frame_with_markup,'%i - %i,%i: rgb: %i,%i,%i' % (self.frame, refPt[0],refPt[1],frame[refPt[1]][refPt[0]][0],frame[refPt[1]][refPt[0]][1],frame[refPt[1]][refPt[0]][2]),(10,50), font, 1,(255,255,255),1,cv2.LINE_AA)
				
				return (frame_cropped, frame_with_markup, rod_positions, failure_count)
			else:
				print("frame read failure")
				self.has_more = False
				return (None, None, None, 0)
		
		return (None, None, None, 0)
	
	def _initialize_rod_gaps(self):
		print("Loading rod gap tracking sizes...")
		key_frame_count = 30
		#key_frame_count = 5
		small_frame_set = []
		for i in range(key_frame_count):
			frame = i * (self.num_frames/key_frame_count)
			self.cap.set(1,frame)
			
			ret, frame = self.cap.read()
			small_frame_set.append(frame)
		
		# Load the rod gap sizes to track from a small subset of the frames
		for rod_name, rod in self.rods.items():
			# Find the gap size
			size = rod.find_gap_size(small_frame_set)
			print("\t%s: gap size %i" % (rod_name, size))
	
	def play(self):
		self.initialize()
		
		# Extract the rod positions
		cv2.namedWindow("image")
		
		global refPt
		
		(frame, frame_with_markup, rod_positions, failure_count) = self.get_next_frame()
		
		while frame is not None:
			cv2.setMouseCallback("image", click_callback)
			
			cv2.imshow('image',frame_with_markup)
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				break
			elif key == ord(' '):
				cv2.waitKey()
			
			(frame, frame_with_markup, rod_positions, failure_count) = self.get_next_frame()
			
			
		print("Finished after processing %i frames." % self.frame)
		cv2.destroyAllWindows()
		
		
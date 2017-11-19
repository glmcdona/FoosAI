

import numpy as np
from PIL import Image
import imageio
import itertools as it

import cv2

import pprint
pp = pprint.PrettyPrinter(depth=6)


# Class for tracking rod position from camera frames	 
class Rod():
	def __init__(self, box, name, gap_colour_edge = (22,28,39),  gap_bar_colour_threshold = 70, gap_min_size = 40, gap_bar_width = 5, line_detection_frequency = 30, rod_left = None, rod_right = None):
		self.box = box
		self.name = name
		self.gap_colour_edge = gap_colour_edge
		self.gap_bar_colour_threshold = gap_bar_colour_threshold
		self.gap_min_size = gap_min_size
		self.gap_bar_width = gap_bar_width
		
		self.gap_tracking_size = None
		self.colour_left = gap_colour_edge
		self.colour_right = gap_colour_edge
		self.line_detection_frequency = line_detection_frequency
		
		self.rod_line = None
		self.rod_left = rod_left
		self.rod_right = rod_right
		if rod_left is not None:
			self.rod_line = (rod_left,rod_right)
		self._last_frame_pos = 0
	
	def _rod_column_is_edge(self, frame, x_range, y_range, colour):
		sum_distance = 0
		count = 0
		for x in x_range:
			for y in y_range:
				distance = np.linalg.norm(frame[y][x]-colour)
				sum_distance += distance
				count += 1
		
		if sum_distance != 0 and sum_distance/count < self.gap_bar_colour_threshold:
			return True
		return False
		
	def _rod_column_get_colour(self, frame, x_range, y_range):
		sums = [0,0,0]
		count = 0
		for x in x_range:
			for y in y_range:
				sums[0] += frame[y][x][0]
				sums[1] += frame[y][x][1]
				sums[2] += frame[y][x][2]
				count += 1
		
		return [sums[0]/count, sums[1]/count, sums[2]/count]
	
	def update_rod_line(self, edges):
		if self.rod_left is not None:
			return True
		
		edges = edges[self.box[1]:(self.box[1] + self.box[3]), self.box[0]:(self.box[0] + self.box[2])]
		lines = cv2.HoughLines(edges,1,np.pi/180,200)
		if lines is not None:
			for rho,theta in lines[0]:
				a = np.cos(theta)
				b = np.sin(theta)
				x0 = a*rho
				y0 = b*rho
				x1 = int(x0 + 1000*(-b))
				y1 = int(y0 + 1000*(a))
				x2 = int(x0 - 1000*(-b))
				y2 = int(y0 - 1000*(a))
				
				# Update the rod line
				self.rod_line = ((x1+self.box[0],y1+self.box[1]),(x2+self.box[0],y2+self.box[1]))
				return True

		# Don't update the rod line. Continue using last tracking tline
		return False
	
	def track_rod_position(self, frame):
		# Searches the box within the rod line for the specified gap, and
		# returns it's resulting position.
		
		max_frame_movement = self.gap_tracking_size / 2
		x_left = None
		y_left = None

		# Search for the right-most edge based on the last frame
		#pp.pprint(range(int(min(self._last_frame_pos + max_frame_movement,np.shape(frame)[1])), int(max(self._last_frame_pos - max_frame_movement,0))))
		for x in range(int(min(self._last_frame_pos + max_frame_movement,np.shape(frame)[1]-2)), int(max(self._last_frame_pos - max_frame_movement,0,self.rod_line[0][0])), -1):
			y_center = round(( float( x - self.rod_line[0][0] ) / float(self.rod_line[1][0] - self.rod_line[0][0]) )
						 * float( self.rod_line[1][1] - self.rod_line[0][1] )) + self.rod_line[0][1]

			if self._rod_column_is_edge(frame, [x], range(y_center-self.gap_bar_width,y_center+self.gap_bar_width), self.colour_left) and not self._rod_column_is_edge(frame, [x+1], range(y_center-self.gap_bar_width,y_center+self.gap_bar_width), self.colour_left) and not self._rod_column_is_edge(frame, [x+2], range(y_center-self.gap_bar_width,y_center+self.gap_bar_width), self.colour_left):
				# Match, found our left edge
				x_left_colour = self._rod_column_get_colour(frame, [x], range(y_center-self.gap_bar_width,y_center+self.gap_bar_width))
				x_left = x
				y_left = y_center
				#print("Found left edge close to old edge at x=%i." % x_left)
				break

		if x_left is not None:
			# Search right, making sure we have no edge until approx gap
			x_right = None
			x_right_first_empty = False
			for x in range(int(x_left + self.gap_tracking_size - round(self.gap_tracking_size / 10.0)), int(min(x_left + self.gap_tracking_size + round(self.gap_tracking_size / 10.0), np.shape(frame)[1] - 2))):
				y_center = round(( float( x - self.rod_line[0][0] ) / float(self.rod_line[1][0] - self.rod_line[0][0]) )
						 * float( self.rod_line[1][1] - self.rod_line[0][1] )) + self.rod_line[0][1]

				if not self._rod_column_is_edge(frame, [x], range(y_center-self.gap_bar_width,y_center+self.gap_bar_width), self.colour_right) and self._rod_column_is_edge(frame, [x+1], range(y_center-self.gap_bar_width,y_center+self.gap_bar_width), self.colour_right) and self._rod_column_is_edge(frame, [x+2], range(y_center-self.gap_bar_width,y_center+self.gap_bar_width), self.colour_right):
					# Match, found our right edge
					x_right = x
					x_right_colour = self._rod_column_get_colour(frame, [x+1], range(y_center-self.gap_bar_width,y_center+self.gap_bar_width))
					#print("Found right close to expectation at x=%i." % x_right)
					break
				else:
					x_right_first_empty = True

			if x_right is not None and x_right_first_empty == True:
				self._last_frame_pos = x_left
				#self.colour_left = x_left_colour
				#self.colour_right = x_right_colour
				return (x_left, ((x_left, y_left), (x_right, y_center)), True)
			#print("Failed to find right close to expected at x=%i." % (x_left + self.gap_tracking_size))

		if self._last_frame_pos == 0:
			last_pos = None
			#print("Raw search for the gap")
			for x in range(max(0,self.rod_line[0][0]),np.shape(frame)[1]-1):
				y_center = round(( float( x - self.rod_line[0][0] ) / float(self.rod_line[1][0] - self.rod_line[0][0]) )
							 * float( self.rod_line[1][1] - self.rod_line[0][1] )) + self.rod_line[0][1]

				if self._rod_column_is_edge(frame, [x], range(y_center-self.gap_bar_width,y_center+self.gap_bar_width), self.colour_left) and not self._rod_column_is_edge(frame, [x+1], range(y_center-self.gap_bar_width,y_center+self.gap_bar_width), self.colour_left):
					if last_pos is not None:
						new_gap = np.linalg.norm(x-last_pos[0])
						#gap = np.linalg.norm((x,y_center)-last_pos)
						if abs(new_gap-self.gap_tracking_size) < self.gap_tracking_size/5.0:
							print("Edge found")
							# Use this gap position
							#print("Found the gap at %i" % last_pos[0])
							self._last_frame_pos = last_pos[0]
							return (last_pos[0], ((last_pos[0],last_pos[1]), (x,y_center)), True)
					last_pos = np.array([x,y_center])
		

		# No result. Return last position, no line
		return (self._last_frame_pos, None, False)
	
	def find_gap_size(self, key_frames):
		# Find the longest distance between two black blobs
		# Idea: Yellow players tracks distance between two bumpers
		#	   Red players tracks distance between two players
		gaps = []

		for frame in key_frames:
			# Get the pixel on the line, and one row up and one row down
			gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			edges = cv2.Canny(gray,50,150,apertureSize = 3)

			last_pos = None
			
			if self.rod_left is not None or self.update_rod_line(edges):
				# Found the rod line
				for x in range(0,np.shape(frame)[1]):
					y_center = round(( float( x - self.rod_line[0][0] ) / float(self.rod_line[1][0] - self.rod_line[0][0]) )
									 * float( self.rod_line[1][1] - self.rod_line[0][1] )) + self.rod_line[0][1]
					
					if self._rod_column_is_edge(frame, [x], range(y_center-self.gap_bar_width,y_center+self.gap_bar_width), self.colour_left):
						if last_pos is not None:
							gap = np.linalg.norm(x-last_pos[0])
							if gap > self.gap_min_size:
								gaps.append( round(gap) )
								#cv2.circle(frame, (x,y_center), 4, (0,255,25))
								#cv2.circle(frame, tuple(last_pos), 4, (0,255,25))
						last_pos = np.array([x,y_center])

			#cv2.imshow('image',frame)
			#pp.pprint(gaps)
			#key = cv2.waitKey()

		# Pick the single most common gap size accross the frames
		self.gap_tracking_size = max(set(gaps), key=gaps.count)
		return self.gap_tracking_size
	
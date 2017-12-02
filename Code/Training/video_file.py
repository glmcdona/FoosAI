from __future__ import print_function

import cv2
import sys
import os
import csv
import numpy as np
import random
from random import randint

import threading

from PIL import Image
import imageio
import itertools as it

import scipy.misc

from keras.preprocessing.image import *

import pprint
pp = pprint.PrettyPrinter(depth=6)

def normalize_frame(data):
	'''
	Based on http://vlg.cs.dartmouth.edu/c3d/c3d_video.pdf
	crop.
	'''
	image = Image.fromarray(data)

	norm_image = np.array(image, dtype=np.float32)
	norm_image -= 128.0
	norm_image /= 128.0

	# (height, width, channels)
	data = np.ascontiguousarray(norm_image)
	#cv2.imshow("norm",data[:,:,:])
	#cv2.waitKey(0)
	return data

class Chunk(object):
	'''
	Simple class that wraps video frames loaded in memory
	'''

	def __init__(self, video_file, position_file, min_positions, max_positions, position_rel_indexes, frame_rel_indexes, validation_rate, shuffle, transformer = None):
		# Load the position data for each frame
		f = open(position_file)
		self.positions = []
		for line in f.readlines():
			self.positions.append( list(map(float, line.split("\t")) ) )
		f.close()
		
		self.transformer = transformer
		
		# Normalize the position data on the range [0.0, 1.0]
		self.positions_norm = []
		for position in self.positions:
			for idx, value in enumerate(position):
				if min_positions is not None and max_positions is not None:
					position[idx] = float(position[idx] - min_positions[idx]) / float(max_positions[idx] - min_positions[idx])
					if position[idx] < 0:
						position[idx] = 0
					if position[idx] > 1:
						position[idx] = 1
			self.positions_norm.append(position)
		
		self.output_size = len(self.positions_norm[0])*len(position_rel_indexes)
		#pp.pprint("Output size %i" % (self.output_size))

		# Camera frames in memory
		self.validation_rate = validation_rate
		self.video_file = video_file

		print(self.video_file)
		video_reader = imageio.get_reader(self.video_file)
		self.num_frames   = len(video_reader)
		first_frame = normalize_frame(video_reader.get_data(0))
		self.input_width = np.shape(first_frame)[1]
		self.input_height = np.shape(first_frame)[0]
		
		if self.transformer is not None:
			(first_frame, junk) = self.transformer._process_frame(first_frame, [0]*500)
			
		
		pp.pprint(np.shape(first_frame))
		self.width = np.shape(first_frame)[1]
		self.height = np.shape(first_frame)[0]
		
		#pp.pprint("Width: %i, Height %i" % (self.width, self.height))
		video_reader.close()
		
		self.reader = cv2.VideoCapture(self.video_file)
		self.lock = threading.Lock()

		frames = list(range(-min(frame_rel_indexes), self.num_frames - max(position_rel_indexes) ))
		
		if shuffle:
			# Pick random frames for validation and training
			random.shuffle(frames)
		
		self.frames_training = frames[0:round(self.num_frames * (1.0-validation_rate))]
		self.frames_validation = frames[round(self.num_frames * (1.0-validation_rate)):-1]
		self.size_training = len(self.frames_training)
		self.size_validation = len(self.frames_validation)

		self.current_frame_training = 0
		self.current_frame_validation = 0
		self.position_rel_indexes = position_rel_indexes
		self.frame_rel_indexes = frame_rel_indexes
		
		self.shuffle = shuffle
		
		self.reset()
	


	def get_frame(self, index):
		# video_data: [frame#, x, y, channels]
		
		# Prepare the frames
		with self.lock:
			frames = np.zeros(shape=(len(self.frame_rel_indexes), self.input_height, self.input_width, 3), dtype=np.float32)
			count = 0
			for idx, rel_idx in enumerate(self.frame_rel_indexes):
				if self.reader.isOpened():
					self.reader.set(1, index + rel_idx)
					ret, frame = self.reader.read()
					if ret == True and frame is not None:
						frames[idx, :, :, :] = normalize_frame(frame)[:,:,:]
						count += 1
					else:
						break
		
		if count != len(self.frame_rel_indexes):
			return (None, None)
		
		# Load the sequence of output positions
		output = []
		for idx, rel_idx in enumerate(self.position_rel_indexes):
			output += list(self.positions_norm[index+rel_idx][:])


		if self.transformer is None:
			return (frames, output)

		return self.transformer.process_set(frames, output)

	def reset(self):
		if self.shuffle:
			random.shuffle(self.frames_training)
		self.current_frame_training = 0
		self.current_frame_validation = 0

	def get_next_training_frame(self):
		# Returns:
		# ([frames], [training outputs])
		if self.current_frame_training < len(self.frames_training):
			# Load the sequence of frames
			(frames, output) = self.get_frame( self.frames_training[self.current_frame_training])
			self.current_frame_training += 1
			return (frames, output)
		else:
			# Reached the end
			return (None, None)


	def get_next_validation_frame(self):
		 # Returns:
		# ([frames], [training outputs])
		if self.current_frame_validation < len(self.frames_validation):
			# Load the sequence of frames
			(frames, output) = self.get_frame( self.frames_validation[self.current_frame_validation])
			self.current_frame_validation += 1
			return (frames, output)
		else:
			# Reached the end
			return (None, None)
			
	def get_training_count(self):
		return len(self.frames_training)
	
	def get_validation_count(self):
		return len(self.frames_validation)


# Based on Keras ImageDataGenerator, but modified for video files:
# 	https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
class VideoTransform():
	# Transforms a frame+output pair for better generalization
	def __init__(self, zoom_range=0.1, rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, shear_range= 0.1, fill_mode='nearest', vertical_flip=False, horizontal_flip=True, horizontal_flip_invert_indices = [3,4,5], horizontal_flip_reverse_indices = [0,1,2], data_format=None, random_crop=None, resize=None):

		# Camera frame transform logic
		self.cval = 0.
		self.rotation_range = rotation_range
		self.width_shift_range = width_shift_range
		self.height_shift_range = height_shift_range
		self.shear_range = shear_range
		self.fill_mode = fill_mode
		self.data_format = data_format # 'channels_first' or 'channels_last'
		if data_format not in {'channels_last'}:
			raise ValueError('`data_format` should be `"channels_last"`')
		if data_format == 'channels_last':
			self.frame_axis = 0
			self.row_axis = 1
			self.col_axis = 2
			self.channel_axis = 3
		self.seed = 1
		self.seed_random = 1
		self.prng = np.random.RandomState(self.seed)
		self.random_crop = random_crop
		self.output_size = resize
		if self.output_size is None and self.random_crop is not None:
			self.output_size = self.random_crop

		# Zoom range
		if np.isscalar(zoom_range):
			self.zoom_range = [1 - zoom_range, 1 + zoom_range]
		elif len(zoom_range) == 2:
			self.zoom_range = [zoom_range[0], zoom_range[1]]
		else:
			raise ValueError('`zoom_range` should be a float or '
							 'a tuple or list of two floats. '
							 'Received arg: ', zoom_range)

		# Vertical flip
		self.vertical_flip = vertical_flip

		# Horizontal flipping logic
		self.horizontal_flip = horizontal_flip
		self.horizontal_flip_invert_indices = horizontal_flip_invert_indices # flip sign on these
		self.horizontal_flip_reverse_indices = horizontal_flip_reverse_indices # reverse on the range 0 to 1.

	def new_random_transform(self):
		# Selects new random transforms
		self.seed_random = self.prng.randint(0,10000000)

	def process_set(self, frames, output):
		# frames: (frame #, row, col, channels)
		if self.output_size is None:
			output_frames = frames
		else:
			output_frames = np.zeros(shape=(frames.shape[0], self.output_size[0], self.output_size[1], 3), dtype=np.float32)

		# Transform the frames + output pair with the random transforms. They all use the same seed.
		self.new_random_transform()

		# Transform each frame using the same seed. This is because we want to apply the exact same
		# transform to the whole set of input frames. This may be important for RNN batches or multiple frame inputs.
		processed_output = False
		#print(frames.shape[self.frame_axis])
		for i in range(output_frames.shape[self.frame_axis]):
			if not processed_output:
				(output_frames[i], output) = self._process_frame( frames[i], output )
				output_frames[i] = output_frames[i]
				processed_output = True
			else:
				(output_frames[i], junk) = self._process_frame( frames[i], output )
				output_frames[i] = output_frames[i]

		#pp.pprint(output_frames)
		return (output_frames, output)

	def _process_frame(self, frame, output):
		# x is a single image, so it doesn't have image number at index 0
		x = frame
		y = output
		
		
		img_row_axis = self.row_axis - 1
		img_col_axis = self.col_axis - 1
		img_channel_axis = self.channel_axis - 1

		# Set the current random seed
		prng = np.random.RandomState(self.seed_random)

		# use composition of homographies
		# to generate final transform that needs to be applied
		if self.rotation_range:
			theta = np.pi / 180 * prng.uniform(-self.rotation_range, self.rotation_range)
		else:
			theta = 0

		if self.height_shift_range:
			tx = prng.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
		else:
			tx = 0

		if self.width_shift_range:
			ty = prng.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
		else:
			ty = 0

		if self.shear_range:
			shear = prng.uniform(-self.shear_range, self.shear_range)
		else:
			shear = 0

		if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
			zx, zy = 1, 1
		else:
			zx, zy = prng.uniform(self.zoom_range[0], self.zoom_range[1], 2)

		transform_matrix = None
		if theta != 0:
			rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
										[np.sin(theta), np.cos(theta), 0],
										[0, 0, 1]])
			transform_matrix = rotation_matrix

		if tx != 0 or ty != 0:
			shift_matrix = np.array([[1, 0, tx],
									 [0, 1, ty],
									 [0, 0, 1]])
			transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

		if shear != 0:
			shear_matrix = np.array([[1, -np.sin(shear), 0],
									[0, np.cos(shear), 0],
									[0, 0, 1]])
			transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

		if zx != 1 or zy != 1:
			zoom_matrix = np.array([[zx, 0, 0],
									[0, zy, 0],
									[0, 0, 1]])
			transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

		if transform_matrix is not None:
			h, w = x.shape[img_row_axis], x.shape[img_col_axis]
			transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
			
			x = apply_transform(x, transform_matrix, img_channel_axis,
								fill_mode=self.fill_mode, cval=self.cval)

		if self.horizontal_flip:
			if prng.uniform() < 0.5:
				x = flip_axis(x, img_col_axis)

				# Modify the outputs for the flipped axis
				for index in self.horizontal_flip_invert_indices:
					y[index] = -y[index]

				for index in self.horizontal_flip_reverse_indices:
					y[index] = 1-y[index]

		if self.vertical_flip:
			if prng.uniform() < 0.5:
				x = flip_axis(x, img_row_axis)
		
		# Apply the random cropping
		if self.random_crop is not None:
			result = np.zeros(shape=(self.random_crop[0], self.random_crop[1], 3), dtype=np.float32)
			top_x = round(prng.uniform(0,x.shape[0]-self.random_crop[0]))
			top_y = round(prng.uniform(0,x.shape[1]-self.random_crop[1]))
			result = x[top_x:top_x+self.random_crop[0],top_y:top_y+self.random_crop[1],:]
			x = result
			
			
		# Apply the resize
		if self.output_size is not None:
			#result = np.zeros(shape=(self.output_size[0], self.output_size[1], 3), dtype=np.float32)
			#x = scipy.misc.imresize(x, (self.output_size[0], self.output_size[1]))
			x = cv2.resize(x, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_AREA)
		
		#pp.pprint(x)
		#cv2.imshow("norm",x[:,:,:])
		#cv2.waitKey(0)
		return (x, y)




class TrainingInput(object):
	def __init__(self, transformer, settings_file, position_rel_indexes, frame_rel_indexes, valdiation_rate, shuffle_frames=True, shuffle_chunks=True):
		self.base_path = os.path.dirname(settings_file)

		# Create the chunks
		f = open(settings_file,"r")
		self.chunks = []
		self.length = 0
		self.valdiation_rate = valdiation_rate
		self.width = None
		self.height = None
		self.depth = len(frame_rel_indexes)
		self.channels = 3
		self.output_size = None
		self.size_training = 0
		self.size_validation = 0
		self.transformer = transformer
		self.lock = threading.Lock()
		self.shuffle_frames = shuffle_frames
		self.shuffle_chunks = shuffle_chunks

		for row in f.readlines():
			tokens = row.replace("\n","").split("\t")
			num_columns = len(tokens[2:])
			
			min_range = None
			max_range = None
			if num_columns > 0:
				min_range = list(map(int, tokens[2:int(2+num_columns/2)]))
				max_range = list(map(int, tokens[int(2+num_columns/2):]))

			print("Creating training chunk from %s" % os.path.join(self.base_path, tokens[0]))
			chunk = Chunk(os.path.join(self.base_path, tokens[0]), os.path.join(self.base_path, tokens[1]), min_range, max_range, position_rel_indexes, frame_rel_indexes, valdiation_rate, shuffle_frames, transformer)
			self.length += chunk.num_frames
			self.width = chunk.width
			self.height = chunk.height
			self.output_size = chunk.output_size
			self.size_training += chunk.size_training
			self.size_validation += chunk.size_validation

			print("added %i new frames for a total of %i" % (chunk.num_frames, self.length))
			self.chunks.append(chunk)

		self.chunk_order = list(range(len(self.chunks)))
		self.reset()

	def reset(self):
		self.active_chunk = 0
		if self.shuffle_chunks:
			random.shuffle(self.chunk_order)
		
		for chunk in self.chunks:
			chunk.reset()

	def move_first_training_frame(self):
		self.active_chunk = 0

		self.reset()

	def move_first_validation_frame(self):
		self.active_chunk = 0

		self.reset()

	def get_next_training_frame(self):
		if self.active_chunk < len(self.chunks):
			# Get the next training frame from the active chunk
			(frames, output) = self.chunks[self.chunk_order[self.active_chunk]].get_next_training_frame()
			reset_memory = False
			
			if self.shuffle_chunks:
				self.active_chunk += 1
				if self.active_chunk >= len(self.chunks):
					self.active_chunk = 0
					random.shuffle(self.chunk_order)
			
			if frames is None:
				# Reset any model memory since we moved to a new chunk
				reset_memory = True
				self.reset() # reset when any chunk is out of frames
				
				(frames, output, junk) = self.get_next_training_frame()
			
			return (frames, output, reset_memory)
			
		
		return (None, None, None)

	def get_next_validation_frame(self):
		if self.active_chunk < len(self.chunks):
			# Get the next validation frame from the active chunk
			(frames, output) = self.chunks[self.chunk_order[self.active_chunk]].get_next_validation_frame()
			reset_memory = False
			
			if frames is None:
				# Reset any model memory since we moved to a new chunk
				reset_memory = True
				
				# Move to the next chunk
				self.active_chunk += 1
				if self.active_chunk < len(self.chunks):
					self.chunks[self.chunk_order[self.active_chunk]].reset()
					(frames, output, junk) = self.get_next_validation_frame()
			
			return (frames, output, reset_memory)

		return (None, None, None)
		
	def get_training_count(self):
		count = 0
		for chunk in self.chunks:
			count += chunk.get_training_count()
		return count
	
	def get_validation_count(self):
		count = 0
		for chunk in self.chunks:
			count += chunk.get_validation_count()
		return count

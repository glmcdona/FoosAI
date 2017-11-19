from chunk import *
from recording import *

import pprint
pp = pprint.PrettyPrinter(depth=6)

class Experiment(object):
	'''
	Simple class wraps an experiment
	'''

	def __init__(self, experiment_file):
		self.tree = ET.parse(experiment_file)
		self.root = self.tree.getroot()
		self.base_path = os.path.dirname(experiment_file)
		
		# Chunk settings
		self.chunk_min_frames = int(self.root.find("PROCESSING").find("CHUNK_MIN_FRAMES").text)
		self.chunk_rod_failure_limit = int(self.root.find("PROCESSING").find("CHUNK_ROD_FAILURE_LIMIT").text)
		
		# Results
		self.output_columns = self.root.find("OUTPUTS").text.split(",")
		self.output_folder = os.path.join(self.base_path, self.root.find("OUTPUT_FOLDER").text)
		self.output_width = int(self.root.find("PROCESSING").find("VIDEO").find("SIZE").find("W").text)
		self.output_height = int(self.root.find("PROCESSING").find("VIDEO").find("SIZE").find("H").text)
		
		# Recordings
		self.recordings = []
		for recording in self.root.find("RECORDINGS").iter("RECORDING"):
			recording_path = os.path.join(self.base_path, recording.text)
			self.recordings.append( Recording(recording_path) )
		
	def process(self):
		# Process the recordings into continuous valid chunks
		chunk_number = 0
		f_settings = open(os.path.join(self.output_folder, "settings.tsv"), "w")
		f_settings.close()
				
		for recording in self.recordings:
			# Process the chunks from this recording
			print("Processing recording %s into training chunks..." % recording.recording_file)
			chunk = None
			recording.initialize()
			frame_count = 0
			
			chunk_files = []
			chunk_positions_max = [None]*len(self.output_columns)
			chunk_positions_min = [None]*len(self.output_columns)
			
			while recording.has_more:
				(frame, frame_with_markup, rod_positions, failure_count) = recording.get_next_frame()
				
				# TODO: improve calculation to require minimum movement of rods in chunk!
				chunk_meets_movement = True
				movement_check = True
				
				if chunk is not None and not movement_check and not chunk_meets_movement:
					# Chunk is finished without results
					pass
				elif chunk is not None and not movement_check and chunk_meets_movement:
					# Chunk is finished with results  
					pass
				#elif chunk is not None and (failure_count >= self.chunk_rod_failure_limit or frame is None):
				elif chunk is not None and frame is None:
					# We failed to find the necessary number of rods. Chunk is finished.
					if chunk_meets_movement:
						print("%i failures at frame %i. Chunk finished." % (failure_count, frame_count))
						(video_file, position_file) = chunk.write()
						if video_file != None:
							chunk_files.append([video_file, position_file])
					chunk = None
				elif failure_count >= self.chunk_rod_failure_limit:
					# TODO: FIX THIS! skip the frame for now.
					print("Skipping tracking failure frame")
					frame = None
					pass
			
				elif chunk is None and failure_count < self.chunk_rod_failure_limit and frame is not None:
					# Start of a new chunk
					print("Started new chunk %i at frame %i." % (chunk_number, frame_count))
					chunk = Chunk(self.output_folder, chunk_number, (self.output_width, self.output_height),self.chunk_min_frames)
					chunk_number += 1
					
				if chunk is not None and frame is not None:
					# Log this chunk
					positions = []
					for idx, column in enumerate(self.output_columns):
						positions.append(str(rod_positions[column]))
						
						if float(frame_count) / float(recording.num_frames) > 0:
							if chunk_positions_max[idx] is None:
								# Use current positions as starting point for max/min value
								chunk_positions_min[idx] = rod_positions[column]
								chunk_positions_max[idx] = rod_positions[column]
							else:
								# Update the min and max
								chunk_positions_min[idx] = min(chunk_positions_min[idx], rod_positions[column])
								chunk_positions_max[idx] = max(chunk_positions_max[idx], rod_positions[column])
					
					chunk.add_frame(frame, positions)
				
				if (frame_count % 100) == 0 and chunk is not None:
					print("Processed %i of %i frames. Added %i frames to chunk so far. On chunk %i in %s." % (frame_count,recording.num_frames, chunk.get_count(), chunk_number-1,recording.recording_file))
					print("Current min bounds:")
					pp.pprint(chunk_positions_min)
					print("Current max bounds:")
					pp.pprint(chunk_positions_max)
				
				frame_count += 1
			
			# Write the final chunk
			if chunk is not None:
				(video_file, position_file) = chunk.write()
				if video_file is not None:
					chunk_files.append([video_file, position_file])
			
			# Write the list of chunks
			f_settings = open(os.path.join(self.output_folder, "settings.tsv"), "a")
			for chunk_file in chunk_files:
				f_settings.write("\t".join(chunk_file))
				for value in chunk_positions_min:
					f_settings.write("\t%i" % value)
				for value in chunk_positions_max:
					f_settings.write("\t%i" % value)
				f_settings.write("\n")
			f_settings.close()

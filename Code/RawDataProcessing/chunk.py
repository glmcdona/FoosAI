import cv2
import os

class Chunk():
	def __init__(self, result_folder, chunk_number, size, minimum_length):
		self.chunk_number = chunk_number
		self.result_folder = result_folder
		self.size = size
		self.count = 0
		self.minimum_length = minimum_length
		
		self.frames = []
		self.positions = []
	
	def write(self):
		if self.count > self.minimum_length:
			# Create the output folder
			if not os.path.exists(self.result_folder):
				os.makedir(self.result_folder)
			
			# Create the output chunk
			fourcc = cv2.VideoWriter_fourcc(*'DIVX')
			file_video = os.path.join(self.result_folder, "chunk%i.avi" % self.chunk_number)
			video = cv2.VideoWriter( file_video, fourcc, 30.0, self.size)
			for frame in self.frames:
				video.write(frame)
			video.release()
			
			file_output = os.path.join(self.result_folder, "chunk%i.tsv" % self.chunk_number)
			f_positions = open( file_output, "w")
			f_positions.write("\n".join(self.positions))
			f_positions.close()
			
			return ("chunk%i.avi" % self.chunk_number, "chunk%i.tsv" % self.chunk_number)
		
		return (None, None)
	
	def add_frame(self, frame, positions):
		# Add the frame
		resized_frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA) 
		self.frames.append( resized_frame )
		
		# Add the positions
		self.positions.append("\t".join(positions))
		
		self.count += 1
		
	def get_count(self):
		return self.count
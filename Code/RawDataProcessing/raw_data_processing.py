# Dependencies:
# !pip install numpy
# !pip install imageio
# !pip install matplotlib

import sys
from experiment import *

# Settings
#data_path  = ".\\..\\Recorder\\FeatureSetBuilder\\Experiments\\experiment4.config"

if len(sys.argv) != 3:
	print("ERROR: You need to pass in two arguments.")
	print("raw_data_processing.py play <path to recording.config of recording to view>")
	print("Eg: raw_data_processing.py play .\\..\\..\\TrainingData\\Raw\\Am1\\recording.config\n")
	print("raw_data_processing.py process <path to experiment.config>")
	print("Eg: raw_data_processing.py process .\\..\\..\\TrainingData\\Processed\\AmateurDefender\\experiment.config")
elif( sys.argv[1] == "play" ):
	# Play the video live
	print("Playing recording config frames from path %s." % (sys.argv[2]))
	rec = Recording(sys.argv[2])
	rec.play()
elif( sys.argv[1] == "playexp" ):
	# Play the video live
	print("Playing experiment config frames from path %s." % (sys.argv[2]))
	exp = Experiment(sys.argv[2])
	exp.play()
elif( sys.argv[1] == "process" ):
	print("Processing experimient config frames from path %s." % (sys.argv[2]))
	exp = Experiment(sys.argv[2])
	exp.process()
elif( sys.argv[1] == "play_rod" ):
	# Play the video live
	print("Playing recording config frames from path %s." % (sys.argv[2]))
	rec = Recording(sys.argv[2])
	rec.play()
else:
	print("ERROR: Invalid command %s. Must be play or process." % sys.argv[1])



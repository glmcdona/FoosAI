# FoosAI
This is a research project where we are automating a foosball table using robotics and deep learning.

We've got a single foosball rod actuator running:
[![Watch the foosball rod actuator](/Media/VideoCapture.png)](https://www.youtube.com/watch?v=sD1xugH3fjA "Watch the foosball rod actuator")

Here are a couple photos:
![We're talking about table soccer (Foosball)](/Media/Foosball.png)
![Automating it using deep learning](/Media/System.png)
![The robot](/Media/FoosAI.png)
![Rod coupling](/Media/RodCoupling.png)


## Overview

### AI training setup overview
* Webcam and profession tour foosball videos are used as training inputs.
* Video is processed in python to extract foosball table rod positions from the video to use to prepare training outputs.
* A deep-learning model is trained that inputs camera frames, and outputs predicted changes to the rod positions in the next fraction of a second.
* Transfer learning is used, to first each it to extract rod positions from the camera frames, then this model is transferred to predict the change in rod positions in the next X seconds.
* After learning from human players, future plans include running reinforcement learning afterwards with the robot playing against itself to better adapt to the motor responses, and allow for creative new foosball strategies.


### AI implementation overview
* A webcam is hooked up to a laptop/desktop that has a graphics card.
* The laptop/computer runs the AI model in real-time, outputting the prediction of the future rod positions to the Arduino via USB.
* The Arduino controls the stepper motors to do it's best to carry out the desired rod position change.

## Project stages

### Stage 1 (In progress)
Stage 1 of the project is just simple automation of the defense player rods:
* CNN deep neural network trained to control only rod translation (not rod rotations) to defend against a simple attacker.
* Processes only current camera frame to make the prediction (not previous frames).
* No recurrent or memory layers (RNN).

We are hoping to observe it controlling the defense rods to keep them between the ball and the net.

### Stage 2
After proving Stage 1, we are hoping to build out robotic actuators to automate a whole side of a foosball table.
* Include rod rotations in training.
* Explore RNN models.
* Explore multiple camera frame inputs. Still train off human data.

### Stage 3
* Automate both sides of a foosball table.
* Extract ball position, and build rewards and penalties for ball advancement, ball loss, scoring, being scored against.
* Add lifts to one side of the table to unstuck the ball automatically.
* Reinforcement learning where it plays against itself for hours (or days?).

## Installation and starting the training notebook
Most contributors are likely to work with the processed training data to try building their own ML models. This section is a quick start on the tools to install to get the example training code working.

Install python 3.x using Ananconda, preferably the 64 bit version for a larger memory working set:
* https://www.anaconda.com/download/

Install a deep-learning framework. If you would like the example to work, you can install Keras. I use it currently with Tensorflow on Windows 64 bit myself:
* Install Keras and Tensorflow. You can follow the instructions here https://keras.io/#installation.

There are some prerequisites for the python code to work. In the anaconda prompt, you can install the prerequisites with the following commands:
* pip install keras
* pip install numpy
* pip install imageio
* pip install matplotlib
* pip install opencv-python

The following jupyter notebook has the training code to fool around with:
* https://github.com/glmcdona/FoosAI/blob/master/Code/Training/TrainingFoosbot.ipynb

To open up this notebook, from the anaconda environment run "jupyter notebook TrainingFoosbot.ipynb".

There is a big python file which does most of the heavy-lifting that this jupyter notebook uses in the same folder:
* https://github.com/glmcdona/FoosAI/blob/master/Code/Training/video_file.py

There is a lot of code in here around not having to load all the training data into memory at the same time. It only keeps one chunk in memory at a time. It also has a lot of code around applying random transformations to the camera frames (rotations, skews, zooms, horizontal flipping, etc) to make the resulting models more robust.

If you train a fun model, I can run it on the physical FoosAI robot to check out how it works :) Generally when implemented on the FoosAI robot, it needs a change in position to apply to the rods as the control signal.

## Only loading the training data
The training data has two parts:
1. Video camera frames as input.
1. Corresponding array of foosball table rod positions.

The processed training data is downsampled to a resolution of 100x55 and has the RGB colour channels. A training set is composed of multiple video chunks and a tsv file that has the rod positions. In the case of the first training set provided:
* https://github.com/glmcdona/FoosAI/tree/master/TrainingData/Processed/AmateurDefender/Result

It has two video chunks. Each row in chunk0.tsv corresponds to each frame in chunk0.avi and has the rod positions. The units here are in pixels before the frame downselection to 100x55 occured. In this case the column header would be "Rod1 position, Rod2 position, Rod3 position" - these are defined in https://github.com/glmcdona/FoosAI/blob/master/TrainingData/Processed/AmateurDefender/experiment.config.

The settings.tsv has the list of chunk files to include for the complete training data, and also the normalizations to convert the rod positions to a floating point scale in the range 0.0 to 1.0. The first three numbers are the minimum displacements of each rod, and the last three numbers are the maximum displacement of each rod to be used in each chunk.

When preparing the training data, you will want to calculate the difference in position in the next X frames as well as a training output, since this is actually what the robot can implement.

I wrote code to load the video chunks and corresponding positions as training data, as well as modified the Keras image transformer to work with the video chunks to apply random transforms such as translations, rotations, skews, and zooms. You may want to use it here:
* https://github.com/glmcdona/FoosAI/blob/master/Code/Training/video_file.py

To load the training data using this class, you can instantiate a class like in the code here:
* https://github.com/glmcdona/FoosAI/blob/master/Code/Training/TrainingFoosbot.ipynb

First you create a transformer which describes how the random alterations of the video should be performed. In our case training output indices are set up as:
* 0: Rod 1 position (goalie) from 0.0 to 1.0. 0.0 is the farthest left, 1.0 is the farthest right.
* 1: Rod 2 position (defencemen) from 0.0 to 1.0. 0.0 is the farthest left, 1.0 is the farthest right.
* 2: Rod 3 position (attackers) from 0.0 to 1.0. 0.0 is the farthest left, 1.0 is the farthest right.
* 3: Delta rod 1 position in 2 camera frames from now. Range is -1.0 to 1.0.
* 4: Delta rod 2 position in 2 camera frames from now. Range is -1.0 to 1.0.
* 5: Delta rod 3 position in 2 camera frames from now. Range is -1.0 to 1.0.

```python
from video_file import *
transformer = VideoTransform( zoom_range=0.1, # +-10% zoom
                              rotation_range=20, # +-20 degrees rotation
                              width_shift_range=0.1, # Shift +-10 percent of width
                              height_shift_range=0.1, # Shift +-10 percent of height
                              shear_range= 0.1,  # Apply +-10 percent shearing
                              fill_mode='nearest',
                              vertical_flip=False,
                              horizontal_flip=True,
                              horizontal_flip_invert_indices = [3,4,5], # Invert the rod position deltas on horizontally flipped frames
                              horizontal_flip_reverse_indices = [0,1,2], # Reverse the rod positions on horizontally flipped frames
                              data_format='channels_last' )
```

Among all the usual transforms, it also tells it that it can flip the video frames horizontally. When it flips it horizontally, it instructs it to invert the delta rod displacements, and to reverse the scale from 0.0 to 1.0 instead mapping to 1.0 to 0.0 for the rod positions. This effectively allows us to almost double our training data from our video data by flipping the frames horizontally. 

Then to load the input video with this transformer, we do:
```python
# Paths relative to current python file.
data_path  = ".\\..\\..\\TrainingData\\Processed\\AmateurDefender\\Result\\settings.tsv"

print("Opening training frames from config %s." % (data_path))
position_rel_indexes = [0, 2] # Predict current rod positions and future position in 2 frames
frame_rel_indexes = [0] # Use only current video frame as input
  training = TrainingInput(transformer, data_path, position_rel_indexes, frame_rel_indexes, 0.2) # 20% validation holdout
  training.clear_memory()
```

Here it only inputs the current camera frame as the model input. If you want the current camera frame and the previous two, you would set:
```python
position_rel_indexes = [0, -1, -2]
```

To get the first training input/output pair you can run:
```python
import numpy as np

training.move_first_training_frame()
(frame, position) = training.get_next_training_frame()

print("Shape of training input:")
print(np.shape(frame))

print("Shape of training output:")
print(np.shape(position))
```

This should output something like:
```
Shape of training input:
(1, 54, 100, 3)
Shape of training output:
(6,)
```

The input format is (frames, x, y, colour channels). In this case we only input the current camera frame to the model, so the size of the first axis is 1.

The corresponding output format is (rod 1 pos, rod 2 pos, rod 3 pos, rod 1 delta pos compared to 2 frames in the future, rod 2 delta pos, rod 3 delta pos).

To load minibatches for training for my usage in Keras, I train it like the following:
```
image_height       = training.height
image_width        = training.width
image_depth        = training.depth

def TrainGen():
    while True:
        #print("TrainGen restarting training input.")
        training.move_first_training_frame()
        (frames, output) = training.get_next_training_frame()
        while frames is not None:
            yield (frames, output)
            (frames, output) = training.get_next_training_frame()
            
def ValidateGen():
    while True:
        #print("Validation restarting training input.")
        training.move_first_validation_frame()
        (frames, output) = training.get_next_validation_frame()
        while frames is not None:
            yield (frames, output)
            (frames, output) = training.get_next_validation_frame()

def TrainBatchGen(batch_size):
    gen = TrainGen()
    while True:
        # Build the next batch
        batch_frames = np.zeros(shape=(batch_size, image_depth, image_height, image_width, image_channels), dtype=np.float32)
        batch_outputs = np.zeros(shape=(batch_size, 3), dtype=np.float32)
        for i in range(batch_size):
            (frames, output) = next(gen)
            batch_frames[i,:,:,:,:] = frames
            #batch_outputs[i,:] = output[3:6] # NOTE: This is a hackjob to get the model to ONLY predict the change in rod positions!
            #batch_outputs[i,:] = output[0:3] # NOTE: Similarly, only predict current rod positions.
            batch_outputs[i,:] = output # NOTE: Predict both rod positions and rod deltas
            
        #pp.pprint(batch_outputs)
        yield (batch_frames, batch_outputs)
        #pp.pprint("Yielded batch")
        
def ValidateBatchGen(batch_size):
    gen = ValidateGen()
    while True:
        # Build the next batch
        batch_frames = np.zeros(shape=(batch_size, image_depth, image_height, image_width, image_channels), dtype=np.float32)
        batch_outputs = np.zeros(shape=(batch_size, 3), dtype=np.float32)
        for i in range(batch_size):
            (frames, output) = next(gen)
            batch_frames[i,:,:,:,:] = frames
            #batch_outputs[i,:] = output[3:6]
            #batch_outputs[i,:] = output[0:3]
            batch_outputs[i,:] = output
        
        #pp.pprint("Yielding batch")
        #pp.pprint(batch_outputs)
        yield (batch_frames, batch_outputs)
        #pp.pprint("Yielded batch")

WEIGHTS_FNAME = 'mnist_cnn_weights_%i.hdf'
MODELS_FNAME = 'mnist_cnn_models_%i.h5'
for epoch in range(10000):
    print("------ Epoch %i -------" % epoch)
    model.fit_generator(TrainBatchGen(20), 1552, epochs=1, verbose=1, callbacks=None, class_weight=None, max_q_size=10, workers=1, validation_data=ValidateBatchGen(20), validation_steps = 500, pickle_safe=False, initial_epoch=0)
    model.save_weights(WEIGHTS_FNAME % epoch)
    model.save(MODELS_FNAME % epoch)
```









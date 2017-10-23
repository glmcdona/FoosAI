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

## Installation
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


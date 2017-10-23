# FoosAI
This is a research project where we are automating a foosball table using robotics and deep learning.
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

### Stage 1
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




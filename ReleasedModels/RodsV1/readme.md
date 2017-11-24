# FoosAI rod position and angle models V1

These are deep-learning models trained to extract a rod position and angle given a single 640x180 image of the rod. This extracted rod position and angle is used to prepare full-table all-rod training data to teach FoosAI.

Position is in the range 0.0 for furthest left and 1.0 for furthest right.

Angle is in the range -1.0 to 1.0. Value 0.0 is straight down, -0.5 is halfway towards to camera, +0.5 is halfway awa from the camera.
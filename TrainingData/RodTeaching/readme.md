# Training data for extracting rod positions and angles
Foosball clips are split into small chunks where each end of the video clip are labelled (eg. starting from an angle of -1 and ending at an angle of 1), and all the frames inbetween are assumed to be interpolated between those two values.

Clips are taken keeping angle constant and moving the each rod up and down the table. Clips are also taken while the rod is rotating at a human controlled constantish rate. The camera when looking a spinning rod observers blurring of the rod some, so it is important that some of the training data is from actively rotating rods.

Position data is processed similarly. 0.0 trained as farthest left, and 1.0 trained as farthest right. Clips are taken that are moving approximately constantly from one side to the other, and the position is estimated accordingly.

A rod is taken as a 640x180 volume of pixels. The goal is to train deep-learning models output the estimated rod angle (-1 to 1) and estimated rod position (0 to 1). Then, when processing a foosball game, a box is drawn around each individual rod. These trained models output all rod positions + angles. This allows the deep-learning models to be trained to output all predicted rod position changes and rod angle changes to be run by the robot.
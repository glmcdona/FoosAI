# Defence V3 notes
In this model, the bar it is controlling is hidden from the model's input view. It is tasked with guessing where that hidden rod position should be at a time 3 camera frames in the future. The input to this model takes in the previous three camera frames.

The rod positions as input for the training are build using the RodV1 CNN model that extracts a rod position from a single rod image cross-section. To run this model in realtime, this RodV1 model is used to calculate the 2-bar's current position. The difference between the current rod position to the model's guess at the rod position is used to control the stepper motor.

Notes:
* 3D CNN takes previous 3 camera frames (3x100x55x3) as input.
* The 2-bar is blacked-out so that it can't see where the rod currently is.
* Outputs the guess at where the 2-bar should be positioned in 3 camera frames in the future.
* For the robot, it moves the motor to implement the difference from the current 2-bar position to the model's outputted guess.
* Current 2-bar position is also loaded using the RodV1 CNN model.
* Trained for around 24 hours in total.
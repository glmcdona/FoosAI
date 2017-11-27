# Defence V1 notes
Transfer learning used. First it was trained to learn the current position of the three rods. Afterwards, the model was kept and trained instead to predict the difference in position in a half second.

Notes:
* 3D CNN takes previous 3 camera frames (3x100x55x3) as input.
* A much simpler CNN model structure was used.
* Trained for about 24 hours in total.
* Trained only from one view perspective, mounted on the light rack looking down.
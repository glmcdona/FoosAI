# Defence V1 notes
Transfer learning used. First it was trained to learn the current position of the three rods (config12_model_posonly.hdf). Afterwards, the model was kept and trained instead to predict the difference in position in a half second.

Notes:
* Spent ~12 hours teaching it to track rod positions
* Spent 5 days teaching it to predict the future rod position until it finally started overfitting
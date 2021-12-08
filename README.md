# ml-project

Important scripts: run main.py to train the model, the python notebook contains the data exploration and the vizualizations, evaluation.py calculates BLEU-score.

dataset.py: Handles loading the pre-encoded images (as output by feature_extraction.py) and sets up a dataset class to use when training the models.

decoder.py: Contains the different classes for all the decoder architecture we tried.

download_dataset.py: Downloads the MS COCO dataset and sets up the directory structure for the project.

encoder.py: Contains the different classes for the encoder we tried. These classes take the output of the pretrained CNN as input.

evaluation.py: Calculates the BLEU scores for our models.

feature_extraction.py: Runs the images through the pre-trained CNNs and saves the encoded images to disk for faster training.

main.py: Contains hyperparameter for the model and is the main script to run to train the model.

ml-project.ipynb: Notebook that contains all the data exploration and vizualizations for the project.

model.py: Contains the functions to train the model and run inference including with/without attention, with/without curriculum learning.

preprocessing.py: Loads the captions, preprocesses them and creates the vocabulary. Also used to rebuild sentences from the output of the model.
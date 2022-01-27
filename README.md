# ml-project

This project was completed in the context of a PHD machine learning class at Carnegie Mellon University by Michael Agaby, olivier Filion, and Nicholas Amano. We implement a CNN-LSTM architecture with a visual attention mechanism which achieves a 23.9 BLEU-4 score on the MS COCO dataset using curriculum learning. We interpret the CNN encoder through visualizations and show that the model is able to automatically learn to look at the most important elements of the image when generating captions.

## Running the project


Important scripts: run main.py to train the model, the python notebook contains the data exploration and the vizualizations, evaluation.py calculates BLEU-score.

----------------

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

-----------------

The code as submitted is set up to run the best model (attention with curriculum learning). To remove the curriculum learning, set DO_CURRICULUM to False in model.py.

To run the baseline model,
    in main.py: uncomment lines 39-41 and comment out lines 44-46 in main.py. Change "mp" to "fc" in lines 35 and 26. Also, change the LEARNING_RATE parameter to 1e-3.
    in model.py: uncomment lines 87 and 121 in model.py and comment out lines 88 and 122.

Note that the main.py script needs the output of feature_extraction.py to be saved on the disk with the correct directory structure.

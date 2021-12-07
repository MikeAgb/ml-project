
import os
from numpy.lib.twodim_base import tril_indices

import torch
import numpy as np

import encoder
import decoder
import model
import preprocessing
import dataset

MAX_CAPTION_LENGTH = 18
EMBEDDING_DIM = 256
LEARNING_RATE = 1e-4
PERC_DATASET = 0.3
BATCH_SIZE = 64
EPOCHS = 20
MIN_FREQ = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NAME = "attention"

if __name__ == "__main__":
    # Get captions
    train_captions = preprocessing.load_captions(os.path.join("dataset", "annotations", "annotations", "captions_train2017.json"))
    val_captions = preprocessing.load_captions(os.path.join("dataset", "annotations", "annotations", "captions_val2017.json"))
    train_pre_captions = preprocessing.preprocess_captions(train_captions, MAX_CAPTION_LENGTH)
    val_pre_captions = preprocessing.preprocess_captions(val_captions, MAX_CAPTION_LENGTH)
    vocab = preprocessing.create_vocabulary(train_pre_captions, min_freq=MIN_FREQ)

    # Create datasets
    train_ds = dataset.EncodedDataset(vocab, train_pre_captions, MAX_CAPTION_LENGTH, "train", "mp", google=False)
    val_ds = dataset.EncodedDataset(vocab, val_pre_captions, MAX_CAPTION_LENGTH, "val", "mp", google=False)

    # Create model
    # encoder_model = encoder.LinearDimensionalityReduction(1024, EMBEDDING_DIM)
    # decoder_model = decoder.BasicDecoder(EMBEDDING_DIM, EMBEDDING_DIM, vocab, MAX_CAPTION_LENGTH)
    # captionning_model = model.CaptionningModel(encoder_model, decoder_model)

    encoder_model = encoder.EncodeForAttention(512, 256)
    decoder_model = decoder.AttentionDecoder(256, 256, 512, vocab, MAX_CAPTION_LENGTH)
    captionning_model = model.CaptionningModel(encoder_model, decoder_model).to(model.DEVICE)

    print("Number of parameters:", format(np.sum([np.prod(p.size()) for p in captionning_model.parameters()]), ","))

    train_loss = model.train(train_ds, captionning_model, num_epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, epoch_perc=PERC_DATASET)
    np.savetxt("train_loss.txt", train_loss.numpy())

    torch.save(encoder_model, os.path.join("models", NAME + "_encoder.pt"))
    torch.save(decoder_model, os.path.join("models", NAME + "_decoder.pt"))
    torch.save(captionning_model, os.path.join("models", NAME + "_model.pt"))

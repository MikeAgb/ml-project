
import os

import torch
import numpy as np

import encoder
import decoder
import model
import preprocessing
import dataset

MAX_CAPTION_LENGTH = 16
CAPTIONS_PER_IMAGE = 5
EMBEDDING_DIM = 256
LEARNING_RATE = 1e-3
PERC_DATASET = 0.1

if __name__ == "__main__":
    # Get captions
    train_captions = preprocessing.load_captions(os.path.join("dataset", "annotations", "annotations", "captions_train2017.json"))
    val_captions = preprocessing.load_captions(os.path.join("dataset", "annotations", "annotations", "captions_val2017.json"))
    train_pre_captions = preprocessing.preprocess_captions(train_captions, MAX_CAPTION_LENGTH)
    val_pre_captions = preprocessing.preprocess_captions(val_captions, MAX_CAPTION_LENGTH)
    vocab = preprocessing.create_vocabulary(train_pre_captions, min_freq=20)

    # Create datasets
    train_ds = dataset.EncodedDataset(vocab, train_pre_captions, MAX_CAPTION_LENGTH, "train", "fc", google=True, captions_per_image=CAPTIONS_PER_IMAGE)
    val_ds = dataset.EncodedDataset(vocab, val_pre_captions, MAX_CAPTION_LENGTH, "val", "fc", google=True, captions_per_image=CAPTIONS_PER_IMAGE)

    # Create model
    encoder_model = encoder.LinearDimensionalityReduction(1024, EMBEDDING_DIM)
    # encoder_model = encoder.EncodeFromCNNLayer(EMBEDDING_DIM)
    decoder_model = decoder.BasicDecoder(EMBEDDING_DIM, EMBEDDING_DIM, vocab, MAX_CAPTION_LENGTH)
    captionning_model = model.CaptionningModel(encoder_model, decoder_model)

    print("Number of parameters:", format(np.sum([np.prod(p.size()) for p in captionning_model.parameters()]), ","))

    model.train(train_ds, val_ds, captionning_model, num_epochs=20, batch_size=32, lr=LEARNING_RATE, captions_per_image=CAPTIONS_PER_IMAGE,
                epoch_perc=PERC_DATASET)

    torch.save(encoder_model, "baseline_encoder.pt")
    torch.save(decoder_model, "baseline_decoder.pt")
    torch.save(captionning_model, "baseline_model.pt")

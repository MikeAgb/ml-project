
import os

import torch
import numpy as np

import encoder
import decoder
import model
import preprocessing
import dataset

MAX_CAPTION_LENGTH = 16
CAPTIONS_PER_IMAGE = 1
EMBEDDING_DIM = 256
LEARNING_RATE = 1e-4
PERC_DATASET = 0.05

if __name__ == "__main__":
    # Get captions
    train_captions = preprocessing.load_captions(os.path.join("dataset", "annotations", "annotations", "captions_train2017.json"))
    val_captions = preprocessing.load_captions(os.path.join("dataset", "annotations", "annotations", "captions_val2017.json"))
    train_pre_captions = preprocessing.preprocess_captions(train_captions, MAX_CAPTION_LENGTH)
    val_pre_captions = preprocessing.preprocess_captions(val_captions, MAX_CAPTION_LENGTH)
    vocab = preprocessing.create_vocabulary(train_pre_captions, min_freq=20)

    # Create datasets
    train_ds = dataset.EncodedDataset(vocab, train_pre_captions, MAX_CAPTION_LENGTH, "train", "mp", google=False)
    val_ds = dataset.EncodedDataset(vocab, val_pre_captions, MAX_CAPTION_LENGTH, "val", "mp", google=False)

    # Create model
    # encoder_model = encoder.LinearDimensionalityReduction(1024, EMBEDDING_DIM)
    # # encoder_model = encoder.EncodeFromCNNLayer(EMBEDDING_DIM)
    # decoder_model = decoder.BasicDecoder(EMBEDDING_DIM, EMBEDDING_DIM, vocab, MAX_CAPTION_LENGTH)
    # captionning_model = model.CaptionningModel(encoder_model, decoder_model)

    encoder_model = encoder.EncodeForAttention(512, 256)
    decoder_model = decoder.AttentionDecoder(256, 256, 512, vocab, MAX_CAPTION_LENGTH)
    captionning_model = model.CaptionningModel(encoder_model, decoder_model)
    

    print("Number of parameters:", format(np.sum([np.prod(p.size()) for p in captionning_model.parameters()]), ","))

    model.train(train_ds, val_ds, captionning_model, num_epochs=10, batch_size=128, lr=LEARNING_RATE, epoch_perc=PERC_DATASET)

    torch.save(encoder_model, "attention_encoder.pt")
    torch.save(decoder_model, "attention_decoder.pt")
    torch.save(captionning_model, "attention_model.pt")


import os

import torch

import encoder
import decoder
import model
import preprocessing
import dataset

MAX_CAPTION_LENGTH = 22
CAPTIONS_PER_IMAGE = 1

if __name__ == "__main__":
    # Get captions
    train_captions = preprocessing.load_captions(os.path.join("dataset", "annotations", "annotations", "captions_train2017.json"))
    val_captions = preprocessing.load_captions(os.path.join("dataset", "annotations", "annotations", "captions_val2017.json"))
    train_pre_captions = preprocessing.preprocess_captions(train_captions, MAX_CAPTION_LENGTH)
    val_pre_captions = preprocessing.preprocess_captions(val_captions, MAX_CAPTION_LENGTH)
    vocab = preprocessing.create_vocabulary(train_pre_captions)

    # Create datasets
    train_ds = dataset.EncodedDataset(vocab, train_pre_captions, MAX_CAPTION_LENGTH, "train", "fc", CAPTIONS_PER_IMAGE)
    val_ds = dataset.EncodedDataset(vocab, val_pre_captions, MAX_CAPTION_LENGTH, "val", "fc", CAPTIONS_PER_IMAGE)

    # Create model
    encoder_model = encoder.LinearDimensionalityReduction(4096, 512)
    decoder_model = decoder.BasicDecoder(512, 512, vocab, MAX_CAPTION_LENGTH)
    captionning_model = model.CaptionningModel(encoder_model, decoder_model)

    model.train(train_ds, val_ds, captionning_model, num_epochs=10, batch_size=64, lr=0.1, captions_per_image=CAPTIONS_PER_IMAGE)

    torch.save(encoder_model, "baseline_encoder.pt")
    torch.save(decoder_model, "baseline_decoder.pt")
    torch.save(captionning_model, "baseline_model.pt")

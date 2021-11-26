import os

import torch

import preprocessing
import dataset

if __name__ == "__main__":
    caption_model = torch.load("attention_model.pt")
    vocab = caption_model.decoder.vocab

    val_captions = preprocessing.load_captions(os.path.join("dataset", "annotations", "annotations", "captions_train2017.json"))
    val_pre_captions = preprocessing.preprocess_captions(val_captions, 22)

    val_ds = dataset.EncodedDataset(vocab, val_pre_captions, 22, train_val_test="train", fc_mp="mp", google=False)

    idx = 0
    im, l = val_ds[idx]
    print(val_ds.indices[idx][0])
    pred = caption_model(im.unsqueeze(0))
    words = torch.argmax(pred, dim=-1)

    print(preprocessing.rebuild_sentence(words[0], vocab))
    

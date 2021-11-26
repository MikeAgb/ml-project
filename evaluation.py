import os

import torch

import preprocessing
import dataset
import model

if __name__ == "__main__":
    caption_model = torch.load(os.path.join("models", "teacher_baseline_model.pt"))
    vocab = caption_model.decoder.vocab

    val_captions = preprocessing.load_captions(os.path.join("dataset", "annotations", "annotations", "captions_val2017.json"))
    val_pre_captions = preprocessing.preprocess_captions(val_captions, caption_model.decoder.caption_length)

    val_ds = dataset.EncodedDataset(vocab, val_pre_captions, caption_model.decoder.caption_length, train_val_test="val", fc_mp="fc", google=True)

    idx = 0
    im, l = val_ds[idx]
    print(val_ds.indices[idx][0])
    pred = model.inference(caption_model, im.unsqueeze(0))

    print("Label:", preprocessing.rebuild_sentence(l, vocab))
    print("Pred:", preprocessing.rebuild_sentence(pred[0], vocab))

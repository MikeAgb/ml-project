import os

import torch

import preprocessing
import dataset
import model

IDX = 6
TRAIN_VAL = "val"
FC_MP = "fc"
NAME = "teacher_baseline"

if __name__ == "__main__":
    caption_model = torch.load(os.path.join("models", NAME + "_model.pt"))
    vocab = caption_model.decoder.vocab

    val_captions = preprocessing.load_captions(os.path.join("dataset", "annotations", "annotations", "captions_val2017.json"))

    image_id = list(val_captions.keys())[IDX]
    image_str = str(image_id)
    image_filename = "0" * (12 - len(image_str)) + image_str + ".pt"
    im = torch.load(os.path.join("dataset", "features_google", TRAIN_VAL, FC_MP, image_filename))

    pred = model.inference(caption_model, im)

    print("Image ID:", image_id)
    print("Labels:")
    for caption in val_captions[image_id]:
        print("\t", caption)
    print("Model output:", preprocessing.rebuild_sentence(pred[0], vocab))

import os

import torch

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json

import preprocessing
import dataset
import model

IDX = 6
TRAIN_VAL = "val"
FC_MP = "fc"
NAME = "teacher_baseline"
ANNFILE = os.path.join("dataset", "annotations", "annotations", "captions_val2017.json")
RESULTS_PATH = os.path.join('best_model','results')
RESULTS_FILENAME = 'results_val.json'
METRICS = True


def evaluateModel(model_json, coco):
    # Evaluates the Model using the prebuild coco eval scripts
    # from https://github.com/mosessoh/CNN-LSTM-Caption-Generator/blob/173f7fa23f1a827a7f6894504a00a2af19e36724/evaluate_captions.py
    cocoRes = coco.loadRes(model_json)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()  
    cocoEval.evaluate()
    results = {}
    for metric, score in cocoEval.eval.items():
        results[metric] = score
    return results


def caption_dataset(model, val_captions, results_dir):
    # Generates a json that is a list of dictionaries containing the 
    # image_id and the resulting caption from the model
    caption_model = model
    vocab = caption_model.decoder.vocab

    
    captions = []
    temp_cap = {}
    image_ids = list(val_captions.keys())

    for image_id in image_ids:
        temp_cap["image_id"] = image_id
        image_str = str(image_id)
        image_filename = "0" * (12 - len(image_str)) + image_str + ".pt"
        im = torch.load(os.path.join("dataset", "features_google", TRAIN_VAL, FC_MP, image_filename))

        pred = model.inference(caption_model, im)
        temp_cap["caption"] = preprocessing.rebuild_sentence(pred[0], vocab)
        captions.append(temp_cap)
    json.dump(captions, open(results_dir, 'wb'))

if __name__ == "__main__":
    caption_model = torch.load(os.path.join("models", NAME + "_model.pt"))
    vocab = caption_model.decoder.vocab

    val_captions = preprocessing.load_captions(ANNFILE)

    if METRICS:
        results_dir = os.path.join(RESULTS_PATH, RESULTS_FILENAME)
        coco = COCO(ANNFILE)
        # Checks if we already have already generated the captions json
        if not os.path.exists(results_dir):
            caption_dataset(caption_model, val_captions, results_dir)
        results = evaluateModel(results_dir,coco)
        for metric, score in results:
            print(f'{metric} metric scored {score}')
    else:
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

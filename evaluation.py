import os

import torch

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
import csv

import preprocessing
import model
from tqdm import tqdm

IDX = 156
TRAIN_VAL = "val"
MODEL_PATH = os.path.join("models","curriculum" )
MODEL_NAME = "check_att20.pt"
ATTENTION = True
FC_MP = "fc"
if ATTENTION:
    FC_MP = "mp"

ALL_CHECKPOINTS = True
SCORE_FILE = 'scores.csv'

METRICS = True
FEATURES_PATH = os.path.join("dataset", "features")
ANNFILE = os.path.join("dataset", "annotations", "annotations", "captions_val2017.json")
RESULTS_PATH = os.path.join('models','captions', 'curriculum')
RESULTS_FILENAME = 'new_curr_results.json'


def evaluateModel(val_dict, res_dict):
    # Evaluates the Model using the prebuild coco eval scripts
    # from https://github.com/mosessoh/CNN-LSTM-Caption-Generator/blob/173f7fa23f1a827a7f6894504a00a2af19e36724/evaluate_captions.py
    cocoEval = COCOEvalCap(val_dict, res_dict)
    cocoEval.params['image_id'] = val_dict.keys()
    cocoEval.evaluate()
    results = {}
    for metric, score in cocoEval.eval.items():
        results[metric] = score
    return results


def caption_dataset(cap_model, val_captions, results_dir, temp_file = True):
    # Generates a json that is a list of dictionaries containing the 
    # image_id and the resulting caption from the model
    print("Generating Captions From model")
    caption_model = cap_model
    vocab = caption_model.decoder.vocab

    
    captions = {}
    temp_cap = {}
    image_ids = list(val_captions.keys())

    for image_id in tqdm(image_ids):
        temp_cap = {}
        temp_cap["image_id"] = image_id
        image_str = str(image_id)
        image_filename = "0" * (12 - len(image_str)) + image_str + ".pt"
        im = torch.load(os.path.join(FEATURES_PATH, TRAIN_VAL, FC_MP, image_filename))

        if ATTENTION:
            pred = model.inference_attention(caption_model, im)
        else:
            pred = model.inference(caption_model, im)
        captions[image_id] = preprocessing.rebuild_sentence(pred[0], vocab)
    if temp_file:
        json.dump(captions, open(results_dir, 'w'))
    else:
        return captions

if __name__ == "__main__":
    if ALL_CHECKPOINTS:
        scores = {}
        for n in range(1,21):
            temp_score = {}
            caption_model = torch.load(os.path.join(MODEL_PATH, f"check_att{n}.pt"))
            vocab = caption_model.decoder.vocab
            val_captions = preprocessing.load_captions(ANNFILE)

            results_dir = os.path.join(RESULTS_PATH, f"check_results{n}.pt")
        
            # Checks if we already have already generated the captions json
            if not os.path.exists(results_dir):
                caption_dataset(caption_model, val_captions, results_dir)
            with open(results_dir) as file:
                results_dict = (json.load(file))
            results = evaluateModel(val_captions, results_dict)
            scores[n] = results
        with open(os.path.join(RESULTS_PATH, SCORE_FILE), 'w') as file:
            w = csv.writer(file)
            w.writerow(['id','blue1', 'blue2', 'blue3', 'blue4', 'CIDEr'])
            for key, val in scores.items():
                # write every key and value to file
                v = list(val.values())
                v.insert(0,key)
                w.writerow(v)

    else:
        caption_model = torch.load(os.path.join(MODEL_PATH, MODEL_NAME))
        vocab = caption_model.decoder.vocab

        val_captions = preprocessing.load_captions(ANNFILE)

        if METRICS:
            results_dir = os.path.join(RESULTS_PATH, RESULTS_FILENAME)
            
            # Checks if we already have already generated the captions json
            if not os.path.exists(results_dir):
                caption_dataset(caption_model, val_captions, results_dir)
            with open(results_dir) as file:
                results_dict = (json.load(file))
            results = evaluateModel(val_captions, results_dict)
            for metric, score in results:
                print(f'{metric} metric scored {score}')
        else:
            image_id = list(val_captions.keys())[IDX]
            image_str = str(image_id)
            image_filename = "0" * (12 - len(image_str)) + image_str + ".pt"
            im = torch.load(os.path.join(FEATURES_PATH, TRAIN_VAL, FC_MP, image_filename))

            if ATTENTION:
                pred = model.inference_attention(caption_model, im)
            else:
                pred = model.inference(caption_model, im)
            print("Image ID:", image_id)
            print("Labels:")
            for caption in val_captions[image_id]:
                print("\t", caption)
            print("Model output:", preprocessing.rebuild_sentence(pred[0], vocab))

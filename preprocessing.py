
import json

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

PUNCTUATION = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
PUNCTUATION_TRANS = str.maketrans("", "", PUNCTUATION)

def load_captions(path):
    with open(path, "r") as json_file:
        json_dict = json.load(json_file)
    captions = {}
    for annotation in json_dict["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in captions:
            captions[image_id] = []
        captions[image_id].append(annotation["caption"])
    return captions

def preprocess_captions(captions, max_length=22):
    tokenizer = get_tokenizer("basic_english")
    preprocessed_captions = {}
    for image_id in captions:
        preprocessed_captions[image_id] = []
        for caption in captions[image_id]:
            caption = caption.translate(PUNCTUATION_TRANS)
            caption = "<start> " + caption.lower().strip() + " <end>"
            caption = tokenizer(caption)
            if len(caption) <= max_length:
                preprocessed_captions[image_id].append(caption)
    return preprocessed_captions

def create_vocabulary(tokenized_captions, min_freq=5):
    all_tokens = []
    for image_id in tokenized_captions:
        for caption in tokenized_captions[image_id]:
            all_tokens += caption
    vocab = build_vocab_from_iterator([all_tokens], specials=["<unk>", "<null>"], min_freq=min_freq)
    vocab.set_default_index(vocab["<unk>"])
    return vocab

def rebuild_sentence(model_output, vocabulary):
    sentence = ""
    for token in vocabulary.lookup_tokens(model_output.tolist()):
        if token == "<start>" or token == "<null>":
            continue
        if token == "<end>":
            break
        sentence += token + " "

    return sentence[:-1] + "."

if __name__ == "__main__":
    import os
    train_captions = load_captions(os.path.join("dataset", "annotations", "annotations", "captions_train2017.json"))
    train_pre_captions = preprocess_captions(train_captions, 22)
    vocab = create_vocabulary(train_pre_captions, min_freq=15)

    print(len(vocab))


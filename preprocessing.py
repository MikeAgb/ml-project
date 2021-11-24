
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

def create_vocabulary(tokenized_captions):
    all_tokens = []
    for image_id in tokenized_captions:
        for caption in tokenized_captions[image_id]:
            all_tokens += caption
    vocab = build_vocab_from_iterator([all_tokens], min_freq=1, specials=["<unk>", "<null>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

def rebuild_sentence(model_output, vocabulary):
    return " ".join(vocabulary.lookup_tokens(model_output.tolist()))


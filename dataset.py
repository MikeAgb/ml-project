
import os
from PIL import Image
from functools import lru_cache

import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize, CenterCrop

import preprocessing

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

class CaptionDataset(Dataset):
    """Builds a dataset directly from the images --- we ended up not actually using this class"""

    def __init__(self, image_folder, preprocessed_captions, vocab, max_caption_length=22):
        self.image_folder = image_folder
        self.captions = preprocessed_captions
        self.vocab = vocab
        self.indices = [(image_id, caption_id)
                        for image_id in preprocessed_captions
                        for caption_id in range(len(preprocessed_captions[image_id]))
                        if len(preprocessed_captions[image_id][caption_id]) <= max_caption_length]
        self.tensor_transform = ToTensor()
        self.scale_transform = Resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        self.crop_transform = CenterCrop((224,224))
        # ******** might not be normalizing Correctly should investigate **********
        # self.norm_transform = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.max_caption_length = max_caption_length # includes <start> and <end> tokens, captions shorter than this will be padded with <null> tokens
    
    def __len__(self):
        return len(self.indices)

    @lru_cache(maxsize=128)
    def _get_image(self, image_id, crop_and_scale):
        image_str = str(image_id)
        image_filename = "0" * (12 - len(image_str)) + image_str + ".jpg"
        image_path = os.path.join(self.image_folder, image_filename)
        image = Image.open(image_path)

        if crop_and_scale:
            image = self.scale_transform(image)
            image = self.crop_transform(image)
            image = self.tensor_transform(image)
            # image = self.norm_transform(image)
            return image
        return self.tensor_transform(image)

    def __getitem__(self, idx, crop_and_scale=True):
        image_id, caption_id = self.indices[idx]
        
        image_tensor = self._get_image(image_id, crop_and_scale)

        caption = self.captions[image_id][caption_id]
        labels = torch.zeros(self.max_caption_length, 1)
        for i in range(self.max_caption_length):
            token = ("<null>" if i >= len(caption)
                     else caption[i] if caption[i] in self.vocab
                     else "<unk>")
            labels[i][0] = self.vocab[token]
        return image_tensor, labels


class EncodedDataset(Dataset):
    """Builds the dataset from the encoded images"""

    def __init__(self, vocab, captions, max_caption_length, train_val_test="train", fc_mp="fc", google=True) -> None:
        super(Dataset, self).__init__()
        self.folder = os.path.join("dataset", "features_google" if google else "features", train_val_test, fc_mp)
        # self.images = [image_id for image_id in captions if len(captions[image_id]) >= captions_per_image]
        self.indices = [(image_id, caption_id)
                        for image_id in captions
                        for caption_id in range(len(captions[image_id]))]
        self.max_caption_length = max_caption_length
        self.captions = captions
        # self.captions_per_image = captions_per_image
        self.vocab = vocab
    
    def __len__(self):
        return len(self.indices)

    @lru_cache(maxsize=None)
    def _get_image(self, image_id):
        image_str = str(image_id)
        image_filename = "0" * (12 - len(image_str)) + image_str + ".pt"
        image_tensor = torch.load(os.path.join(self.folder, image_filename))
        return image_tensor.squeeze()

    @lru_cache(maxsize=None)
    def _get_captions(self, image_id, caption_id):
        # labels = torch.zeros(self.max_caption_length, dtype=torch.long)

        # for i in range(self.captions_per_image):
        caption = self.captions[image_id][caption_id]
        caption += ["<null>"] * (self.max_caption_length - len(caption))
        labels = torch.tensor(self.vocab.lookup_indices(caption), dtype=torch.long)
        return labels

    def __getitem__(self, index):
        image_id, caption_id = self.indices[index]
        image_tensor = self._get_image(image_id)
        labels = self._get_captions(image_id, caption_id)
        
        return image_tensor, labels

if __name__ == "__main__":
    captions = preprocessing.load_captions(os.path.join("dataset", "annotations", "annotations", "captions_train2017.json"))
    pre_captions = preprocessing.preprocess_captions(captions, 18)
    vocab = preprocessing.create_vocabulary(pre_captions, min_freq=20)
    print(len(vocab))

    train_ds = EncodedDataset(vocab, pre_captions, 18, fc_mp="mp")
    print(len(train_ds))

    im, l = train_ds[0]
    print(im.shape)
    print(l.shape)

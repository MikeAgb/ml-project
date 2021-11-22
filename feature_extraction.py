"""
Code to extract features from teh vgg16 model
"""

from torchvision.models import vgg16 #googlenet, 
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import glob
import torch
import os

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

# Let's make a helper function to help us find data files
def get_files(file_directory, extension='*.jpg'):
    """
    Arguments:
        file_directory: path to directory to search for files
        extension: desired file type, default *.jpg
    
    Return:
        files: list of files in file_directory with extension
    """
    files = glob.glob(os.path.join(file_directory, extension))
    return files

# define our model
class VGG_extractor(nn.Module):
    def __init__(self):
        super(VGG_extractor, self).__init__()
        self.features = vgg16(pretrained=True).features # convolutional layers
        self.avgpool = vgg16(pretrained=True).avgpool
        self.fc1 = vgg16(pretrained=True).classifier[0] # first layer of classifier
        
    def forward(self, x):
        """Extract first fully connected feature vector"""
        x = self.features(x)
        y = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x , y

def get_image(file_path, model):
    image_path = file_path
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
     
    image = transform(image)
    fcrep, maxpool_rep = model(image.unsqueeze(0))
    return fcrep.detach(), maxpool_rep.detach()
        
if __name__ == "__main__":
    if not os.path.exists(os.path.join("dataset", "features")):
        os.mkdir(os.path.join("dataset", "features"))
        
    model = VGG_extractor().eval()
    
    for ds in ["train","val", "test"]:
        in_folder = os.path.join("dataset", ds, f'{ds}2017')
        fc_out_folder = os.path.join("dataset","features", ds, 'fc')
        mp_out_folder = os.path.join("dataset","features", ds,'mp')
        if not os.path.exists(os.path.join("dataset","features", ds)):
            os.mkdir(os.path.join("dataset","features", ds))
        if not os.path.exists(fc_out_folder):
            os.mkdir(fc_out_folder)
        if not os.path.exists(mp_out_folder):
            os.mkdir(mp_out_folder)
        print(f"extracting {ds} images.")
        image_paths = get_files(in_folder)
        for image_path in tqdm(image_paths):
            fc_layer, mp_layer = get_image(image_path, model)
            fc_filename = os.path.splitext(os.path.basename(image_path))[0] + '.pt'
            mp_filename = os.path.splitext(os.path.basename(image_path))[0] + '.pt'
            torch.save(fc_layer, os.path.join(fc_out_folder, fc_filename))
            torch.save(mp_layer, os.path.join(mp_out_folder, mp_filename))
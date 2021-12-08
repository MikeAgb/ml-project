"""
Code to extract features from teh vgg16 model
"""

from torchvision.models import vgg16, googlenet, inception_v3
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

# Create an Identity layer to replay last fully connected
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):  
        return x


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

class GoogleNet_extractor(nn.Module):
    def __init__(self):
        super(GoogleNet_extractor, self).__init__()
        self.googlenet = googlenet(pretrained=True)
        self.googlenet.avgpool = Identity()
        self.googlenet.dropout = Identity()
        self.googlenet.fc = Identity()
        self.avgpool = googlenet(pretrained=True).avgpool
        self.dropout = googlenet(pretrained=True).dropout 
        
    def forward(self, x):
        """Extract first fully connected feature vector"""
        x = self.googlenet(x).reshape(1024,7,7)
        y = x
        x = self.avgpool(x) 
        x = self.dropout(x) 
        return x , y

class Inception_extractor(nn.Module):
    def __init__(self):
        super(Inception_extractor, self).__init__()
        self.incept = inception_v3(pretrained=True)
        self.incept.avgpool = Identity()
        self.incept.dropout = Identity()
        self.incept.fc = Identity()
        self.avgpool = inception_v3(pretrained=True).avgpool
        self.dropout = inception_v3(pretrained=True).dropout 
        
    def forward(self, x):
        """Extract first fully connected feature vector"""
        x = self.incept(x).reshape(2048,5,5)
        y = x
        x = self.avgpool(x) 
        x = self.dropout(x) 
        return x, y

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
    if not os.path.exists(os.path.join("dataset", "features_google")):
        os.mkdir(os.path.join("dataset", "features_google"))
    if not os.path.exists(os.path.join("dataset", "features_vgg")):
        os.mkdir(os.path.join("dataset", "features_vgg"))
    if not os.path.exists(os.path.join("dataset", "features_incept")):
        os.mkdir(os.path.join("dataset", "features_incept"))
    

    model_vgg = VGG_extractor().eval() 
    model_goo = GoogleNet_extractor().eval()
    model_inc = Inception_extractor().eval() 
    
    for ds in ["train","val", "test"]:
        in_folder = os.path.join("dataset", ds, f'{ds}2017')

        fc_out_folder_vgg = os.path.join("dataset","features_vgg", ds, 'fc')
        mp_out_folder_vgg = os.path.join("dataset","features_vgg", ds,'mp')

        fc_out_folder_goo = os.path.join("dataset","features_google", ds, 'fc')
        mp_out_folder_goo = os.path.join("dataset","features_google", ds,'mp')

        fc_out_folder_inc = os.path.join("dataset","features_incept", ds, 'fc')
        mp_out_folder_inc = os.path.join("dataset","features_incept", ds,'mp')

        if not os.path.exists(os.path.join("dataset","features_vgg", ds)):
            os.mkdir(os.path.join("dataset","features_vgg", ds))

        if not os.path.exists(os.path.join("dataset","features_google", ds)):
            os.mkdir(os.path.join("dataset","features_google", ds))
        
        if not os.path.exists(os.path.join("dataset","features_incept", ds)):
            os.mkdir(os.path.join("dataset","features_incept", ds))


        if not os.path.exists(fc_out_folder_vgg):
            os.mkdir(fc_out_folder_vgg)
        if not os.path.exists(mp_out_folder_vgg):
            os.mkdir(mp_out_folder_vgg)

        if not os.path.exists(fc_out_folder_goo):
            os.mkdir(fc_out_folder_goo)
        if not os.path.exists(mp_out_folder_goo):
            os.mkdir(mp_out_folder_goo)

        if not os.path.exists(fc_out_folder_inc):
            os.mkdir(fc_out_folder_inc)
        if not os.path.exists(mp_out_folder_inc):
            os.mkdir(mp_out_folder_inc)


        print(f"extracting {ds} images.")
        image_paths = get_files(in_folder)
        for image_path in tqdm(image_paths):
            fc_vgg, mp_vgg = get_image(image_path, model_vgg)
            fc_goo, mp_goo = get_image(image_path, model_goo)
            fc_inc, mp_inc = get_image(image_path, model_inc)
    
            filename = os.path.splitext(os.path.basename(image_path))[0] + '.pt'

            torch.save(fc_vgg, os.path.join(fc_out_folder_vgg, filename))
            torch.save(mp_vgg, os.path.join(fc_out_folder_vgg, filename))

            torch.save(fc_inc, os.path.join(fc_out_folder_inc, filename))
            torch.save(mp_inc, os.path.join(fc_out_folder_inc, filename))

            torch.save(fc_goo, os.path.join(fc_out_folder_goo, filename))
            torch.save(mp_goo, os.path.join(fc_out_folder_goo, filename))
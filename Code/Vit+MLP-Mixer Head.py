# The following requires torch v0.12+ and torchvision v0.13+
import random

import torch
import torchvision
from torchinfo import summary
from torch import nn
import data_setup
import engine
from helper_functions import set_seeds
from helper_functions import plot_loss_curves, pred_and_plot_image
import sys
sys.path.append('/home/ubuntu/Test_Capstone/pytorch-image-models/timm')
import timm
import math
import numpy as np
from torch.nn import init


from PIL import Image
from data_setup import create_dataloaders
from torchvision.transforms import Grayscale
print(torch.__version__)
print(torchvision.__version__)

#-----------------------------------------------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


#-----------------------------------------------------------------------------------------------------------------------



model = timm.create_model('vit_base_patch32_224', pretrained=True).to(device)


# use only for less complex mlp mixer
#model.blocks[0].attn.num_heads = 24
#model.blocks[1].attn.num_heads = 16


weights = model.patch_embed.proj.weight
new_weights = weights[:, :1, ...]
model.patch_embed.proj.weight = torch.nn.Parameter(new_weights)



# 3. Freeze the base parameters
for parameter in model.parameters():
    parameter.requires_grad = False

# 4. Change the classifier head (set the seeds to ensure same initialization with linear head)
set_seeds()

#-----------------------------------------------------------------------------------------------------------------------


# Linear head
#model.head = nn.Linear(in_features=768, out_features=10).to(device)


# MLP head
'''model.head = nn.Sequential(
            nn.Linear(in_features=768, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=10))'''



class LSE_Pooling(nn.Module):
    def __init__(self, r=0.1):
        super(LSE_Pooling, self).__init__()
        self.r = r

    def forward(self, x):
        exp_x = torch.exp(x - self.r)
        sum_exp_x = torch.sum(exp_x.unsqueeze(-1), dim=-1)
        lse_x = torch.log(1 / x.shape[-1] * sum_exp_x + self.r)
        return lse_x

# Less Complex MLP Mixer
class MLP_mixer_Head1(nn.Module):
    def __init__(self):
        super(MLP_mixer_Head1, self).__init__()

        self.fc1 = nn.Linear(768, 512)
        self.ln1 = nn.LayerNorm(512)
        self.lrelu1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.lrelu2 = nn.LeakyReLU()

        self.fc3 = nn.Linear(256, 10)

        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.lrelu1(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = self.lrelu2(x)
        x = self.drop(x)

        x = self.fc3(x)
        x = self.drop(x)

        return x


#mlp_mixer_head = MLP_mixer_Head1()



# More Complex MLP Mixer
class MLP_mixer_Head2(nn.Module):
    def __init__(self):
        super(MLP_mixer_Head2, self).__init__()

        # First block
        self.fc1 = nn.Linear(768, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.gelu1 = nn.GELU()
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.gelu2 = nn.GELU()
        self.drop2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.gelu3 = nn.GELU()
        self.drop3 = nn.Dropout(0.2)

        # Second block
        self.fc4 = nn.Linear(256, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.gelu4 = nn.GELU()
        self.drop4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(512, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.gelu5 = nn.GELU()
        self.drop5 = nn.Dropout(0.2)

        self.fc6 = nn.Linear(1024, 768)
        self.bn6 = nn.BatchNorm1d(768)
        self.gelu6 = nn.GELU()
        self.drop6 = nn.Dropout(0.25)

        # Output layer
        self.fc7 = nn.Linear(768, 10)
        self.drop7 = nn.Dropout(0.2)

    def forward(self, x):
        # First block
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.gelu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.gelu2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.gelu3(x)
        x = self.drop3(x)

        # Second block
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.gelu4(x)
        x = self.drop4(x)

        x = self.fc5(x)
        x = self.bn5(x)
        x = self.gelu5(x)
        x = self.drop5(x)

        x = self.fc6(x)
        x = self.bn6(x)
        x = self.gelu6(x)
        x = self.drop6(x)

        # Output layer
        x = self.fc7(x)
        x = self.drop7(x)

        return x

mlp_mixer_head = MLP_mixer_Head2()

model.head = mlp_mixer_head



# pretrained_vit # uncomment for model output
#-----------------------------------------------------------------------------------------------------------------------

# # Print a summary using torchinfo (uncomment for actual output)
summary(model=model,
         input_size=(512, 1, 224, 224), # (batch_size, color_channels, height, width)
         # col_names=["input_size"], # uncomment for smaller output
         col_names=["input_size", "output_size", "num_params", "trainable"],
         col_width=20,
        row_settings=["var_names"])


#-----------------------------------------------------------------------------------------------------------------------

image_path = '/home/ubuntu/Test_Capstone/Data/'

#-----------------------------------------------------------------------------------------------------------------------
# Setup dirs
train_dir = '/home/ubuntu/Test_Capstone/Data_Transformer_Combined/Train/'
test_dir = '/home/ubuntu/Test_Capstone/Data_Transformer_Combined/Test/'

#train_dir = '/home/ubuntu/Test_Capstone/Data_Transformer/Train/'
#test_dir = '/home/ubuntu/Test_Capstone/Data_Transformer/Test/'

#-----------------------------------------------------------------------------------------------------------------------
# augmentation
import cv2

class DataAugmentation(object):
    def __init__(self):
        pass

    def no_aug(self, img):
        return img

    def perspective_flip_aug(self, img):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
            torchvision.transforms.RandomVerticalFlip(p=0.5)])
        return transform(img)

    def sharp_aug(self, img):
        transform = torchvision.transforms.Compose([torchvision.transforms.RandomAdjustSharpness(sharpness_factor=3)])
        return transform(img)

    def affine_aug(self, img):
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))])
        return transform(img)

    def EdgeDetection(self, img):
        img_np = np.array(img)
        img_np = cv2.blur(img_np, ksize=(2, 2))
        edges = cv2.Canny(img_np, 10, 20)
        edges_pil = torchvision.transforms.functional.to_pil_image(edges)
        return edges_pil


    def __call__(self, img):
        aug_func = random.choice([self.no_aug, self.perspective_flip_aug, self.sharp_aug, self.affine_aug])
        return aug_func(img)






#-----------------------------------------------------------------------------------------------------------------------
# Get automatic transforms from pretrained ViT weights
transforms_train = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    DataAugmentation(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485],std=[0.229])])

transforms_test = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485],std=[0.229])])



#-----------------------------------------------------------------------------------------------------------------------

# Setup dataloaders
train_dataloader_pretrained, test_dataloader_pretrained, class_names = data_setup.create_dataloaders(train_dir=train_dir,
        test_dir=test_dir, transform=None, batch_size=512)

train_dataloader_pretrained.dataset.transform = transforms_train
test_dataloader_pretrained.dataset.transform = transforms_test

print(class_names)

#-----------------------------------------------------------------------------------------------------------------------




# Create optimizer and loss function
# Only use weight decay for the complex mlp mixer
optimizer = torch.optim.Adam(params=model.parameters(),lr=1e-4, weight_decay=1e-5)


loss_fn = torch.nn.CrossEntropyLoss()



# Train the classifier head of the pretrained ViT feature extractor model
set_seeds()

best_model_path = '/home/ubuntu/Test_Capstone/Code/MLP_Mixer_Best_Models/delete_later.pt'

pretrained_vit_results = engine.train(model=model,
                                      train_dataloader=train_dataloader_pretrained,
                                      test_dataloader=test_dataloader_pretrained,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=50,
                                      device=device,
                                      best_model_path= best_model_path,
                                      scheduler=None)



#-----------------------------------------------------------------------------------------------------------------------

plot_loss_curves(pretrained_vit_results)

#-----------------------------------------------------------------------------------------------------------------------



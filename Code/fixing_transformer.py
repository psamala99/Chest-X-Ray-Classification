# The following requires torch v0.12+ and torchvision v0.13+

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



from PIL import Image
from data_setup import create_dataloaders
from torchvision.transforms import Grayscale
print(torch.__version__)
print(torchvision.__version__)

#-----------------------------------------------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


#-----------------------------------------------------------------------------------------------------------------------



model = timm.create_model('vit_base_patch32_224', pretrained=True).to(device)

#model.blocks[0].attn.num_heads = 32
model.blocks[1].attn.num_heads = 32
#model.patch_embed.patch_size = (56,56)
weights = model.patch_embed.proj.weight
new_weights = weights[:, :1, ...]
model.patch_embed.proj.weight = torch.nn.Parameter(new_weights)

block=model.blocks[0]
print(block.attn.num_heads)
print(model.patch_embed.patch_size)



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

# MLP Mixer

class LSE_Pooling(nn.Module):
    def __init__(self, r=0.1):
        super(LSE_Pooling, self).__init__()
        self.r = r

    def forward(self, x):
        exp_x = torch.exp(x - self.r)
        sum_exp_x = torch.sum(exp_x.unsqueeze(-1), dim=-1)
        lse_x = torch.log(1 / x.shape[-1] * sum_exp_x + self.r)
        return lse_x


class MLP_mixer_Head(nn.Module):
    def __init__(self):
        super(MLP_mixer_Head, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.ln1 = nn.LayerNorm(256)

        self.fc2 = nn.Linear(256, 100)
        self.ln2 = nn.LayerNorm(100)

        self.fc3 = nn.Linear(100, 10)

        self.gelu = nn.GELU()
        self.drop = nn.Dropout(0.2)
        self.lse_pooling = LSE_Pooling()


    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.drop(x)

        # Apply LSE pooling after fc1
        x = self.lse_pooling(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = self.gelu(x)
        x = self.drop(x)

        # Apply LSE pooling after fc2
        x = self.lse_pooling(x)


        x = self.fc3(x)
        x = self.drop(x)
        return x


mlp_mixer_head = MLP_mixer_Head()



# CNN head
class CNNHead(nn.Module):
    def __init__(self):
        super(CNNHead, self).__init__()
        self.conv1 = nn.Conv2d(768, 576, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(576, 380, kernel_size=3, stride=1, padding=1)


        self.conv3 = nn.Conv2d(380, 160, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(160, 80, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d((1,1))
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(80, 10)

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        x = torch.logsumexp(x.unsqueeze(1), dim=1) - math.log(x.shape[1])
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        x = torch.logsumexp(x.unsqueeze(1), dim=1) - math.log(x.shape[1])
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

cnn_head = CNNHead()

#model.head = cnn_head

model.head = mlp_mixer_head



# pretrained_vit # uncomment for model output
#-----------------------------------------------------------------------------------------------------------------------

# # Print a summary using torchinfo (uncomment for actual output)
summary(model=model,
         input_size=(256, 1, 224, 224), # (batch_size, color_channels, height, width)
         # col_names=["input_size"], # uncomment for smaller output
         col_names=["input_size", "output_size", "num_params", "trainable"],
         col_width=20,
        row_settings=["var_names"])


#-----------------------------------------------------------------------------------------------------------------------

image_path = '/home/ubuntu/Test_Capstone/Data/'

#-----------------------------------------------------------------------------------------------------------------------
# Setup dirs
train_dir = '/home/ubuntu/Test_Capstone/Data_Transformer/Train/'
test_dir = '/home/ubuntu/Test_Capstone/Data_Transformer/Test/'


#-----------------------------------------------------------------------------------------------------------------------

# Get automatic transforms from pretrained ViT weights
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485],std=[0.229])])



#-----------------------------------------------------------------------------------------------------------------------

# Setup dataloaders
train_dataloader_pretrained, test_dataloader_pretrained, class_names = data_setup.create_dataloaders(train_dir=train_dir,
        test_dir=test_dir, transform=transforms, batch_size=256)

print(class_names)

#-----------------------------------------------------------------------------------------------------------------------




# Create optimizer and loss function
optimizer = torch.optim.Adam(params=model.parameters(),lr=1e-3)


loss_fn = torch.nn.CrossEntropyLoss()


# Train the classifier head of the pretrained ViT feature extractor model
set_seeds()
pretrained_vit_results = engine.train(model=model,
                                      train_dataloader=train_dataloader_pretrained,
                                      test_dataloader=test_dataloader_pretrained,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=25,
                                      device=device)



#-----------------------------------------------------------------------------------------------------------------------

plot_loss_curves(pretrained_vit_results)

#-----------------------------------------------------------------------------------------------------------------------

'''pred_transforms=torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224)])

predictions = pred_and_plot_image(model=model,
                            image_path=test_dir + 'Mass/00030594_000.png',
                            class_names=class_names,
                            transform=pred_transforms,
                            device=device)'''


#-----------------------------------------------------------------------------------------------------------------------








#-----------------------------------------------------------------------------------------------------------------------

from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


classes_dict = {'Atelectasis':0,'Cardiomegaly':1,'Consolidation':2,'Effusion':3, 'Infiltration':4,'Mass':5,'No Finding':6,'Nodule':7,'Pleural_Thickening':8,'Pneumothorax':9}


ig_transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485],std=[0.229]),
    lambda x: x.unsqueeze(0)])  # add batch dimension

image = Image.open(test_dir + 'Mass/00030594_000.png')
outputs = ig_transform(image).to(device)
print(outputs.shape)
ig = IntegratedGradients(model)
target_class = 5
attributions = ig.attribute(outputs, target=target_class, n_steps=100)
print(attributions)


attr = attributions.squeeze().unsqueeze(2).cpu().detach().numpy() # returns a [224, 224, 1] numpy array


_ = viz.visualize_image_attr(None, ig_transform(image).squeeze(0).cpu().permute(1, 2, 0).squeeze().detach().numpy(),
                      method="original_image", title="Original Image")


default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#0000ff'),
                                                  (1, '#0000ff')], N=224)

_ = viz.visualize_image_attr(attr,
                             ig_transform(image).squeeze(0).cpu().permute(1, 2, 0).squeeze().detach().numpy(),
                             method='heat_map',
                             cmap='magma',
                             show_colorbar=True,
                             sign='positive',
                             title='Integrated Gradients')







'''import matplotlib.pyplot as plt
plt.imshow(attributions.squeeze(0).cpu().permute(1,2,0).squeeze().detach().numpy(), cmap='gray')


'''
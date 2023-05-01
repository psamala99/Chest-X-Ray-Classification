import torch
import torchvision
from PIL import Image
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



device = "cuda" if torch.cuda.is_available() else "cpu"
#-----------------------------------------------------------------------------------------------------------------------
# Setup dirs
train_dir = '/home/ubuntu/Test_Capstone/Data_Transformer/Train/'
test_dir = '/home/ubuntu/Test_Capstone/Data_Transformer/Test/'

# Load Model
#best_dict = torch.load('/home/ubuntu/Test_Capstone/Code/MLP_Mixer_Best_Models/less_comp_mlp_mixer_10_classes.pt')
#best_dict = torch.load('/home/ubuntu/Test_Capstone/Code/MLP_Mixer_Best_Models/more_comp_mlp_mixer.pt')
best_dict = torch.load('/home/ubuntu/Test_Capstone/Code/CNN_Best_Models/EfficentNet_bs_256_10_classes.pt')
#-----------------------------------------------------------------------------------------------------------------------

model = timm.create_model('vit_base_patch32_224', pretrained=True).to(device)


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



mlp_mixer_head = MLP_mixer_Head1()



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

#mlp_mixer_head = MLP_mixer_Head2()


# pretrained EfficentNetV3 head
class EfficientNetV3FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        efficientnetv3 = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(efficientnetv3.children())[:-1])  # Remove the original classifier
        self.classifier = nn.Sequential(
            nn.Linear(1280, 1024), nn.LeakyReLU(), nn.Linear(1024, 10))
        self.features[0] = nn.Conv2d(768, 32, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)

        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

efficientnetv3_head = EfficientNetV3FeatureExtractor()

model.head = efficientnetv3_head

#model.head = mlp_mixer_head


model.load_state_dict(best_dict)
# Move Model to Device
model.to(device)

#-----------------------------------------------------------------------------------------------------------------------



'''pred_transforms=torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224)])

class_names=['Atelectasis_Pneumothorax','Cardiomegaly','Consolidation_Infiltration','Mass_Nodule','No Finding','Pleural_Thickening_Effusion']

predictions = pred_and_plot_image(model=model,
                            image_path='/home/ubuntu/Test_Capstone/Data_Transformer_Combined/Chest-X-ray.png',
                            class_names=class_names,
                            transform=pred_transforms,
                            device=device)'''



#-----------------------------------------------------------------------------------------------------------------------
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


#classes_dict = {'Atelectasis_Pneumothorax':0,'Cardiomegaly':1,'Consolidation_Infiltration':2,'Mass_Nodule':3, 'No Finding':4,'Pleural_Thickening_Effusion':5}
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
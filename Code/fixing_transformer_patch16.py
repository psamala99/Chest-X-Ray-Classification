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



from PIL import Image
from data_setup import create_dataloaders
from torchvision.transforms import Grayscale
print(torch.__version__)
print(torchvision.__version__)

#-----------------------------------------------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


#-----------------------------------------------------------------------------------------------------------------------



model = timm.create_model('vit_base_patch32_224', pretrained=True).to(device)




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

# LSE Pooling

class LSE_Pooling(nn.Module):
    def __init__(self, r=0.2):
        super(LSE_Pooling, self).__init__()
        self.r = r

    def forward(self, x):
        exp_x = torch.exp(x - self.r)
        sum_exp_x = torch.sum(exp_x.unsqueeze(-1), dim=-1)
        lse_x = torch.log(1 / x.shape[-1] * sum_exp_x + self.r)
        return lse_x



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


# pretrained resnet50 head
class ResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet18 = torchvision.models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet18.children())[:-1])  # Remove the original classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10))
        self.features[0] = nn.Conv2d(768, 64, kernel_size=7, stride=2, padding=3,
                                     bias=False)  # Modify the first layer to accept hidden state

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)

        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

resnet18_head = ResNet18FeatureExtractor()


# initializing head

model.head = efficientnetv3_head

#model.head = resnet18_head



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



class EdgeDetection(object):
    def __call__(self, img):
        # Convert image to numpy array
        img_np = np.array(img)

        img_np = cv2.blur(img_np, ksize=(2, 2))

        # Apply Canny edge detection filter
        edges = cv2.Canny(img_np, 10, 20)

        # Convert edges back to PIL Image and return
        edges_pil = torchvision.transforms.functional.to_pil_image(edges)
        return edges_pil

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
        test_dir=test_dir, transform = None, batch_size=256)

train_dataloader_pretrained.dataset.transform = transforms_train
test_dataloader_pretrained.dataset.transform = transforms_test

print(class_names)

#-----------------------------------------------------------------------------------------------------------------------




# Create optimizer and loss function
optimizer = torch.optim.Adam(params=model.parameters(),lr=1e-3)


loss_fn = torch.nn.CrossEntropyLoss()



# Train the classifier head of the pretrained ViT feature extractor model
set_seeds()

best_model_path = '/home/ubuntu/Test_Capstone/Code/CNN_Best_Models/delete_later.pt'

pretrained_vit_results = engine.train(model=model,
                                      train_dataloader=train_dataloader_pretrained,
                                      test_dataloader=test_dataloader_pretrained,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=50,
                                      device=device,
                                      best_model_path=best_model_path,
                                      scheduler=None)



#-----------------------------------------------------------------------------------------------------------------------

plot_loss_curves(pretrained_vit_results)

#-----------------------------------------------------------------------------------------------------------------------

pred_transforms=torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224)])

predictions = pred_and_plot_image(model=model,
                            image_path=test_dir + 'Mass/00030594_000.png',
                            class_names=class_names,
                            transform=pred_transforms,
                            device=device)


#-----------------------------------------------------------------------------------------------------------------------








#-----------------------------------------------------------------------------------------------------------------------

'''from captum.attr import IntegratedGradients
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
                             title='Integrated Gradients')'''

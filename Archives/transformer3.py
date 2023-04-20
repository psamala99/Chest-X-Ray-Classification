# The following requires torch v0.12+ and torchvision v0.13+
import torch
import torchvision
from torchinfo import summary
from torch import nn
import data_setup
import engine
from helper_functions import set_seeds
from PIL import Image
from data_setup import create_dataloaders
from torchvision.transforms import Grayscale
from helper_functions import plot_loss_curves
print(torch.__version__)
print(torchvision.__version__)

#-----------------------------------------------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


#-----------------------------------------------------------------------------------------------------------------------


# 1. Get pretrained weights for ViT-Base
#pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT  # requires torchvision >= 0.13, "DEFAULT" means best available
pretrained_vit_weights = torchvision.models.ViT_B_32_Weights.DEFAULT



# 2. Setup a ViT model instance with pretrained weights
#pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)
pretrained_vit = torchvision.models.vit_b_32(weights=pretrained_vit_weights).to(device)



# 3. Freeze the base parameters
for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False

# 4. Change the classifier head (set the seeds to ensure same initialization with linear head)
set_seeds()
# Linear head
#pretrained_vit.heads = nn.Linear(in_features=768, out_features=10).to(device)

# MLP head
'''pretrained_vit.heads = nn.Sequential(
            nn.Linear(in_features=768, out_features=1536),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=1536, out_features=10))'''

# CNN head
class CNNHead(nn.Module):
    def __init__(self):
        super(CNNHead, self).__init__()
        self.conv1 = nn.Conv2d(768, 576, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(576, 380, kernel_size=3, stride=1, padding=1)


        self.conv3 = nn.Conv2d(380, 160, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(160, 80, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((1,1))
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(80, 10)

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

cnn_head = CNNHead()
pretrained_vit.heads = cnn_head

for parameter in cnn_head.parameters():
    parameter.requires_grad = True


    # pretrained_vit # uncomment for model output
#-----------------------------------------------------------------------------------------------------------------------

# # Print a summary using torchinfo (uncomment for actual output)
summary(model=pretrained_vit,
         input_size=(1024, 3, 224, 224), # (batch_size, color_channels, height, width)
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
pretrained_vit_transforms = pretrained_vit_weights.transforms()
#print(pretrained_vit_transforms)


#-----------------------------------------------------------------------------------------------------------------------
'''gray_resize_transform = torchvision.transforms.Compose([
    Grayscale(),
    torchvision.transforms.Resize((224, 224)),
])

def copy_channels(img):
    return Image.merge("RGB", [img]*3)

pretrained_vit_transforms = torchvision.transforms.Compose([
    gray_resize_transform,
    torchvision.transforms.Lambda(copy_channels),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])'''

#-----------------------------------------------------------------------------------------------------------------------

# Setup dataloaders
train_dataloader_pretrained, test_dataloader_pretrained, class_names = data_setup.create_dataloaders(train_dir=train_dir,
        test_dir=test_dir, transform=pretrained_vit_transforms, batch_size=1024)


#-----------------------------------------------------------------------------------------------------------------------




# Create optimizer and loss function
optimizer = torch.optim.Adam(params=pretrained_vit.parameters(),lr=1e-3)
#optimizer = torch.optim.AdamW(params=pretrained_vit.parameters(),lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()


# Train the classifier head of the pretrained ViT feature extractor model
set_seeds()
pretrained_vit_results = engine.train(model=pretrained_vit,
                                      train_dataloader=train_dataloader_pretrained,
                                      test_dataloader=test_dataloader_pretrained,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=2,
                                      device=device)

#-----------------------------------------------------------------------------------------------------------------------

plot_loss_curves(pretrained_vit_results)



#-----------------------------------------------------------------------------------------------------------------------

imgs = [test_dir + 'Mass/00030594_000.png', test_dir + 'Mass/00030159_014.png']

labels_path = '/home/ubuntu/Test_Capstone/Data_Transformer/Clean_Data.xlsx'




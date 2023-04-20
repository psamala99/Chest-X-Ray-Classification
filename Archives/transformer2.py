# The following requires torch v0.12+ and torchvision v0.13+
import torch
import torchvision
from torchinfo import summary
from torch import nn
import data_setup
import engine
from helper_functions import set_seeds
from data_setup import create_dataloaders
print(torch.__version__)
print(torchvision.__version__)

#-----------------------------------------------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


#-----------------------------------------------------------------------------------------------------------------------


# 1. Get pretrained weights for ViT-Base
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT  # requires torchvision >= 0.13, "DEFAULT" means best available

# 2. Setup a ViT model instance with pretrained weights
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

# 3. Freeze the base parameters
for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False

# 4. Change the classifier head (set the seeds to ensure same initialization with linear head)
set_seeds()
pretrained_vit.heads = nn.Linear(in_features=768, out_features=10).to(device)
# pretrained_vit # uncomment for model output
#-----------------------------------------------------------------------------------------------------------------------

# # Print a summary using torchinfo (uncomment for actual output)
summary(model=pretrained_vit,
         input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
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
print(pretrained_vit_transforms)


#-----------------------------------------------------------------------------------------------------------------------

# Setup dataloaders
train_dataloader_pretrained, test_dataloader_pretrained, class_names = data_setup.create_dataloaders(train_dir=train_dir,
        test_dir=test_dir, transform=pretrained_vit_transforms, batch_size=32)

#-----------------------------------------------------------------------------------------------------------------------




# Create optimizer and loss function
optimizer = torch.optim.Adam(params=pretrained_vit.parameters(),lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the classifier head of the pretrained ViT feature extractor model
set_seeds()
pretrained_vit_results = engine.train(model=pretrained_vit,
                                      train_dataloader=train_dataloader_pretrained,
                                      test_dataloader=test_dataloader_pretrained,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=1,
                                      device=device)





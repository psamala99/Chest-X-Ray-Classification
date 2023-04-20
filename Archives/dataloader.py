#------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import transformers
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
import os
#------------------------------------------------------------------------------------------------------------------
'''
LAST UPDATED 11/10/2021, lsdr
02/14/2022 am ldr checking for consistency
02/14/2022 pm ldr class model same as train_solution pytorch, change of numpy() to cpu().numpy()
'''
#------------------------------------------------------------------------------------------------------------------

OR_PATH = os.getcwd()
os.chdir("..") # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep

os.chdir(OR_PATH) # Come back to the folder where the code resides , all files will be left on this directory

n_epoch = 50
BATCH_SIZE = 64
LR = 0.001

## Image processing
CHANNELS = 3
IMAGE_SIZE = 100
DROPOUT = 0.5

NICKNAME = "best_model"

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SAVE_MODEL = True






#------------------------------------------------------------------------------------------------------------------

class Dataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''
    def __init__(self, list_IDs, type_data, target_type):
        #Initialization'
        self.type_data = type_data
        self.list_IDs = list_IDs
        self.target_type = target_type

    def __len__(self):
        #Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        #Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]


        # Load data and get label

        if self.type_data == 'train':
            y = xdf_dset.target_class.get(ID)

            if self.target_type == 2:
                y = y.split(",")
        else:
            y = xdf_dset_test.target_class.get(ID)

            if self.target_type == 2:
                y = y.split(",")

        if self.target_type == 2:
            labels_ohe = [ int(e) for e in y]
        else:

            labels_ohe = np.zeros(OUTPUTS_a)

            for idx, label in enumerate(range(OUTPUTS_a)):
                if label == y:
                    labels_ohe[idx] = 1
                    #print(labels_ohe)

        y = torch.FloatTensor(labels_ohe)


        if self.type_data == 'train':
            file = DATA_DIR + xdf_dset.id.get(ID)
        else:
            file = DATA_DIR + xdf_dset_test.id.get(ID)

        img = cv2.imread(file)

        img= cv2.resize(img,(IMAGE_SIZE, IMAGE_SIZE))


        # Augmentation only for train
        X = torch.FloatTensor(img)

        X = torch.reshape(X, (CHANNELS, IMAGE_SIZE, IMAGE_SIZE))

        return X, y
#------------------------------------------------------------------------------------------------------------------

def read_data(target_type):

    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])
    ds_targets = xdf_dset['target_class']

    # ---------------------- Parameters for the data loader --------------------------------

    list_of_ids = list(xdf_dset.index)
    list_of_ids_test = list(xdf_dset_test.index)


    # Datasets
    partition = {
        'train': list_of_ids,
        'test' : list_of_ids_test
    }

    # Data Loaders


    params = {'batch_size': BATCH_SIZE}

    training_set = Dataset(partition['train'], 'train', target_type)
    training_sampler = data.RandomSampler(training_set, replacement=True, num_samples=5000, generator=None)
    training_generator = data.DataLoader(training_set, **params, sampler= training_sampler)



    params = {'batch_size': BATCH_SIZE}

    test_set = Dataset(partition['test'], 'test', target_type)
    test_generator = data.DataLoader(test_set, **params)

    ## Make the channel as a list to make it variable

    return training_generator, test_generator
#------------------------------------------------------------------------------------------------------------------

def process_target(target_type):
    '''
        1- Multiclass  target = (1...n, text1...textn)
        2- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
        3- Multiclass or Multilabel ( binary  ( 0,1 ) )
    :return:
    '''


    if target_type == 1:
        # takes the classes and then
        xtarget = list(np.array(xdf_data['target'].unique()))
        le = LabelEncoder()
        le.fit(xtarget)
        final_target = le.transform(np.array(xdf_data['target']))
        class_names=(xtarget)
        xdf_data['target_class'] = final_target
        xdf_data.to_csv('/home/ubuntu/Test_Capstone/excel/Tar_Data.csv')


    if target_type == 2:
        target = np.array(xdf_data['target'].apply( lambda x : x.split(",")))
        final_target = mlb.fit_transform(target)
        xfinal = []
        if len(final_target) ==0:
            xerror = 'Could not process Multilabel'
            class_names = []
        else:
            class_names = mlb.classes_
            for i in range(len(final_target)):
                joined_string = ",".join( str(e) for e in final_target[i])
                xfinal.append(joined_string)
            xdf_data['target_class'] = xfinal


    if target_type == 3:
        target = np.array(xdf_data['target'].apply(lambda x: x.split(",")))
        final_target = mlb.fit_transform(target)
        xfinal = []
        if len(final_target) == 0:
            xerror = 'Could not process Multilabel'
            class_names = []
        else:
            class_names = mlb.classes_

    ## We add the column to the main dataset


    return class_names
#------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    for file in os.listdir(PATH+os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH+ os.path.sep+ "excel" + os.path.sep + file

    # Reading and filtering Excel file
    xdf_data = pd.read_excel(FILE_NAME)


    ## Process Classes
    ## Input and output

    ## Processing Train dataset
    ## Target_type = 1  Multiclass   Target_type = 2 MultiLabel
    class_names = process_target(target_type = 1)

    ## Comment

    xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()


    xdf_dset_test= xdf_data[xdf_data["split"] == 'test'].copy()

    ## read_data creates the dataloaders, take target_type = 2

    train_ds,test_ds = read_data(target_type = 1)

    OUTPUTS_a = len(class_names)
#---------------------------------------------------------------------------------------------------------------------

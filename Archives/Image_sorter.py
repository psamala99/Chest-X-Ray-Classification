import pandas as pd
import shutil


#--------------------------------------------------Test Functions-------------------------------------------------------
def split_test(i):
    shutil.copy2(IMAGES + i, IMAGES_test + i)
    return

def test_move_Consolidation(i):
    shutil.move(IMAGES_test + i, IMAGES_test_Consolidation + i)
    return

def test_move_Pneumothorax(i):
    shutil.move(IMAGES_test + i, IMAGES_test_Pneumothorax + i)
    return

def test_move_Pleural_Thickening(i):
    shutil.move(IMAGES_test + i, IMAGES_test_Pleural_Thickening + i)
    return

def test_move_Nodule(i):
    shutil.move(IMAGES_test + i, IMAGES_test_Nodule + i)
    return

def test_move_No_Finding(i):
    shutil.move(IMAGES_test + i, IMAGES_test_No_Finding + i)
    return

def test_move_Mass(i):
    shutil.move(IMAGES_test + i, IMAGES_test_Mass + i)
    return

def test_move_Infiltration(i):
    shutil.move(IMAGES_test + i, IMAGES_test_Infiltration + i)
    return

def test_move_Effusion(i):
    shutil.move(IMAGES_test + i, IMAGES_test_Effusion + i)
    return

def test_move_Cardiomegaly(i):
    shutil.move(IMAGES_test + i, IMAGES_test_Cardiomegaly + i)
    return

def test_move_Atelectasis(i):
    shutil.move(IMAGES_test + i, IMAGES_test_Atelectasis + i)
    return


#--------------------------------------------------Train Functions------------------------------------------------------
def split_train(i):
    shutil.copy2(IMAGES + i, IMAGES_train + i)
    return

def train_move_Consolidation(i):
    shutil.move(IMAGES_train + i, IMAGES_train_Consolidation + i)
    return

def train_move_Pneumothorax(i):
    shutil.move(IMAGES_train + i, IMAGES_train_Pneumothorax + i)
    return

def train_move_Pleural_Thickening(i):
    shutil.move(IMAGES_train + i, IMAGES_train_Pleural_Thickening + i)
    return

def train_move_Nodule(i):
    shutil.move(IMAGES_train + i, IMAGES_train_Nodule + i)
    return

def train_move_No_Finding(i):
    shutil.move(IMAGES_train + i, IMAGES_train_No_Finding + i)
    return

def train_move_Mass(i):
    shutil.move(IMAGES_train + i, IMAGES_train_Mass + i)
    return

def train_move_Infiltration(i):
    shutil.move(IMAGES_train + i, IMAGES_train_Infiltration + i)
    return

def train_move_Effusion(i):
    shutil.move(IMAGES_train + i, IMAGES_train_Effusion + i)
    return

def train_move_Cardiomegaly(i):
    shutil.move(IMAGES_train + i, IMAGES_train_Cardiomegaly + i)
    return

def train_move_Atelectasis(i):
    shutil.move(IMAGES_train + i, IMAGES_train_Atelectasis + i)
    return

#--------------------------------------------------Source/Destinations-------------------------------------------------------
IMAGES = '/home/ubuntu/Test_Capstone/Data/'

IMAGES_test = '/home/ubuntu/Test_Capstone/Data_Transformer/Test/'

IMAGES_test_Consolidation = '/home/ubuntu/Test_Capstone/Data_Transformer/Test/Consolidation/'
IMAGES_test_Pneumothorax = '/home/ubuntu/Test_Capstone/Data_Transformer/Test/Pneumothorax/'
IMAGES_test_Pleural_Thickening = '/home/ubuntu/Test_Capstone/Data_Transformer/Test/Pleural_Thickening/'
IMAGES_test_Nodule = '/home/ubuntu/Test_Capstone/Data_Transformer/Test/Nodule/'
IMAGES_test_No_Finding = '/home/ubuntu/Test_Capstone/Data_Transformer/Test/No Finding/'
IMAGES_test_Mass = '/home/ubuntu/Test_Capstone/Data_Transformer/Test/Mass/'
IMAGES_test_Infiltration = '/home/ubuntu/Test_Capstone/Data_Transformer/Test/Infiltration/'
IMAGES_test_Effusion = '/home/ubuntu/Test_Capstone/Data_Transformer/Test/Effusion/'
IMAGES_test_Cardiomegaly = '/home/ubuntu/Test_Capstone/Data_Transformer/Test/Cardiomegaly/'
IMAGES_test_Atelectasis = '/home/ubuntu/Test_Capstone/Data_Transformer/Test/Atelectasis/'

IMAGES_train = '/home/ubuntu/Test_Capstone/Data_Transformer/Train/'

IMAGES_train_Consolidation = '/home/ubuntu/Test_Capstone/Data_Transformer/Train/Consolidation/'
IMAGES_train_Pneumothorax = '/home/ubuntu/Test_Capstone/Data_Transformer/Train/Pneumothorax/'
IMAGES_train_Pleural_Thickening = '/home/ubuntu/Test_Capstone/Data_Transformer/Train/Pleural_Thickening/'
IMAGES_train_Nodule = '/home/ubuntu/Test_Capstone/Data_Transformer/Train/Nodule/'
IMAGES_train_No_Finding = '/home/ubuntu/Test_Capstone/Data_Transformer/Train/No Finding/'
IMAGES_train_Mass = '/home/ubuntu/Test_Capstone/Data_Transformer/Train/Mass/'
IMAGES_train_Infiltration = '/home/ubuntu/Test_Capstone/Data_Transformer/Train/Infiltration/'
IMAGES_train_Effusion = '/home/ubuntu/Test_Capstone/Data_Transformer/Train/Effusion/'
IMAGES_train_Cardiomegaly = '/home/ubuntu/Test_Capstone/Data_Transformer/Train/Cardiomegaly/'
IMAGES_train_Atelectasis = '/home/ubuntu/Test_Capstone/Data_Transformer/Train/Atelectasis/'

#--------------------------------------------------Read Clean Data Excel-------------------------------------------------------
df_sorting = pd.read_excel('/home/ubuntu/Test_Capstone/excel/Clean_Data.xlsx')

#--------------------------------------------------Test Dataset-------------------------------------------------------
df_sorting_test = df_sorting[df_sorting["split"] == 'test'].copy()
df_sorting_test['id'].apply(split_test)

df_sorting_test_Consolidation = df_sorting_test[df_sorting_test["target"] == 'Consolidation'].copy()
df_sorting_test_Consolidation['id'].apply(test_move_Consolidation)

df_sorting_test_Pneumothorax = df_sorting_test[df_sorting_test["target"] == 'Pneumothorax'].copy()
df_sorting_test_Pneumothorax['id'].apply(test_move_Pneumothorax)

df_sorting_test_Pleural_Thickening = df_sorting_test[df_sorting_test["target"] == 'Pleural_Thickening'].copy()
df_sorting_test_Pleural_Thickening['id'].apply(test_move_Pleural_Thickening)

df_sorting_test_Nodule = df_sorting_test[df_sorting_test["target"] == 'Nodule'].copy()
df_sorting_test_Nodule['id'].apply(test_move_Nodule)

df_sorting_test_No_Finding = df_sorting_test[df_sorting_test["target"] == 'No Finding'].copy()
df_sorting_test_No_Finding['id'].apply(test_move_No_Finding)

df_sorting_test_Mass = df_sorting_test[df_sorting_test["target"] == 'Mass'].copy()
df_sorting_test_Mass['id'].apply(test_move_Mass)

df_sorting_test_Cardiomegaly = df_sorting_test[df_sorting_test["target"] == 'Cardiomegaly'].copy()
df_sorting_test_Cardiomegaly['id'].apply(test_move_Cardiomegaly)

df_sorting_test_Atelectasis = df_sorting_test[df_sorting_test["target"] == 'Atelectasis'].copy()
df_sorting_test_Atelectasis['id'].apply(test_move_Atelectasis)

df_sorting_test_Effusion = df_sorting_test[df_sorting_test["target"] == 'Effusion'].copy()
df_sorting_test_Effusion['id'].apply(test_move_Effusion)

df_sorting_test_Infiltration = df_sorting_test[df_sorting_test["target"] == 'Infiltration'].copy()
df_sorting_test_Infiltration['id'].apply(test_move_Infiltration)

#--------------------------------------------------Train Functions-------------------------------------------------------

df_sorting_train = df_sorting[df_sorting["split"] == 'train'].copy()
df_sorting_train['id'].apply(split_train)

df_sorting_train_Consolidation = df_sorting_train[df_sorting_train["target"] == 'Consolidation'].copy()
df_sorting_train_Consolidation['id'].apply(train_move_Consolidation)

df_sorting_train_Pneumothorax = df_sorting_train[df_sorting_train["target"] == 'Pneumothorax'].copy()
df_sorting_train_Pneumothorax['id'].apply(train_move_Pneumothorax)

df_sorting_train_Pleural_Thickening = df_sorting_train[df_sorting_train["target"] == 'Pleural_Thickening'].copy()
df_sorting_train_Pleural_Thickening['id'].apply(train_move_Pleural_Thickening)

df_sorting_train_Nodule = df_sorting_train[df_sorting_train["target"] == 'Nodule'].copy()
df_sorting_train_Nodule['id'].apply(train_move_Nodule)

df_sorting_train_No_Finding = df_sorting_train[df_sorting_train["target"] == 'No Finding'].copy()
df_sorting_train_No_Finding['id'].apply(train_move_No_Finding)

df_sorting_train_Mass = df_sorting_train[df_sorting_train["target"] == 'Mass'].copy()
df_sorting_train_Mass['id'].apply(train_move_Mass)

df_sorting_train_Cardiomegaly = df_sorting_train[df_sorting_train["target"] == 'Cardiomegaly'].copy()
df_sorting_train_Cardiomegaly['id'].apply(train_move_Cardiomegaly)

df_sorting_train_Atelectasis = df_sorting_train[df_sorting_train["target"] == 'Atelectasis'].copy()
df_sorting_train_Atelectasis['id'].apply(train_move_Atelectasis)

df_sorting_train_Effusion = df_sorting_train[df_sorting_train["target"] == 'Effusion'].copy()
df_sorting_train_Effusion['id'].apply(train_move_Effusion)

df_sorting_train_Infiltration = df_sorting_train[df_sorting_train["target"] == 'Infiltration'].copy()
df_sorting_train_Infiltration['id'].apply(train_move_Infiltration)
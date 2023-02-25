import pandas as pd
from PIL import Image
import random


df_main = pd.read_excel('/home/ubuntu/Test_Capstone/excel/Clean_Data.xlsx')
Data_dir = '/home/ubuntu/Test_Capstone/Data/'

#-------------------------------------function---------------------------------------#

# function to copy and append subsets to main data frame
def append_df_to_excel(df, FILE_NAME):
    df_excel = pd.read_excel(FILE_NAME)
    result = pd.concat([df_excel, df])
    result.to_excel(FILE_NAME)

# function to isolate necessary columns
def shuffle_excel(FILE_NAME):
    df_excel = pd.read_excel(FILE_NAME)
    df_main = df_excel.sample(frac=1).reset_index(drop=True)
    df_main = df_main[['Image Index', 'Finding Labels']]
    df_main.to_excel(FILE_NAME)

# function to rotate image
def rot_img(img):
    img = img.rotate(45)
    return img

# function to crop image
def crop_img(img):
    (left, upper, right, lower) = (20, 20, 100, 100)
    img = img.crop((left, upper, right, lower))
    return img

my_list = [rot_img, crop_img]

def aug(i):
    file = Data_dir + i
    Img = Image.open(file)
    img2 = Img.copy()
    img2 = random.choice(my_list)(img2)
    img2.save(Data_dir + 'aug_' + i)
    return img2

# classes that are augmented: Consolidation, Pleural_Thickening, Cardiomegaly

#-------------------------------------Consolidation---------------------------------------#
df_main_Consolidation = df_main[df_main["Finding Labels"] == 'Consolidation'].copy()

df_main_Consolidation ['Image Index'].apply(aug)

df_main_Consolidation ['Image Index'] = str('aug_') + df_main_Consolidation ['Image Index']

#-------------------------------------Pleural_Thickening---------------------------------------#
df_main_Pleural_Thickening = df_main[df_main["Finding Labels"] == 'Pleural_Thickening'].copy()

df_main_Pleural_Thickening ['Image Index'].apply(aug)

df_main_Pleural_Thickening ['Image Index'] = str('aug_') + df_main_Pleural_Thickening ['Image Index']

#-------------------------------------Cardiomegaly---------------------------------------#
df_main_Cardiomegaly = df_main[df_main["Finding Labels"] == 'Cardiomegaly'].copy()

df_main_Cardiomegaly ['Image Index'].apply(aug)

df_main_Cardiomegaly ['Image Index'] = str('aug_') + df_main_Cardiomegaly ['Image Index']

#-------------------------------------Appending new ids to excel---------------------------------------#

FILE_NAME = '/home/ubuntu/Test_Capstone/excel/Clean_Data.xlsx'

append_df_to_excel(df_main_Consolidation,FILE_NAME)

append_df_to_excel(df_main_Pleural_Thickening,FILE_NAME)

append_df_to_excel(df_main_Cardiomegaly,FILE_NAME)

shuffle_excel(FILE_NAME)






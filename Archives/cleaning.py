import pandas as pd
import numpy as np
import re

# %%
df = pd.read_csv('/home/ubuntu/Test_Capstone/excel/Data_Entry_2017_v2020.csv')
# %%
df.columns
# %%
df['Finding Labels'][1]
# %%
cnt = 0
visited = []
for i in range(0, len(df['Finding Labels'])):

    if df['Finding Labels'][i] not in visited:
        visited.append(df['Finding Labels'][i])

        cnt += 1

print("No.of.unique values :",
      cnt)

# 836 different types of rows. 836 classes including ones with multiple labels.

# or run: df['Finding Labels'].unique().size

# %%
# checking rows that have multiple labels associated with them
# Diseases with multiple labels
df['Finding Labels'].str.contains('[|]') == True
# 20796 rows have diseases with multiple labels.
# %%
# Removing those from the dataset:
df_main = df[df['Finding Labels'].str.contains('[|]') == False]

df_main_No_Finding = df_main[df_main["Finding Labels"] == 'No Finding']
df_main_No_Finding = df_main_No_Finding.sample(n=2500)

df_main_Infiltration = df_main[df_main["Finding Labels"] == 'Infiltration']
df_main_Infiltration = df_main_Infiltration.sample(n=2500)

df_main_Atelectasis = df_main[df_main["Finding Labels"] == 'Atelectasis']
df_main_Atelectasis = df_main_Atelectasis.sample(n=2500)

df_main = df_main[df_main["Finding Labels"] != 'No Finding']
df_main = df_main[df_main["Finding Labels"] != 'Infiltration']
df_main = df_main[df_main["Finding Labels"] != 'Atelectasis']
df_main = df_main[df_main["Finding Labels"] != 'Emphysema']
df_main = df_main[df_main["Finding Labels"] != 'Fibrosis']
df_main = df_main[df_main["Finding Labels"] != 'Edema']
df_main = df_main[df_main["Finding Labels"] != 'Pneumonia']
df_main = df_main[df_main["Finding Labels"] != 'Hernia']


df_main = pd.concat([df_main, df_main_No_Finding, df_main_Infiltration, df_main_Atelectasis])


# %%
# Unique labels in updated dataset:
df_main['Finding Labels'].nunique()
# names of labels:
df_main['Finding Labels'].nunique()
# target value numbers associated with each disease:
print(df_main['Finding Labels'].value_counts())
# Data looks good. There are many 'No finding' labels (60k), with other labels just being in thousands.
#print(df_main)
df_main.to_excel('/home/ubuntu/Test_Capstone/excel/Clean_Data.xlsx')


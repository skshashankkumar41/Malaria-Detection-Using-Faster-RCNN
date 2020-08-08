import pandas as pd
import numpy as np
import pickle
import os
import random
import json
from collections import defaultdict
from tqdm import tqdm
from imutils import paths 

# reading training json file and creating a dataframe for bounding boxes and image paths
print("creating training annotated data...")
train = pd.read_json ('input/training.json')
data = []
for i in tqdm(range(train.shape[0])):
    for j in range(len(train.iloc[i,1])):
        img_name = train.iloc[i,0]['pathname'].split('/')[2]
        label = train.iloc[i,1][j]['category']
        x_min = train.iloc[i,1][j]['bounding_box']['minimum']['c']
        x_max = train.iloc[i,1][j]['bounding_box']['maximum']['c']
        y_min = train.iloc[i,1][j]['bounding_box']['minimum']['r']
        y_max = train.iloc[i,1][j]['bounding_box']['maximum']['r']
        
        data.append([img_name,label,x_min,y_min,x_max,y_max])

df_train = pd.DataFrame(data,columns=['img_name','label','x_min','y_min','x_max','y_max'])

df_train.img_name = df_train.img_name.apply(lambda x: "training_images/"+str(x))

# removing all data with category as difficult 
df_train = df_train[df_train['label'] != "difficult"]
df_train.to_csv('annotated_data/original_train_annotation.csv',index=False)
# all non-rbc cells 
non_rbc = ['trophozoite', 'schizont', 'ring','gametocyte','leukocyte']

# converting all cells other than rbc to non-rbc
for i in range(df_train.shape[0]):
    if df_train.iloc[i,1] in non_rbc:
        df_train.iloc[i,1] = 'non_rbc'

# reading testing json file and creating a dataframe for bounding boxes and image paths
print("creating testing annotated data...")
test = pd.read_json ('input/test.json')

data = []
for i in tqdm(range(test.shape[0])):
    for j in range(len(test.iloc[i,1])):
        img_name = test.iloc[i,0]['pathname'].split('/')[2]
        label = test.iloc[i,1][j]['category']
        x_min = test.iloc[i,1][j]['bounding_box']['minimum']['c']
        x_max = test.iloc[i,1][j]['bounding_box']['maximum']['c']
        y_min = test.iloc[i,1][j]['bounding_box']['minimum']['r']
        y_max = test.iloc[i,1][j]['bounding_box']['maximum']['r']
        
        data.append([img_name,label,x_min,y_min,x_max,y_max])

df_test = pd.DataFrame(data,columns = ['img_name','label','x_min','y_min','x_max','y_max'])

df_test.img_name = df_test.img_name.apply(lambda x: "testing_images/"+str(x))

# removing all data with category as difficult 
df_test = df_test[df_test['label'] != "difficult"]

df_train.to_csv('annotated_data/original_test_annotation.csv',index=False)
# converting all cells other than rbc to non-rbc
for i in range(df_test.shape[0]):
    if df_test.iloc[i,1] in non_rbc:
        df_test.iloc[i,1] = 'non_rbc'

# saving annotated training and test csv
df_train.to_csv('annotated_data/train_annotation.csv',index=False)
df_test.to_csv('annotated_data/test_annotation.csv',index=False)
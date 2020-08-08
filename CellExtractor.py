import pandas as pd
import numpy as np
from imutils import paths 
import os
import random
from tqdm import tqdm
import matplotlib.image as mpimg
from PIL import Image

train = pd.read_json('input/training.json')

print("Creating Dataframe with Bounding Boxes of Training Data...")
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

#creating dataframe of all non-rbc cells
df_non_rbc=df_train[(df_train.label!='red blood cell') & (df_train.label!='difficult')]

df_non_rbc.img_name = df_non_rbc.img_name.apply(lambda x: "input/training_images/"+str(x))
df_non_rbc.reset_index(drop=True,inplace=True)

if not os.path.exists('output/cell_images'):
    os.makedirs('output/cell_images')

print("cropping cell images of each sample and storing...")

# cropping non_rbc cells from images based on there bounding boxes
for i in tqdm(range(2252)):
    bbox=df_non_rbc.iloc[i,2:].values
    im=Image.open(df_non_rbc.iloc[i,0])
    im=im.crop(df_non_rbc.iloc[i,2:].values)
    im.save('output/cell_images/{}_{}.png'.format(i,df_non_rbc.iloc[i,1]))
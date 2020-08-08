import pandas as pd
import numpy as np
from imutils import paths 
import os
import random
from tqdm import tqdm
import matplotlib.image as mpimg
from PIL import Image
import pandas as pd
import numpy as np
from collections import defaultdict
from imutils import paths 
import os
import random
from tqdm import tqdm
from sklearn.externals import joblib 
from keras.applications import VGG16 #importing Keras implementation of the pre-trained VGG16 network
from keras.applications import imagenet_utils #Utilities for ImageNet data preprocessing & prediction decoding
from keras.preprocessing.image import img_to_array,load_img #importing functions to load image and covert to array
from keras.models import load_model
from keras import backend as K
from collections import defaultdict
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import imutils
import cv2
from PIL import Image

# train = pd.read_json('input/test.json')

# print("Creating Dataframe with Bounding Boxes of Training Data...")
# data = []
# for i in tqdm(range(train.shape[0])):
#     for j in range(len(train.iloc[i,1])):
#         img_name = train.iloc[i,0]['pathname'].split('/')[2]
#         label = train.iloc[i,1][j]['category']
#         x_min = train.iloc[i,1][j]['bounding_box']['minimum']['c']
#         x_max = train.iloc[i,1][j]['bounding_box']['maximum']['c']
#         y_min = train.iloc[i,1][j]['bounding_box']['minimum']['r']
#         y_max = train.iloc[i,1][j]['bounding_box']['maximum']['r']
        
#         data.append([img_name,label,x_min,y_min,x_max,y_max])

# df_train = pd.DataFrame(data,columns=['img_name','label','x_min','y_min','x_max','y_max'])

# #creating dataframe of all non-rbc cells
# df_non_rbc=df_train[(df_train.label!='red blood cell') & (df_train.label!='difficult')]

# df_non_rbc.img_name = df_non_rbc.img_name.apply(lambda x: "input/testing_images/"+str(x))
# df_non_rbc.reset_index(drop=True,inplace=True)

# if not os.path.exists('output/cell_images_testing'):
#     os.makedirs('output/cell_images_testing')

# print("cropping cell images of each sample and storing...")

# # cropping non_rbc cells from images based on there bounding boxes
# for i in tqdm(range(df_non_rbc.shape[0])):
#     bbox=df_non_rbc.iloc[i,2:].values
#     im=Image.open(df_non_rbc.iloc[i,0])
#     im=im.crop(df_non_rbc.iloc[i,2:].values)
#     im.save('output/cell_images_testing/{}_{}.png'.format(i,df_non_rbc.iloc[i,1]))


imagePaths=list(paths.list_images("output/cell_images_testing")) 
lr_model= joblib.load('output/models/model_LR.pkl')  
vgg_model=VGG16(weights="imagenet", include_top=False)
classes=['gametocyte', 'leukocyte', 'ring', 'schizont', 'trophozoite']
labels = []
targets =[]
for i in imagePaths:
    targets.append(i.split(os.path.sep)[1].split('_')[1].split('.')[0])
    im=Image.open(i)
    cr_img=im.resize((224,224))   
    data=np.array(cr_img)
    data=np.expand_dims(data,axis=0)
    data=imagenet_utils.preprocess_input(data)
    data=vgg_model.predict(data)
    data=np.array(data)
    data=data.reshape(1,512*7*7)
    pred = lr_model.predict(data)
    label = classes[pred[0]]
    labels.append(label)
labelsMapping = {'gametocyte': 3, 'leukocyte': 4, 'ring': 1, 'schizont': 2, 'trophozoite': 0}

final_labels = [labelsMapping[i] for i in labels]
final_targets = [labelsMapping[i] for i in targets]

from sklearn.metrics import classification_report
cl = classification_report(final_labels,final_targets)
print(classification_report(final_labels,final_targets))
print(type(cl))
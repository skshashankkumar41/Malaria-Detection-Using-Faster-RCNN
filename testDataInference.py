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

test=pd.read_json ('input/test.json')

data=[]
for i in tqdm(range(test.shape[0])):
    for j in range(len(test.iloc[i,1])):
        img_name=test.iloc[i,0]['pathname'].split('/')[2]
        label=test.iloc[i,1][j]['category']
        x_min=test.iloc[i,1][j]['bounding_box']['minimum']['c']
        x_max=test.iloc[i,1][j]['bounding_box']['maximum']['c']
        y_min=test.iloc[i,1][j]['bounding_box']['minimum']['r']
        y_max=test.iloc[i,1][j]['bounding_box']['maximum']['r']
        
        data.append([img_name,label,x_min,y_min,x_max,y_max])
df_test=pd.DataFrame(data,columns=['img_name','label','x_min','y_min','x_max','y_max'])   

non_rbc = ['trophozoite', 'schizont', 'ring','gametocyte','leukocyte']

for i in range(df_test.shape[0]):
    if df_test.iloc[i,1] in non_rbc:
        df_test.iloc[i,1]='non_rbc'

df_test.img_name=df_test.img_name.apply(lambda x: "input/testing_images/"+str(x))
df_test_two=df_test[df_test['label']!="difficult"]

data=[]
for i in tqdm(range(test.shape[0])):
    for j in range(len(test.iloc[i,1])):
        img_name=test.iloc[i,0]['pathname'].split('/')[2]
        label=test.iloc[i,1][j]['category']
        x_min=test.iloc[i,1][j]['bounding_box']['minimum']['c']
        x_max=test.iloc[i,1][j]['bounding_box']['maximum']['c']
        y_min=test.iloc[i,1][j]['bounding_box']['minimum']['r']
        y_max=test.iloc[i,1][j]['bounding_box']['maximum']['r']
        
        data.append([img_name,label,x_min,y_min,x_max,y_max])
df_test=pd.DataFrame(data,columns=['img_name','label','x_min','y_min','x_max','y_max'])
df_test.img_name=df_test.img_name.apply(lambda x: "testing_images/"+str(x))
df_test_all=df_test[df_test['label']!="difficult"]

testing_images=np.unique(df_test_two.img_name.values)

modelPath = "output/models/frozen_inference_graph.pb" #saved Faster-RCNN model graph
labels_loc = "output/records/classes.pbtxt" #saved classes files
min_confidence = 0.5
num_classes = 2

lr_model= joblib.load('output/models/model_LR.pkl')  

model=tf.Graph()
with model.as_default():
    graphDef=tf.GraphDef()
    with tf.gfile.GFile("C:/Users/SKS/Desktop/AAIC/Malaria_Detection_TS/experiments/exported_model/frozen_inference_graph.pb","rb") as f:
        serializedGraph=f.read()
        graphDef.ParseFromString(serializedGraph)
        tf.import_graph_def(graphDef,name="")
        
labelMap=label_map_util.load_labelmap(labels_loc)
categories=label_map_util.convert_label_map_to_categories(labelMap,max_num_classes=num_classes,use_display_name=True)
categoryIdx=label_map_util.create_category_index(categories)
classes=['gametocyte', 'leukocyte', 'ring', 'schizont', 'trophozoite']

testing_predicition_rnn={}
testing_predicition_vgg={}
predicted_boxes_stacked_test=defaultdict(dict)

with model.as_default():
    with tf.Session(graph=model) as sess:
        imageTensor=model.get_tensor_by_name("image_tensor:0")
        boxesTensor=model.get_tensor_by_name("detection_boxes:0")
        scoresTensor=model.get_tensor_by_name("detection_scores:0")
        classesTensor=model.get_tensor_by_name("detection_classes:0")
        numDetections=model.get_tensor_by_name("num_detections:0")
        with tf.Session() as sess2:
            vgg_model=VGG16(weights="imagenet", include_top=False)
            for img in tqdm(testing_images):
            # loading just 1 image for testing 
                image=cv2.imread(img)
                (H,W)= image.shape[:2]  

                output=image.copy()
                image=cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)
                image=np.expand_dims(image,axis=0)

                (boxes,scores,labels,N)= sess.run([boxesTensor,scoresTensor,classesTensor,numDetections],feed_dict={imageTensor:image})
                boxes=np.squeeze(boxes)
                scores=np.squeeze(scores)
                labels=np.squeeze(labels)
                o=[]
                boxes_nm=[]
                for (box,score,label) in zip(boxes,scores,labels):
                    if score<0.4:
                        continue
                    (startY,startX,endY,endX)=box
                    startX=int(startX*W)
                    startY=int(startY*H)
                    endX=int(endX*W)
                    endY=int(endY*H)
                    if img in predicted_boxes_stacked_test:
                        predicted_boxes_stacked_test[img]['boxes'].append([startX,startY,endX,endY])
                        predicted_boxes_stacked_test[img]['scores'].append(score)
                    else:
                        predicted_boxes_stacked_test[img]['boxes']=[[startX,startY,endX,endY]]
                        predicted_boxes_stacked_test[img]['scores']=[score]
                    if categoryIdx[label]['name']=="non_rbc":
                        if categoryIdx[label]['name'] in testing_predicition_rnn:
                            testing_predicition_rnn[categoryIdx[label]['name']]+=1
                        else:
                            testing_predicition_rnn[categoryIdx[label]['name']]=1
                        b_box=[startX,startY,endX,endY]
                        im=Image.open(img)
                        cr_img=im.crop(b_box)
                        cr_img=cr_img.resize((224,224))   
                        data=np.array(cr_img)
                        data=np.expand_dims(data,axis=0)
                        data=imagenet_utils.preprocess_input(data)
                        data=vgg_model.predict(data)
                        data=np.array(data)
                        data=data.reshape(1,512*7*7)
                        # Initialize model
                        pred = lr_model.predict(data)
                        label = classes[pred[0]]
                        if label in testing_predicition_vgg:
                            testing_predicition_vgg[label]+=1
                        else:
                            testing_predicition_vgg[label]=1
                    else:
                        label=categoryIdx[label]
                        idx=int(label["id"])-1
                        label=label['name']
                        if label in testing_predicition_rnn:
                            testing_predicition_rnn[label]+=1
                        else:
                            testing_predicition_rnn[label]=1
                        if label in testing_predicition_vgg:
                                testing_predicition_vgg[label]+=1
                        else:
                            testing_predicition_vgg[label]=1

print("Ground Truth for F-RCNN::",dict(df_test_two.label.value_counts()))
print("Prediction for F-RCNN::",testing_predicition_rnn)

print("Ground Truth for Finetuned VGG-16::",dict(df_test_all.label.value_counts()))
print("Prediction for Finetuned VGG-16::",testing_predicition_vgg)
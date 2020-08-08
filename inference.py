import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths 
import os
import random
import matplotlib.image as mpimg
from tqdm import tqdm
from sklearn.externals import joblib 
from matplotlib import patches
from keras.applications import VGG16
from keras.applications import imagenet_utils
from object_detection.utils import label_map_util
import tensorflow as tf
from keras.models import load_model
import imutils
import cv2
from PIL import Image
import PIL.Image

model="output/models/frozen_inference_graph.pb" #saved Faster-RCNN model graph
labels_loc="output/records/classes.pbtxt" #saved classes files
min_confidence=0.5
num_classes=2

colors=np.float64(np.array([[255,1,1],
       [86, 1,255],
       [1,231,255],
       [1,255,61],
       [214,255,1],
       [255,120,1]]))

def inference(image_paths):
    report = 'Negative'
    lr_model= joblib.load('output/models/model_LR.pkl')  
    model=tf.Graph()
    with model.as_default():
        graphDef=tf.GraphDef()
        with tf.gfile.GFile("output/models/frozen_inference_graph.pb","rb") as f:
            serializedGraph=f.read()
            graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(graphDef,name="")

    labelMap=label_map_util.load_labelmap(labels_loc)
    categories=label_map_util.convert_label_map_to_categories(labelMap,max_num_classes=num_classes,use_display_name=True)
    categoryIdx=label_map_util.create_category_index(categories)
    classes=['gametocyte', 'leukocyte', 'ring', 'schizont', 'trophozoite']
    predicition={}
    with model.as_default():
        with tf.Session(graph=model) as sess:
            imageTensor=model.get_tensor_by_name("image_tensor:0")
            boxesTensor=model.get_tensor_by_name("detection_boxes:0")
            scoresTensor=model.get_tensor_by_name("detection_scores:0")
            classesTensor=model.get_tensor_by_name("detection_classes:0")
            numDetections=model.get_tensor_by_name("num_detections:0")
            with tf.Session() as sess2:
                vgg_model=VGG16(weights="imagenet", include_top=False)
                for img in tqdm(image_paths):
                # loading just 1 image for testing 
                    image=cv2.imread(img)
                    (H,W)= image.shape[:2]  
                    output=image.copy()
                    image=cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)
                    image=np.expand_dims(image,axis=0)
                    print("read image")
                    (boxes,scores,labels,N)= sess.run([boxesTensor,scoresTensor,classesTensor,numDetections],feed_dict={imageTensor:image})
                    boxes=np.squeeze(boxes)
                    scores=np.squeeze(scores)
                    labels=np.squeeze(labels)
                    o=[]
                    boxes_nm=[]
                    print("started predicting classes")
                    for (box,score,label) in zip(boxes,scores,labels):
                        if score<0.4:
                            continue
                        (startY,startX,endY,endX)=box
                        startX=int(startX*W)
                        startY=int(startY*H)
                        endX=int(endX*W)
                        endY=int(endY*H)
                        if categoryIdx[label]['name']=="non_rbc":
                            b_box=[startX,startY,endX,endY]
                            im=PIL.Image.open(img)
                            cr_img=im.crop(b_box)
                            cr_img=cr_img.resize((224,224))   
                            data=np.array(cr_img)
                            data=data[:,:,:3]
                            data=np.expand_dims(data,axis=0)
                            data=imagenet_utils.preprocess_input(data)
                            data=vgg_model.predict(data)
                            data=np.array(data)
                            data=data.reshape(1,512*7*7)
                            pred = lr_model.predict(data)
                            label = classes[pred[0]]
                            if label != 'leukocyte':
                                report = 'Positive'
                            cv2.rectangle(output,(startX,startY),(endX,endY),[0,0,255],3)
                            y=startY-10 if startY-10>10 else startY+10 
                            cv2.putText(output,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,[0,0,255],2)
                            if label in predicition:
                                predicition[label]+=1
                            else:
                                predicition[label]=1
                        else:
                            label=categoryIdx[label]
                            idx=int(label["id"])-1
                            label=label['name']
                            cv2.rectangle(output,(startX,startY),(endX,endY),[255,0,0],3)
                            y=startY-10 if startY-10>10 else startY+10 
                            cv2.putText(output,'rbc',(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,[255,0,0],2)
                            if label in predicition:
                                predicition[label]+=1
                            else:
                                predicition[label]=1
            output = cv2.resize(output, (800, 600))  
            cv2.imshow("output",output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return report


print(inference(["input/training_images/0ca25c88-457f-4f03-bbc1-98fb6663f1d1.png"]))
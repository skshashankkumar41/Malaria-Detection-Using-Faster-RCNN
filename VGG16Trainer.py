#Ref- https://medium.com/@skshashankkumar41/transfer-learning-8386e1a9e34a
import numpy as np
import os 
from keras.applications import VGG16 #importing Keras implementation of the pre-trained VGG16 network
from keras.applications import imagenet_utils #Utilities for ImageNet data preprocessing & prediction decoding
from keras.preprocessing.image import img_to_array,load_img #importing functions to load image and covert to array
from keras.layers import Input
import random
from tqdm import tqdm
from keras.models import Model
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from keras.layers.core import Dense,Dropout,Flatten
from imutils import paths #it will create a list of paths for each image 
from sklearn.preprocessing import LabelBinarizer #it will encode categories into numerical value
from sklearn.model_selection import train_test_split #library for train_test split of data 
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression    #importing Logistic regression model
from sklearn.model_selection import GridSearchCV    #GridSearchCV for hyper-parameter tuning
from sklearn.externals import joblib 

#it contains path for each image in our folder
imagePaths=list(paths.list_images("output/cell_images")) 
random.shuffle(imagePaths) 

#it will extract the labels from the path of each image
labels = [p.split(os.path.sep)[1].split('_')[1].split('.')[0] for p in imagePaths]
classNames = [str(x) for x in np.unique(labels)]

#loading the VGG16 pre-trained on imagenet network
model = VGG16(weights="imagenet", include_top=False)     

#list which will have 25088 featurs for each image
data=[]     
for i in tqdm(imagePaths):
    image=load_img(i,target_size=(224,224))     #loading image by there paths 
    image=img_to_array(image)     #converting images into arrays 
    image = np.expand_dims(image, axis=0)     #inserting a new dimension because keras need extra dimensions 
    image = imagenet_utils.preprocess_input(image)     #preprocessing image according to imagenet data
    features=model.predict(image)     #extracting those features from the model
    data.append(features)     #appending features to the list
data=np.array(data)     #converting list into array
data=data.reshape(data.shape[0],512*7*7)

#splitting train and test into 70:30
X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=.15,stratify=labels,random_state=0) 

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
weights={}
for i in range(5):
    weights[i]=class_weights[i]

# convert the labels from integers to vectors
le=LabelEncoder()
y_train=le.fit_transform(y_train)
y_test=le.transform(y_test)

#hyper-parameter tuning parameters for logistic regression
params = {"C": [0.01,0.1, 1.0, 10.0, 100.0]}    
model = GridSearchCV(LogisticRegression(class_weight=weights), params, cv=3,verbose=1)
model.fit(X_train,y_train)

if not os.path.exists('output/models'):
    os.makedirs('output/models')
# Save the model as a pickle in a file 
joblib.dump(model,'output/models/model_LR.pkl') 
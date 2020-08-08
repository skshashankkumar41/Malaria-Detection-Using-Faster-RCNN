import os
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from TFAnnotation import TFAnnotation

if not os.path.exists('output/records'):
    os.makedirs('output/records')

BASE_PATH = "G:/AIP/Malaria-Detection"
TRAIN_ANNOT_PATH = os.path.sep.join([BASE_PATH,"annotated_data/train_annotation.csv"])#train csv path
TEST_ANNOT_PATH = os.path.sep.join([BASE_PATH,"annotated_data/test_annotation.csv"])#test csv path
TRAIN_RECORD = os.path.sep.join([BASE_PATH,"output/records/training.record"])#record file required by Tensorflow object detection
TEST_RECORD = os.path.sep.join([BASE_PATH,"output/records/testing.record"])
CLASSES_FILE = os.path.sep.join([BASE_PATH,"output/records/classes.pbtxt"])#class file required by Tensorflow object detection
CLASSES = {"red blood cell":1,"non_rbc":2}

#creating class file required by TFOD
f = open(CLASSES_FILE,"w")
for (k,v) in CLASSES.items():
    item = ("item {\n"
    "\tid: " + str(v) + "\n"
    "\tname: '" + k + "'\n"
    "}\n")
    f.write(item)
f.close()

D={}
train_rows=open(TRAIN_ANNOT_PATH).read().strip().split("\n")

#creating the required .record file for TFOD 
for row in tqdm(train_rows[1:]):
    row = row.split(',')
    (imagePath,label,startX,startY,endX,endY) = row
    (startX,startY) = (float(startX),float(startY))
    (endX,endY) = (float(endX),float(endY))
    if label not in CLASSES:
        continue
    p = os.path.sep.join([BASE_PATH,'input/'+imagePath])
    b = D.get(p,[])
    b.append((label,(startX,startY,endX,endY)))
    D[p] = b
    
(trainKeys,testKeys) = train_test_split(list(D.keys()),test_size=.10,random_state=42)
datasets = [
    ("train",trainKeys,TRAIN_RECORD),
    ("test",testKeys,TEST_RECORD)
]
for dType,keys,outputPath in datasets:
    print("processing{}".format(dType))
    writer = tf.python_io.TFRecordWriter(outputPath)
    total = 0
    for k in tqdm(keys):
        encoded = tf.gfile.GFile(k,'rb').read()
        encoded = bytes(encoded)
        pilImage = Image.open(k)
        (w,h) = pilImage.size[:2]
        filename = k.split(os.path.sep)[-1]
        encoding = filename[filename.rfind(".")+ 1:]

        tfAnnot = TFAnnotation()
        tfAnnot.image = encoded
        tfAnnot.encoding = encoding
        tfAnnot.filename = filename
        tfAnnot.width = w
        tfAnnot.height = h
        for (label,(startX,startY,endX,endY)) in D[k]:
            xMin = startX/w
            xMax = endX/w
            yMin = startY/h
            yMax = endY/h
            tfAnnot.xMins.append(xMin)
            tfAnnot.xMaxs.append(xMax)
            tfAnnot.yMins.append(yMin)
            tfAnnot.yMaxs.append(yMax)
            tfAnnot.textLabels.append(label.encode('utf8'))
            tfAnnot.classes.append(CLASSES[label])
            tfAnnot.difficult.append(0)
            total += 1
        
        features = tf.train.Features(feature=tfAnnot.build())
        example = tf.train.Example(features=features)

        writer.write(example.SerializeToString())
    writer.close()
    print('{} saved {}'.format(total,dType))

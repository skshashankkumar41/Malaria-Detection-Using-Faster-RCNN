# Malaria Detection using Faster-RCNN and Transfer Learning

Given images of blood samples predict whether a patient has malaria or not using Tensorflow Object Detection API and Transfer Learning
<br>
<table>
  <tr>
     <td><p align="center">Blood Sample Image </p> </td>
     <td><p align="center">After Detection Image</p> </td>
  </tr>
  <tr>
    <td><img src="https://i.ibb.co/tz4J6Gn/git-ori.jpg" width = "500" height = "250"/></td>
    <td> <img src="https://i.ibb.co/nf634g4/git-out.jpg" width = "500" height = "250"/></td>
  </tr>
 </table>

# Data
Data came from three different labs’ ex vivo samples of P. vivax infected patients in Manaus, Brazil, and Thailand. The Manaus and Thailand data were used for training and validation while the Brazil data were left out as our test set. Blood smears were stained with Giemsa reagent, which attaches to DNA and allow experts to inspect infected cells and determine their stage.

It can be downloaded from [Kaggle](https://www.kaggle.com/kmader/malaria-bounding-boxes)

# Approach
As the data is highly imbalanced and almost 98% cells are Red Blood cells, I made this problem a two stage classification task.
<p align="center"><img src="https://i.ibb.co/5jQsL1Y/git-dist.jpg" alt="mal-dist" border="0" height = 300 width =500></p>

#### First Stage
Label all the cells other than RBC as Non RBC cells and then create a object detector that only detects either cell is RBC or NON RBC. For this task I used Tensorflow Object Detection API to fine-tune and train a Faster RCNN model pre-trained on coco dataset.

#### Second Stage
For all the NON RBC cells create a CNN Classifier to classify given a NON RBC cell which category it belongs (trophozoite, schizont, ring, leukocyte, gametocyte). I fine-tuned VGG-16 Model trained on imagenet data.

#### Two Stage Classification
So by combining the first stage and second stage we build a complete two stage classification model in which we will supply a blood sample image, Faster RCNN detector will detect and gives bouding boxes of RBC and NON-RBC in the image and then for all the NON-RBC we will crop the NON-RBC and pass that cell image to our VGG-16 classifier and it will predict the categories of NON-RBC.

<p align="center"><img src="https://i.ibb.co/gZSy529/git-graph.jpg" alt="mal-dist" border="0" ></p>

# Results

## Training Data 
### MAP Scores
<img src="https://i.ibb.co/JqqJ08X/train-map.jpg" alt="train-map" border="0">

### Counts
<br>
<table border = 2px solid black >
<tr>
<td></td>
<td>Grount Truth Count</td>
<td>Model Truth Count</td>
</tr>
<tr>
<td>red blood cell</td>
<td>77420</td>
<td>71040</td>
</tr>
<tr>
<td>trophozoite</td>
<td>1473</td>
<td>1513</td>
</tr>
<tr>
<td>ring</td>
<td>353</td>
<td>311</td>
</tr>
<tr>
<td>schizont</td>
<td>179</td>
<td>178</td>
</tr>
<tr>
<td>gametocyte</td>
<td>144</td>
<td>157</td>
</tr>
<tr>
<td>leukocyte</td>
<td>103</td>
<td>87</td>
</tr>
</table>

## Testing Data
### MAP Score
<img src="https://i.ibb.co/dm1nTTw/test-map.jpg" alt="test-map" border="0">

### Counts
<br>
<table border = 2px solid black>
<tr>
<td></td>
<td>Grount Truth Count</td>
<td>Model Truth Count</td>
</tr>
<tr>
<td>red blood cell</td>
<td>5614</td>
<td>4551</td>
</tr>
<tr>
<td>trophozoite</td>
<td>111</td>
<td>146</td>
</tr>
<tr>
<td>ring</td>
<td>169</td>
<td>95</td>
</tr>
<tr>
<td>schizont</td>
<td>11</td>
<td>6</td>
</tr>
<tr>
<td>gametocyte</td>
<td>12</td>
<td>7</td>
</tr>
<tr>
<td>leukocyte</td>
<td>0</td>
<td>0</td>
</tr>
</table>


# Usage
Clone the repository and run the following commands from the terminal.

#### Setup Tensorflow Object Detection API
Go through this [blog](https://medium.com/@marklabinski/installing-tensorflow-object-detection-api-on-windows-10-7a4eb83e1e7b) to setup TFOD on your system.

#### Install project dependencies from requirements.txt
```
 pip install -r requirements.txt
```
#### Creating .records and classes files
```
 python TFODRecordCreator.py
```
#### Training Faster RCNN 
In the experiment folder paste the Faster RCNN resnet101 [coco model](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md) in training subfolder and then run the following command from TFOD folder
```
python object_detection/model_main.py --pipeline_config_path faster_rcnn_malaria.config --model_dir experiment/training --num_train_steps 20000 --sample_1_of_n_eval_examples 10 --alsologtostderr
```

#### Export Faster RCNN Model
Run the following command from TFOD folder
```
python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path faster_rcnn_malaria.config --trained_checkpoint_prefix experiments/training/model.ckpt-20000 --output_directory output/models
```
#### Crop and Extract NON-RBC Images
```
 python CellExtractor.py  
```
#### Train VGG-16 on Extracted NON-RBC Images
```
python VGG16Trainer.py
```
## Inference 
```
python inference.py --imagePath "test.png"
```

#  Files

<b>[preprocessing.py](https://github.com/skshashankkumar41/Malaria-Detection-Using-Faster-RCNN/blob/master/preprocessing.py)</b><br>
As data is present in json file this file creates annotated csv files which will be required by our Tensorflow Object Detection API.

<b>[TFAnnotation.py](https://github.com/skshashankkumar41/Malaria-Detection-Using-Faster-RCNN/blob/master/TFAnnotation.py)</b><br>
This class defines the format of data required by Tensorflow Object Detection API.

<b>[TFODRecordCreator.py](https://github.com/skshashankkumar41/Malaria-Detection-Using-Faster-RCNN/blob/master/TFODRecordCreator.py)</b><br>
This script creates the .record and classes files using TFAnnotation class which is required by Tensorflow Object Detection API to train Faster RCNN model.

<b>[plotting.py](https://github.com/skshashankkumar41/Malaria-Detection-Using-Faster-RCNN/blob/master/plotting.py)</b><br>
Script for plotting of bouding boxes of RBC, NON RBC on images.

<b>[CellExtractor.py](https://github.com/skshashankkumar41/Malaria-Detection-Using-Faster-RCNN/blob/master/CellExtractor.py)</b><br>
Script for cropping and extracting NON-RBC images using bounding boxes from main image for the training of VGG-16 model.

<b>[VGG16Trainer.py](https://github.com/skshashankkumar41/Malaria-Detection-Using-Faster-RCNN/blob/master/VGG16Trainer.py)</b><br>
Script for training of cropped NON-RBC images using VGG-16 pretrained model.

<b>[trainDataInference.py](https://github.com/skshashankkumar41/Malaria-Detection-Using-Faster-RCNN/blob/master/trainDataInference.py), [testDataInference](https://github.com/skshashankkumar41/Malaria-Detection-Using-Faster-RCNN/blob/master/testDataInference.py)</b><br>
Script for evaluation of training and testing images using two-staged classification.

<b>[inference.py](https://github.com/skshashankkumar41/Malaria-Detection-Using-Faster-RCNN/blob/master/inference.py)</b><br>
Function to predict the result of blood sample image and also shows the image of input blood sample with labels of each cell and bouding box.


# Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.



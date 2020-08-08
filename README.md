# Malaria Detection using Faster-RCNN and Transfer Learning

Given images of blood samples predict whether patient has malaria or not using Tensorflow Object Detection API and Transfer Learing 
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

                                                              

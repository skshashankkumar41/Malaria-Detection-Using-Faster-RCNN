import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import patches

# plotting function to plot bouding boxes of all cells

def plotAll(imagePath,annotatedPath):
    df = pd.read_csv(annotatedPath)

    fig = plt.figure()

    #add axes to the image
    ax = fig.add_axes([0,0,1,1])

    # read and plot the image
    image = plt.imread(imagePath)

    annotated_imgPath = "/".join(imagePath.split("/")[-2:])

    for _,row in df[df.img_name == annotated_imgPath].iterrows():
        x_min = row.x_min
        x_max = row.x_max
        y_min = row.y_min
        y_max = row.y_max
        
        width = x_max - x_min
        height = y_max - y_min
        
        # assign different color to different classes of objects
        if row.label == 'red blood cell':
            edgecolor = 'r'
            ax.annotate('RBC', xy=(x_max-40,y_min+20),fontsize = 9.0,color = 'r',fontfamily = 'fantasy')
        elif row.label == 'trophozoite':
            edgecolor = 'b'
            ax.annotate('trophozoite', xy=(x_max-40,y_min+20),fontsize = 12.0,color = 'b',fontfamily = 'fantasy')
        elif row.label == 'schizont':
            edgecolor = 'y'
            ax.annotate('v', xy=(x_max-40,y_min+20),fontsize = 12.0,color = 'y',fontfamily = 'fantasy')
        elif row.label == 'ring':
            edgecolor = 'g'
            ax.annotate('ring', xy=(x_max-40,y_min+20),fontsize = 12.0,color = 'g',fontfamily = 'fantasy')
        elif row.label == 'gametocyte':
            edgecolor = 'c'
            ax.annotate('gametocyte', xy=(x_max-40,y_min+20),fontsize = 12.0,color = 'c',fontfamily = 'fantasy')
        elif row.label == 'leukocyte':
            edgecolor = 'm'
            ax.annotate('leukocyte', xy=(x_max-40,y_min+20),fontsize = 12.0,color = 'm',fontfamily = 'fantasy')
        
        # add bounding boxes to the image
        rect = patches.Rectangle((x_min,y_min), width, height, edgecolor = edgecolor, facecolor = 'none', linewidth = 1.5)

        ax.add_patch(rect)
    
    plt.imshow(image)
    plt.show()

def plotRBCNonRBC(imagePath,annotatedPath):
    df = pd.read_csv(annotatedPath)

    fig = plt.figure()

    #add axes to the image
    ax = fig.add_axes([0,0,1,1])

    # read and plot the image
    image = plt.imread(imagePath)

    annotated_imgPath = "/".join(imagePath.split("/")[-2:])

    for _,row in df[df.img_name == annotated_imgPath].iterrows():
        x_min = row.x_min
        x_max = row.x_max
        y_min = row.y_min
        y_max = row.y_max
        
        width = x_max - x_min
        height = y_max - y_min
        
        # assign different color to different classes of objects
        if row.label == 'red blood cell':
            edgecolor = 'r'
            ax.annotate('rbc', xy=(x_max-40,y_min+20))
        elif row.label == 'non_rbc':
            edgecolor = 'y'
            ax.annotate('non_rbc', xy=(x_max-40,y_min+20))
            
        # add bounding boxes to the image
        rect = patches.Rectangle((x_min,y_min), width, height, edgecolor = edgecolor, facecolor = 'none', linewidth = 1.5)
        
        ax.add_patch(rect)
    
    plt.imshow(image)
    plt.show()

# plotting function to plot bouding boxes of only non-rbc cells
def plotNonRBC(imagePath,annotatedPath):
    df = pd.read_csv(annotatedPath)

    fig = plt.figure()

    #add axes to the image
    ax = fig.add_axes([0,0,1,1])

    # read and plot the image
    image = plt.imread(imagePath)

    annotated_imgPath = "/".join(imagePath.split("/")[-2:])

    for _,row in df[df.img_name == annotated_imgPath].iterrows():
        x_min = row.x_min
        x_max = row.x_max
        y_min = row.y_min
        y_max = row.y_max
        
        width = x_max - x_min
        height = y_max - y_min
        k=0
        # assign different color to different classes of objects
        if row.label == 'red blood cell':
            k=1
        elif row.label == 'non_rbc':
            edgecolor = 'y'
            ax.annotate('non_rbc', xy=(x_max-40,y_min+20))
        
        if k!=1:    
            # add bounding boxes to the image
            rect = patches.Rectangle((x_min,y_min), width, height, edgecolor = edgecolor, facecolor = 'none', linewidth = 1.5)
            ax.add_patch(rect)
    
    plt.imshow(image)
    plt.show()
    
plotAll("input/training_images/0ca25c88-457f-4f03-bbc1-98fb6663f1d1.png","annotated_data/original_train_annotation.csv")
plotRBCNonRBC("input/training_images/0ca25c88-457f-4f03-bbc1-98fb6663f1d1.png","annotated_data/train_annotation.csv")
plotNonRBC("input/training_images/0ca25c88-457f-4f03-bbc1-98fb6663f1d1.png","annotated_data/train_annotation.csv")
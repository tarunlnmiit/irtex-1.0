#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
from torch import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
#import cv2
import sys
from skimage import segmentation
import torch.nn.init
from cv2 import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.measure import compare_ssim as ssim
from sklearn.metrics.cluster import adjusted_rand_score

#load input data CIFAR 
def Load_data(dataFile):
    print("load datafile started")
    image_array=[]
    dim = (128,128)
    for label in (os.listdir(dataFile)):
        count=0
        for img in (os.listdir(os.path.join(dataFile, label))):
            count=count+1
            img_cat=[]

            img_read=cv2.imread(os.path.join(dataFile, label, img), cv2.IMREAD_UNCHANGED)
            img_read = cv2.resize(img_read, dim, interpolation = cv2.INTER_AREA)


            img_cat.append(img)
            img_cat.append(img_read)
            img_cat.append(label)
            image_array.append(img_cat)
            # if(count==1):
            #     break
    return image_array

#CNN model architecture (Conv + BatchNormalisation + Conv + BatchNormalisation)--> Initial Conv layer

class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.nChannel = 5
        self.nConv = 3
        self.conv1 = nn.Conv2d(input_dim, self.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(self.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        
        for i in range(self.nConv-1): #Adding extra Conv layers and BatchNormalisation to base architecture
            self.conv2.append( nn.Conv2d(self.nChannel,self.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(self.nChannel) )
        self.conv3 = nn.Conv2d(self.nChannel, self.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(self.nChannel)

    # Function working on the provided inputs to NN

    def forward(self, x): #x is the input shape
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(self.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x   #final output from network


#Function to perform segmentation
def Segmentation(image):

        #hyperparameters initialisation
        nChannel=5
        maxIter= 500
        min_clusters= 3   
        lr=0.1
        nConv=3
        num_superpixels= 1000
        compactness= 50
        visualize= 1 
    
        print("Segmentation started")
        #image_res= cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        
        #Input Image provided for Segmentation
        
        cv2.imshow( "input", image )
        cv2.waitKey(10)
        #plt.imshow(image)
                
        data = torch.from_numpy( np.array([image.transpose( (2, 0, 1) ).astype('float32')/255.]) )
        
        # slic to produce a ground truth to compare
        clusters = segmentation.slic(image, compactness=compactness, n_segments=num_superpixels)
        clusters = clusters.reshape(image.shape[0]*image.shape[1])  
        
        #number of unique labels by combining the values for all labels generated 
        unique_cluster = np.unique(clusters)
        l_inds = []
        for i in range(len(unique_cluster)):
            l_inds.append( np.where( clusters == unique_cluster[ i ] )[ 0 ] )
            
        # Training CNN model 
        model = MyNet( data.size(1) )  #data.size(1) is value 3 
        model.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        label_colours = np.random.randint(255,size=(100,3))
        
        for _ in range(maxIter):
        # forwarding
            optimizer.zero_grad()
            output = model( data )[ 0 ]
            output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )
            ignore, target = torch.max( output, 1 )
            im_target = target.data.cpu().numpy()
            #print(im_target)
            
            #check number of clusters 
            Num_clusters = len(np.unique(im_target))
            if visualize:
                im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
                im_target_rgb = im_target_rgb.reshape( image.shape ).astype( np.uint8 )
                #im_target_rgb_res= cv2.resize(im_target_rgb, dim, interpolation = cv2.INTER_AREA)
                cv2.imshow( "output", im_target_rgb )
                cv2.waitKey(10)
                

            # superpixel refinement
            for i in range(len(l_inds)):
                labels_per_sp = im_target[ l_inds[ i ] ]
                unique_labels_per_sp = np.unique( labels_per_sp )
                hist = np.zeros( len(unique_labels_per_sp) )
                for j in range(len(hist)):
                    hist[ j ] = len( np.where( labels_per_sp == unique_labels_per_sp[ j ] )[ 0 ] )
                im_target[ l_inds[ i ] ] = unique_labels_per_sp[ np.argmax( hist ) ]
            target = torch.from_numpy( im_target )
            target = Variable( target )
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            if Num_clusters <= min_clusters:
                break
        
        
        #Show Segmented Image after total iterations or minimum clusters are reached 
        cv2.imshow("Segmented image", im_target_rgb)
        cv2.waitKey(10)
        #plt.imshow (im_target_rgb)
        return (im_target_rgb)

#Function initialising segmentation
def start_segmentation(image_array):
    segmented=[]
    count=0
    print("Starting segmentation")
    for i in range(len(image_array)):

        image=image_array[i][1]

        im=Segmentation(image)

        segmented.append(im)
        
    return segmented

#Function to compute similarity
def Similarity(segmented_images,segmented_query_img,image_array):
    similarity=[]

    for i in range(len(segmented_images)):
        compared=[]
        image=(segmented_images[i])
        sim = ssim(segmented_query_img, image,multichannel=True)
    #   score = adjusted_rand_score(np.asarray(image),np.asarray(segmented_query_img))
        #print (i,sim,"  ")
        compared.append(image_array[i][0])
        compared.append(sim)
        similarity.append(compared)
    return similarity


#Inputs , segmentation , query image and similarity computation calls 

dataFile="C:\\Users\\Gurpreet\\Desktop\\python\\IRTEX-Segmentation\\media\\cifar10"

loaded_data=Load_data(dataFile)
print("dataset Loaded")

#Call function to start segmentation
segmented_images = start_segmentation(loaded_data) 
print("segmentation done")

#Take Input from user for query Image (query path)
#Hard coded for now
path ="C:\\Users\\Gurpreet\\Desktop\\python\\IRTEX-Segmentation\\media\\cifar10\\airplane\\airplane_1013.png"
dim=(128,128)   
queryimage = cv2.imread(path, cv2.IMREAD_UNCHANGED)
queryimage = cv2.resize(queryimage, dim, interpolation = cv2.INTER_AREA)
segmented_query_img=Segmentation(queryimage)

# Compute similarity between segmented images and query image

similarity= Similarity (segmented_images,segmented_query_img,loaded_data)
similarity= sorted(similarity,key=lambda similarity: similarity[1],reverse=True)


# Display results based on similarity
print("Results based on similarity")
for i in range (len(similarity)):
    print("File : ",similarity[i][0],"  Similarity : ",similarity[i][1])
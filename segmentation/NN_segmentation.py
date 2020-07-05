from django.conf import settings

# %matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init
import sys
import os
import scipy.io as sio
import matplotlib.image as image

from PIL import Image
from os import path
from PIL import Image
from tqdm import tqdm
from torch import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from skimage import segmentation
from cv2 import cv2
from skimage.measure import compare_ssim as ssim
from sklearn.metrics.cluster import adjusted_rand_score
from skimage.segmentation import mark_boundaries
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

code = 'cifar_segment_all'
n_components = 5
pca = PCA(n_components=n_components)  # decided after empherical testing


# load input dataset CIFAR for Segmentation
def Load_data(dataFile):
    # print("load datafile started")
    image_array = []
    dim = (128, 128)
    for label in (os.listdir(dataFile)):
        count = 0
        for img in (os.listdir(os.path.join(dataFile, label))):
            count = count + 1
            img_cat = []

            img_read = cv2.imread(os.path.join(dataFile, label, img), cv2.IMREAD_UNCHANGED)
            img_read = cv2.resize(img_read, dim, interpolation=cv2.INTER_AREA)

            img_cat.append(img)
            img_cat.append(img_read)
            img_cat.append(label)
            image_array.append(img_cat)
            if (count == 500):
                break
    return image_array


# CNN model architecture (Conv + BatchNormalisation + Conv + BatchNormalisation)--> Initial Conv layer

class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.nChannel = 5
        self.nConv = 3
        self.conv1 = nn.Conv2d(input_dim, self.nChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()

        for i in range(self.nConv - 1):  # Adding extra Conv layers and BatchNormalisation to base architecture
            self.conv2.append(nn.Conv2d(self.nChannel, self.nChannel, kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(self.nChannel))
        self.conv3 = nn.Conv2d(self.nChannel, self.nChannel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(self.nChannel)

    # Function working on the provided inputs to NN

    def forward(self, x):  # x is the input shape
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(self.nConv - 1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x  # final output from network


# Function to perform segmentation
def segmentation_cifar(image):
    # hyperparameters initialisation
    nChannel = 5
    maxIter = 300
    min_clusters = 2
    lr = 0.1
    nConv = 3
    num_superpixels = 800
    compactness = 100
    visualize = 1

    # print("Segmentation started")

    # Input Image provided for Segmentation

    # cv2.imshow( "input", image )
    # cv2.waitKey(10)

    data = torch.from_numpy(np.array([image.transpose((2, 0, 1)).astype('float32') / 255.]))

    # slic to produce a ground truth to compare
    clusters = segmentation.slic(image, compactness=compactness, n_segments=num_superpixels)
    clusters = clusters.reshape(image.shape[0] * image.shape[1])

    # number of unique labels by combining the values for all labels generated
    unique_cluster = np.unique(clusters)
    l_inds = []
    for i in range(len(unique_cluster)):
        l_inds.append(np.where(clusters == unique_cluster[i])[0])

    # Training CNN model
    model = MyNet(data.size(1))  # data.size(1) is value 3
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    label_colours = np.random.randint(255, size=(100, 3))

    for _ in range(maxIter):
        # forwarding
        optimizer.zero_grad()
        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, nChannel)
        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        # print(im_target)

        # check number of clusters
        Num_clusters = len(np.unique(im_target))
        if visualize:
            im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
            im_target_rgb = im_target_rgb.reshape(image.shape).astype(np.uint8)
            # im_target_rgb_res= cv2.resize(im_target_rgb, dim, interpolation = cv2.INTER_AREA)
            # cv2.imshow( "output", im_target_rgb )
            # cv2.waitKey(10)

        # superpixel refinement
        for i in range(len(l_inds)):
            labels_per_sp = im_target[l_inds[i]]
            unique_labels_per_sp = np.unique(labels_per_sp)
            hist = np.zeros(len(unique_labels_per_sp))
            for j in range(len(hist)):
                hist[j] = len(np.where(labels_per_sp == unique_labels_per_sp[j])[0])
            im_target[l_inds[i]] = unique_labels_per_sp[np.argmax(hist)]
        target = torch.from_numpy(im_target)
        target = Variable(target)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if Num_clusters <= min_clusters:
            break

    # Show Segmented Image after total iterations or minimum clusters are reached
    # cv2.imshow("Segmented image", im_target_rgb)
    # cv2.waitKey(10)

    # PCA transformation of query image
    segmented_query_img = np.resize(im_target_rgb, (128, 128))
    segmented_query_img = np.array(segmented_query_img)
    segmented_query_img_norm = normalize(segmented_query_img)
    segmented_query_img = pca.fit_transform(segmented_query_img_norm)

    segmented_query_img = segmented_query_img.flatten()

    return segmented_query_img


# Function initialising segmentation
def start_segmentation(image_array):
    feature_list = []
    count = 0

    # path where the segmented masks got saved
    save_path = 'C:\\Users\\Gurpreet\\Desktop\\python\\IRTEX-Segmentation\\irtex-1.0\\segmentation\\segmentation_CNN\\segmented_mask_cifar\\'
    print("Starting segmentation")
    for i in range(len(image_array)):
        image = image_array[i][1]
        image_name = image_array[i][0]

        segmented_img = segmentation_cifar(image)  # segmentation of image

        #### Storing Mask for segmented image  #######
        im = segmented_img
        im = Image.fromarray(np.uint8(im)).convert('RGB')
        name, extension = path.splitext(image_name)
        im.save(save_path + name + '.png')  # Storing images as Pil Image in the file location

        ########## PCA ############
        segmented_img = np.resize(segmented_img, (128, 128))
        segmented_img = np.array(segmented_img)
        segmented_img_norm = normalize(segmented_img)
        segmented_img_pca = pca.fit_transform(segmented_img_norm)

        segmented_img_pca = segmented_img_pca.flatten()  # 1d array for the segmented image

        row = [image_name, segmented_img_pca, image_array[i][2]]
        feature_list.append(row)
        count = count + 1
        if (count == 10):
            break
    df = pd.DataFrame(feature_list, columns=["file_name", code, "label"])
    pd.to_pickle(df, code + '.pkl')


# Function to compute similarity
def get_similarity_segmentation_cifar(segmented_query_img):
    similarity = []

    feature__path = os.path.join(settings.BASE_DIR, 'segmentation/')
    # path = 'C:\\Users\\Gurpreet\Desktop\\python\\IRTEX-Segmentation\\irtex-1.0\\segmentation\\segmentation_CNN'
    df = pd.read_pickle(os.path.join(feature__path, '{}.pkl'.format(code)))

    file_name = df['file_name']
    features = df[code]
    labels = df['label']

    for iter in range(len(features)):
        # sim_ssim = ssim(segmented_query_img.reshape(128,n_components), features[i].reshape(128,n_components),multichannel=True)
        sim_ari = adjusted_rand_score(segmented_query_img, features[iter])

        # row = [file_name[iter], sim_ari, labels[iter]]
        row = {'name': file_name[iter], 'similarity': sim_ari, 'label': labels[iter],
               'url': '/media/cifar10/{}/{}'.format(labels[iter], file_name[iter])}
        similarity.append(row)

    return similarity


def get_similarity_segmentation_cifar_algorithm2(segmented_query_img, images):
    similarity = []

    feature__path = os.path.join(settings.BASE_DIR, 'segmentation/')
    # path = 'C:\\Users\\Gurpreet\Desktop\\python\\IRTEX-Segmentation\\irtex-1.0\\segmentation\\segmentation_CNN'
    df = pd.read_pickle(os.path.join(feature__path, '{}_subset.pkl'.format(code)))

    df = df[df['file_name'].isin(images)]

    file_name = df['file_name'].tolist()
    features = df[code].tolist()
    labels = df['label'].tolist()

    for iter in range(len(features)):
        # sim_ssim = ssim(segmented_query_img.reshape(128,n_components), features[i].reshape(128,n_components),multichannel=True)
        sim_ari = adjusted_rand_score(segmented_query_img, features[iter])

        # row = [file_name[iter], sim_ari, labels[iter]]
        row = {'name': file_name[iter], 'similarity': sim_ari, 'label': labels[iter],
               'url': '/media/cifar10/{}/{}'.format(labels[iter], file_name[iter])}
        similarity.append(row)

    return similarity

def getCifarLocalExplanations(query_image_path, retr_image_path):
    media_path = os.path.join(settings.BASE_DIR, 'media')
    store_path = os.path.join(media_path, 'seg')

    query_image_name = query_image_path.split('/')[-1]
    retr_image_name = retr_image_path.split('/')[-1]

    # retrieve mask for imageS
    query_image_mask = cv2.imread(os.path.join(media_path, 'masks/cifar10/{}.png'.format(query_image_name.split('.')[0])),
                                  cv2.IMREAD_UNCHANGED)
    query_image_mask = cv2.cvtColor(query_image_mask, cv2.COLOR_BGR2GRAY)

    retr_image_mask = cv2.imread(os.path.join(media_path, 'masks/cifar10/{}.png'.format(retr_image_name.split('.')[0])),
                                 cv2.IMREAD_UNCHANGED)
    retr_image_mask = cv2.cvtColor(retr_image_mask, cv2.COLOR_BGR2GRAY)

    clusters_query = unique_clusters(query_image_mask)
    clusters_retr = unique_clusters(retr_image_mask)

    ## Remove background from masks
    query_image_mask = separate_background(clusters_query, query_image_mask)
    retr_image_mask = separate_background(clusters_retr, retr_image_mask)

    ## Find unique clusters exceot background
    clusters_query = unique_clusters(query_image_mask)
    clusters_retr = unique_clusters(retr_image_mask)

    # if(len(clusters_query)!=0 and len(clusters_retr)!=0):
    region_sim_score = region_similarity(clusters_query, clusters_retr)
    df = pd.DataFrame(region_sim_score, columns=["query_region", "image_region", "similarity"])
    row = df.loc[df['similarity'].idxmax()]

    savepath_query_vis = store_img_with_boundary(query_image_path, store_path, query_image_name, row[0])
    savepath_retr_vis = store_img_with_boundary(retr_image_path, store_path, retr_image_name, row[1])

    explanation = {}
    explanation['text'] = ['The regions of query image and result image are compared. '
                           'The similarity achieved is {}%. The most similar regions are '
                           'marked with a boundary.'.format(np.round(row['similarity'] * 100, 3))]

    explanation['images'] = [{'name': 'Query Image', 'url': '/media/seg/{}'.format(savepath_query_vis)},
                             {'name': 'Result Image',
                              'url': '/media/seg/{}'.format(savepath_retr_vis)}]

    return explanation


def unique_clusters(image_mask):
    unique = np.unique(image_mask)
    #     print(unique)
    clusters = []
    for i in range(len(unique) - 1):
        cluster = image_mask.copy()
        cluster[image_mask != unique[i + 1]] = 0
        clusters.append(cluster)

    return clusters


def separate_background(clusters, image):
    len(clusters)
    max = 0
    for i in range(len(clusters)):
        size = clusters[i][np.nonzero(clusters[i])].size
        if (size > max):
            max = size
            background = clusters[i]

    np.unique(background)

    new_image = image.copy()
    new_image[new_image == background] = 0

    return new_image


def region_similarity(clusters_query, clusters_retr):
    score = []
    for reg in clusters_retr:
        for region in clusters_query:
            ari = adjusted_rand_score(region.flatten(), reg.flatten())
            ari = (ari + 1) / 2
            row = [region, reg, ari]
            score.append(row)
    return score


def store_img_with_boundary(image_path, store_path, image_name, row):
    image_org = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image_org = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)
    image_org = cv2.resize(image_org, (128, 128))
    image_org = mark_boundaries(image_org, row, mode='thick')

    image_org = cv2.convertScaleAbs(image_org, alpha=(255.0))
    image_org = cv2.cvtColor(image_org, cv2.COLOR_RGB2BGR)

    image_name, ex = image_name.split('.')

    savepath_vis = store_path + '/' + image_name + '_vis.jpg'

    cv2.imwrite(savepath_vis, image_org)

    return image_name + '_vis.jpg'

# Inputs , segmentation , query image and similarity computation calls

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN Segmentation on CIFAR')

    parser.add_argument('--path', help='path to query image', required=True)
    args = parser.parse_args()
    query_path = args.path

    # #Segment CIFAR dataset and save masks and feature vector in pickle 
    # dataFile="C:\\Users\\Gurpreet\\Desktop\\python\\IRTEX-Segmentation\\media\\cifar10"
    # loaded_data=Load_data(dataFile)
    # print("dataset Loaded")
    # segmented_images = start_segmentation(loaded_data) 
    # print("segmentation done")

    query_img = cv2.imread(query_path, cv2.IMREAD_UNCHANGED)
    query_img = cv2.resize(query_img, (128, 128), interpolation=cv2.INTER_AREA)
    segmented_query_img = segmentation_cifar(query_img)

    # compute similarity of query image and CIFAR images
    similarity = get_similarity_segmentation_cifar(segmented_query_img)

    # sorting based on ARI score
    similarity = sorted(similarity, key=lambda similarity: similarity[1], reverse=True)

    # # Display results based on similarity
    # print("Results based on similarity")
    # for i in range (len(similarity)):
    #     print("File : ",similarity[i][0],"  Similarity : ",similarity[i][1],"   Label : ",similarity[i][2])

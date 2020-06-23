from django.conf import settings

import mxnet as mx
import numpy as np
import gluoncv
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.utils.viz import get_color_pallete
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import pandas as pd
import os
from os import path
import argparse
# import tensorflow as tf
import cv2
from skimage.measure import compare_ssim as ssim
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import scipy.io as sio
import matplotlib.image as image
import pandas as pd
import matplotlib.pyplot as plt
from mxnet import image
from mxnet.gluon.data.vision import transforms
import csv

# using cpu
ctx = mx.cpu(0)


class semantic_segmentation_pascal:
    def __init__(self):
        self.model = gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True)

    def get_model(self):
        return self.model


code = 'pascal_segment_final'
model = semantic_segmentation_pascal().get_model()
pca = PCA(n_components=2)

# pathe where masked images are stored before PCA
save_path = 'C:\\Users\\Gurpreet\\Desktop\\python\\IRTEX-Segmentation\\irtex-1.0\\segmentation\\segmentation_CNN\\segmented_image_pascal_final\\'


def extract_features_pascal(img):
    img = test_transform(img, ctx)
    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
    mask = get_color_pallete(predict, 'pascal_voc')
    mask = mask.resize((128, 128))
    mask_array = np.asarray(mask).flatten()

    mask_query_img = np.resize(mask, (128, 128))
    mask_query_img = np.array(mask_query_img)
    mask_query_img_norm = normalize(mask_query_img)
    mask_query_img_pca = pca.fit_transform(mask_query_img_norm)

    segmented_query_img_pca = mask_query_img_pca.flatten()

    return segmented_query_img_pca


def segment_pascal_dataset(filename):
    feature_list = []
    # dataFile = os.getcwd()
    # print(dataFile)
    # save_path = os.path.join(dataFile, '/segmented_mask_pascal') #location to save segmented images 
    # print(save_path)

    for label in (os.listdir(filename)):
        for img_file in (os.listdir(os.path.join(filename, label))):
            img = image.imread(os.path.join(filename, label, img_file))  # <class 'mxnet.ndarray.ndarray.NDArray'>
            mask, mask_array = extract_features_pascal(img)
            name, extension = path.splitext(img_file)
            mask.save(save_path + name + '.jpg')

            ########## PCA ############
            mask = np.resize(mask, (128, 128))
            mask = np.array(mask)
            mask_norm = normalize(mask)
            mask_pca = pca.fit_transform(mask_norm)

            segmented_img_pca = mask_pca.flatten()  # 1d array for the segmented image

            row = [img_file, segmented_img_pca, label]
            feature_list.append(row)
            # print(feature_list)
            # break

    df = pd.DataFrame(feature_list, columns=["file_name", code, "label"])
    pd.to_pickle(df, code + '.pkl')


def get_similarity_segmentation_pascal(segmented_query_img_pca):
    count = 0
    similarity = []
    feature__path = os.path.join(settings.BASE_DIR, 'segmentation/')

    # path = 'C:\\Users\\Gurpreet\Desktop\\python\\IRTEX-Segmentation\\irtex-1.0\\segmentation\\segmentation_CNN'
    df = pd.read_pickle(os.path.join(feature__path, '{}.pkl'.format(code)))

    file_name = df['file_name']
    features = df[code]
    labels = df['label']

    for iter in range(len(features)):
        sim_ssim = ssim(segmented_query_img_pca.reshape(128, 2), features[iter].reshape(128, 2), multichannel=True)
        # sim_ari = adjusted_rand_score(segmented_query_img,features[i])

        ##Normalise ssim similarity between 0 and 1
        sim_ssim = ((1 + sim_ssim)/2)

        for label in labels[iter]:
            row = {'name': file_name[iter], 'similarity': sim_ssim, 'label': label,
                   'url': '/media/voc/{}/{}'.format(label, file_name[iter])}
            similarity.append(row)

    return similarity


def get_similarity_segmentation_pascal_algorithm2(segmented_query_img_pca, images):
    count = 0
    similarity = []
    feature__path = os.path.join(settings.BASE_DIR, 'segmentation/')

    # path = 'C:\\Users\\Gurpreet\Desktop\\python\\IRTEX-Segmentation\\irtex-1.0\\segmentation\\segmentation_CNN'
    df = pd.read_pickle(os.path.join(feature__path, '{}.pkl'.format(code)))

    df = df[df['file_name'].isin(images)]

    file_name = df['file_name'].tolist()
    features = df[code].tolist()
    labels = df['label'].tolist()

    for iter in range(len(features)):
        sim_ssim = ssim(segmented_query_img_pca.reshape(128, 2), features[iter].reshape(128, 2), multichannel=True)
        # sim_ari = adjusted_rand_score(segmented_query_img,features[i])

        ##Normalise ssim similarity between 0 and 1
        sim_ssim = ((1 + sim_ssim)/2)

        for label in labels[iter]:
            row = {'name': file_name[iter], 'similarity': sim_ssim, 'label': label,
                   'url': '/media/voc/{}/{}'.format(label, file_name[iter])}
            similarity.append(row)

    return similarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FCN_Resnet Segmentation on Pascal dataset')

    parser.add_argument('--path', help='path to query image', required=True)
    args = parser.parse_args()
    query_path = args.path

    # #Segment whole pascal dataset and save masks and feature vector in pickle 
    # filename='C:\\Users\\Gurpreet\\Desktop\\python\\IRTEX-Segmentation\\pascal'
    # segment_pascal_dataset(filename)

    img = image.imread(query_path)
    segmented_query_img_pca = extract_features_pascal(img)

    similarity = get_similarity_segmentation_pascal(segmented_query_img_pca)

    ##SOrted based on SSIM
    similarity = sorted(similarity, key=lambda similarity: similarity[1], reverse=True)
    # Display results based on similarity
    print("Results based on similarity")
    for i in range(len(similarity)):
        print("File : ", similarity[i][0],
              "  Similarity SSIM : ", similarity[i][1], "  Label : ", similarity[i][2])
        # n = args.check
        # stop_at=2

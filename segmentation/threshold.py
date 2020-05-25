#from django.conf import settings

import numpy as np
import os
import cv2
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import segmentation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score
from skimage.color import rgb2gray

def takefirst(elem):
    return elem[0]

def segmentation(image):
    gray = rgb2gray(img_array)
    gray_r = gray.reshape(gray.shape[0] * gray.shape[1])
    for i in range(gray_r.shape[0]):
        if gray_r[i] > gray_r.mean():
            gray_r[i] = 3
        elif gray_r[i] > 0.5:
            gray_r[i] = 2
        elif gray_r[i] > 0.25:
            gray_r[i] = 1
        else:
            gray_r[i] = 0
    gray = gray_r.reshape(gray.shape[0], gray.shape[1])

    return gray


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SLIC Segmentation')

    parser.add_argument('--path', help='path to query image', required=True)
    parser.add_argument('--check', help='Number of images to check with per class', default=2)
    args = parser.parse_args()
    path = args.path
    n = args.check

    dataFile = os.getcwd()
    path_ds = os.path.join(dataFile, '../media/cifar10')
    images = []
    labels = []
    target_names = []
    i = 0

    img_seg = []

    for label in tqdm(os.listdir(path_ds)):
        target_names.append(label)
        i = 0
        for img in tqdm(os.listdir(os.path.join(path_ds, label))):
            img_array = cv2.imread(os.path.join(path_ds, label, img), cv2.IMREAD_UNCHANGED)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img_array = cv2.resize(img_array, (256, 256))
            labels.append(label)

            segments = segmentation(img_array)

            img_seg.append(segments)

            i +=1
            if i == n:
                break

    imgtest = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    imgtest = cv2.cvtColor(imgtest, cv2.COLOR_BGR2RGB)
    imgtest = cv2.resize(imgtest, (256, 256))

    segmentstest = segmentation(imgtest)

    result = []
    for i in range(len(img_seg)):
        q_sim = adjusted_rand_score(img_seg[i].flatten(), segmentstest.flatten())
        result.append([q_sim, labels[i]])

    result.sort(key=takefirst, reverse=True)

    for i in result:
        print('Similarity:', i[0], 'Label:', i[1])
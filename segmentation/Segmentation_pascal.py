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
from skimage.segmentation import mark_boundaries
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


code = 'pascal_segment_all'
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

        # for label in labels[iter]:
        row = {'name': file_name[iter], 'similarity': sim_ssim, 'label': labels[iter],
               'url': '/media/voc/{}/{}'.format(labels[iter][0], file_name[iter])}
        similarity.append(row)

    return similarity


def get_similarity_segmentation_pascal_algorithm2(segmented_query_img_pca, images):
    count = 0
    similarity = []
    feature__path = os.path.join(settings.BASE_DIR, 'segmentation/')

    # path = 'C:\\Users\\Gurpreet\Desktop\\python\\IRTEX-Segmentation\\irtex-1.0\\segmentation\\segmentation_CNN'
    df = pd.read_pickle(os.path.join(feature__path, '{}_subset.pkl'.format(code)))

    df = df[df['file_name'].isin(images)]

    file_name = df['file_name'].tolist()
    features = df[code].tolist()
    labels = df['label'].tolist()

    for iter in range(len(features)):
        sim_ssim = ssim(segmented_query_img_pca.reshape(128, 2), features[iter].reshape(128, 2), multichannel=True)
        # sim_ari = adjusted_rand_score(segmented_query_img,features[i])

        ##Normalise ssim similarity between 0 and 1
        sim_ssim = ((1 + sim_ssim)/2)

        # for label in labels[iter]:
        row = {'name': file_name[iter], 'similarity': sim_ssim, 'label': labels[iter],
               'url': '/media/voc/{}/{}'.format(labels[iter][0], file_name[iter])}
        similarity.append(row)

    return similarity


def getLocalExplanation(query_image_path, retr_image_path):
    media_path = os.path.join(settings.BASE_DIR, 'media')
    store_path = os.path.join(media_path, 'seg')

    query_image_name = query_image_path.split('/')[-1]
    retr_image_name = retr_image_path.split('/')[-1]

    query_image_mask = cv2.imread(os.path.join(media_path, 'masks/voc/{}.png'.format(query_image_name.split('.')[0])),
                                  cv2.IMREAD_UNCHANGED)
    query_image_mask = cv2.cvtColor(query_image_mask, cv2.COLOR_BGR2GRAY)

    retr_image_mask = cv2.imread(os.path.join(media_path, 'masks/voc/{}.png'.format(retr_image_name.split('.')[0])),
                                 cv2.IMREAD_UNCHANGED)
    retr_image_mask = cv2.cvtColor(retr_image_mask, cv2.COLOR_BGR2GRAY)

    clusters_query = unique_clusters(query_image_mask)
    clusters_retr = unique_clusters(retr_image_mask)

    if len(clusters_query) == 0:
        query_image_mask = np.array((np.array(query_image_mask))[0])
        query_image_mask = np.reshape(query_image_mask, (128, 128))
        clusters_query.append(query_image_mask)

    if len(clusters_retr) == 0:
        retr_image_mask = np.array((np.array(retr_image_mask))[0])
        retr_image_mask = np.reshape(retr_image_mask, (128, 128))
        clusters_retr.append(retr_image_mask)

    region_sim_score = region_similarity(clusters_query, clusters_retr)
    df = pd.DataFrame(region_sim_score, columns=["query_region", "image_region", "similarity"])
    row = df.loc[df['similarity'].idxmax()]

    savepath_query_vis = store_img_with_boundary(query_image_path, store_path, query_image_name, row[0])
    savepath_retr_vis = store_img_with_boundary(retr_image_path, store_path, retr_image_name, row[1])

    explanation = {}
    explanation['text'] = ['The objects present in the foreground of query image and result image are compared. '
                           'The similarity achieved is {}%. The most similar regions are marked '
                           'with a boundary.'.format(np.round(row['similarity'] * 100, 3))]

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


#     plt.imshow(image_org)


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

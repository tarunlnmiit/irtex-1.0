# Extract features from VGG16 CNN mocel
# Ignore the django.core.exceptions.ImproperlyConfigured error about settings seen when running the main method directly.

import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
import cv2
from skimage.transform import resize
import csv
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import argparse
import traceback

# idea one: compare feature vector only
# idea two: compare logits
# idea three: compare softmax

image_size = 224  #must be at least 224 for imagenet weights with last top layer


class VGG16FeatureExtractor(object):
    def __init__(self):
        # Include top layer but do not use any activation function
        #self.model = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=True, weights='imagenet',
        #classifier_activation=None)

        # Features only
        self.model = tf.keras.applications.VGG16(input_shape=(image_size,image_size, 3), include_top=False,
        weights='imagenet')

    def get_model(self):
        return  self.model


model = VGG16FeatureExtractor().get_model();
code = 'vgg'


def extract_feature_vgg(query_image_path):
    img = cv2.imread(query_image_path)
    img = resize(img, (image_size, image_size))
    img = tf.expand_dims(img, 0)
    #img = img/255.0
    feature = model(img)
    #sm = tf.nn.softmax(feature)
    return feature.numpy().reshape(-1)


def get_similarity_vgg(query,_path=''):
    query = query.reshape(1, -1)

    features_path = code+'.pkl'
    if _path is not '':
        features_path = os.path.join(_path, 'vgg16_imagenet_features', code + '.pkl')

    df = pd.read_pickle(features_path)

    file_name = df['file_name']
    features = df[code].tolist()
    labels = df['label']

    q_sim = cosine_similarity(features, query)
    json_qsim = [{'name': file_name[i], 'similarity': str(q_sim[i][0]), 'label': labels[i],
                  'url': '/media/cifar10/{}/{}'.format(labels[i], file_name[i])} for i in range(len(q_sim))]

    return json_qsim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VGG16 Extractor with ImageNet weights')
    parser.add_argument('--stop_at', help='number of items per folder to extract, defaults to all')

    args = parser.parse_args()
    stop_at = args.stop_at
    if not stop_at:
        stop_at = 10

    path = '../media/cifar10'
    feature_list = []
    count = 0

    for label in tqdm(os.listdir(path)):
        for img_file in tqdm(os.listdir(os.path.join(path, label))):
            if count == stop_at:
                count=0
                break
            row = [img_file,extract_feature_vgg(os.path.join(path, label, img_file)), label]
            feature_list.append(row)
            count += 1

    df = pd.DataFrame(feature_list, columns=["file_name", code, "label"])
    pd.to_pickle(df, code+'.pkl')
    print('done')



import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
import cv2
from skimage.transform import resize
import csv
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings
import  pandas as pd

#idea one: compare feature vector only
#idea two: compare logits
#idea three: compare softmax

image_size=32
class VGG16FeatureExtractor(object):
    def __init__(self):
        #self.model = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=True, weights='imagenet', classifier_activation=None)
        self.model = tf.keras.applications.VGG16(input_shape=(image_size,image_size, 3), include_top=False, weights='imagenet')


    def get_model(self):
        return  self.model

model=VGG16FeatureExtractor().get_model();


def extract_feature_vgg(query_image_path):
    img = cv2.imread(query_image_path)
    img = resize(img, (image_size, image_size))
    img = tf.expand_dims(img, 0)
    feature = model(img)
    sm = tf.nn.softmax(feature)
    return sm.numpy().reshape(-1)


def get_similarity_vgg(query):
    query = query.reshape(1, -1)

    df = pd.read_csv(os.path.join(settings.BASE_DIR, 'vgg16_imagenet_features','vgg16.csv'))
    #df = pd.read_csv('vgg16.csv')
    file_name = df['file_name']
    vgg = df['vgg']
    vgg = [[float(i) for i in elem.strip('[] ').split()] for elem in vgg]
    labels = df['label']

    q_sim = cosine_similarity(vgg, query)
    json_qsim = [{'name': file_name[i], 'similarity': q_sim[i][0], 'label': labels[i],
                  'url': '/media/cifar10/{}/{}'.format(labels[i], file_name[i])} for i in range(len(q_sim))]

    return json_qsim


if __name__ == "__main__":

    model=VGG16FeatureExtractor().get_model()
    path='../media/cifar10'

    feature_list = []
    feature_list.append(["file_name", "vgg", "label"])
    count=0
    class_count=0
    for label in tqdm(os.listdir(path)):
        for img_file in tqdm(os.listdir(os.path.join(path, label))):
            if(count==100):
                count=0 #reduce number for testing
                break
            img = cv2.imread(os.path.join(path, label, img_file))
            img = resize(img, (image_size, image_size))
            img = tf.expand_dims(img, 0)
            feature = model(img)
            sm = feature #tf.nn.softmax(feature)
            row = [img_file, sm.numpy().reshape(-1), label]
            feature_list.append(row)
            count += 1

    with open('vgg16.csv', 'wt') as file:
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        writer.writerows(feature_list)
        file.close()



from django.conf import settings

import numpy as np
import os
import cv2
import mahotas
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import json
from sklearn.decomposition import PCA
from json import JSONEncoder

# For JSON Encoding
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class RBSDescriptor:
    def __init__(self, dataset):
        self.dataset = dataset
        feature_csv_path = os.path.join(settings.BASE_DIR, 'region_based_descriptor')
        if self.dataset == 'cifar':
            df = pd.read_pickle(os.path.join(feature_csv_path, 'moments_cifar_pca.pkl'))
            #To perform PCA on query image
            df_temp = pd.read_pickle(os.path.join(feature_csv_path, 'moments_cifar.pkl'))
            n_comp = 15
        if self.dataset == 'pascal':
            df = pd.read_pickle(os.path.join(feature_csv_path, 'moments_pascal_pca_updated.pkl'))
            # To perform PCA on query image
            df_temp = pd.read_pickle(os.path.join(feature_csv_path, 'moments_pascal_updated.pkl'))
            n_comp = 20
        self.file_name = df['file_name']
        self.moments = df['moments']
        self.moments = self.moments.tolist()
        self.labels = df['label']

        # To perform PCA on query image
        self.og_moments = df_temp['moments']
        self.og_moments = self.og_moments.tolist()
        self.pca = PCA(n_components=n_comp)
        self.pca.fit_transform(self.og_moments)

# Calculate similarity between the query image and extracted feature and converting it into json format
    def similarity(self, query):
        q_sim = cosine_similarity(self.moments, query)
        if self.dataset == 'cifar':
            json_qsim = [{'name': self.file_name[i], 'similarity': q_sim[i][0], 'label': self.labels[i],
                          'url': '/media/cifar10/{}/{}'.format(self.labels[i], self.file_name[i])} for i in range(len(q_sim))]
            # json_qsim = json.loads(json.dumps(json_qsim, cls=NumpyArrayEncoder))
        if self.dataset == 'pascal':
            json_qsim = [{'name': self.file_name[i], 'similarity': q_sim[i][0], 'label': ', '.join(self.labels[i]),
                          'url': '/media/voc/{}/{}'.format(self.labels[i][0], self.file_name[i])} for i in
                         range(len(q_sim))]

        return json_qsim

# Calculating the zernike moments of query image
    def zernike_moments(self, image):
        if self.dataset =='cifar':
            radius = 16
            degree = 16
        if self.dataset =='pascal':
            radius = 128
            degree = 20

        query_moment = mahotas.features.zernike_moments(image, radius, degree=degree).reshape(1, -1)
        #To perform PCA on query image
        query_moment = self.pca.transform(query_moment)
        return query_moment

# For textual explanation
    def textual_explanation(self):
        return 'text'

# Pre-processing the query image so as to match extracted features
    def image_preprocessing(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.dataset == 'cifar':
            img = cv2.resize(img, (32, 32))
        if self.dataset == 'pascal':
            img = cv2.resize(img, (256, 256))
        # img = img + np.random.normal(scale=2, size=img.shape)
        return img


#rbsd = RBSDescriptor()
#img_array = cv2.imread("C:\\Users\\Administrator\\Desktop\\libin_ovgu\\SoSe20\\IRTEX Project\\cifar10\\dog\\image_104.png", cv2.IMREAD_UNCHANGED)
#img_array = rbsd.image_preprocessing(img_array)

#q_moment = rbsd.zernike_moments(img_array)
#sim = rbsd.similarity(q_moment)
from django.conf import settings

import numpy as np
import os
import cv2
import mahotas
import pandas as pd
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage.measurements import center_of_mass
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
        if self.dataset =='cifar':
            self.radius = 16
            self.degree = 16
        if self.dataset =='pascal':
            self.radius = 128
            self.degree = 20

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

        self.df = df
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
        q_sim = [[(sim[0] + 1) / 2] for sim in q_sim]
        if self.dataset == 'cifar':
            json_qsim = [{'name': self.file_name[i], 'similarity': q_sim[i][0], 'label': self.labels[i],
                          'url': '/media/cifar10/{}/{}'.format(self.labels[i], self.file_name[i])} for i in range(len(q_sim))]
            # json_qsim = json.loads(json.dumps(json_qsim, cls=NumpyArrayEncoder))
        if self.dataset == 'pascal':
            json_qsim = [{'name': self.file_name[i], 'similarity': q_sim[i][0], 'label': self.labels[i],
                          'url': '/media/voc/{}/{}'.format(self.labels[i][0], self.file_name[i])} for i in
                         range(len(q_sim))]

        return json_qsim

    def similarity_algorithm2(self, images, query):
        df_algorithm2 = self.df[self.df['file_name'].isin(images)]
        file_name_algo2 = df_algorithm2['file_name'].tolist()
        labels_algo2 = df_algorithm2['label'].tolist()
        q_sim = cosine_similarity(df_algorithm2['moments'].tolist(), query)
        q_sim = [[(sim[0] + 1) / 2] for sim in q_sim]
        if self.dataset == 'cifar':
            json_qsim = [{'name': file_name_algo2[i], 'similarity': q_sim[i][0], 'label': labels_algo2[i],
                          'url': '/media/cifar10/{}/{}'.format(labels_algo2[i], file_name_algo2[i])} for i in range(len(q_sim))]
            # json_qsim = json.loads(json.dumps(json_qsim, cls=NumpyArrayEncoder))
        if self.dataset == 'pascal':
            json_qsim = [{'name': file_name_algo2[i], 'similarity': q_sim[i][0], 'label': labels_algo2[i],
                          'url': '/media/voc/{}/{}'.format(labels_algo2[i][0], file_name_algo2[i])} for i in
                         range(len(q_sim))]

        return json_qsim

# Calculating the zernike moments of query image
    def zernike_moments(self, image):
        query_moment = mahotas.features.zernike_moments(image, self.radius, degree=self.degree).reshape(1, -1)
        #To perform PCA on query image
        query_moment = self.pca.transform(query_moment)
        return query_moment

# For Textual and Visual Explanation
    def explanation(self, query_path, image_path):
        media_path = os.path.join(settings.BASE_DIR, 'media')
        store_path = os.path.join(media_path, 'rbsd')

        query_image_name = query_path.split('/')[-1]
        retr_image_name = image_path.split('/')[-1]

        img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        query_array = cv2.imread(query_path, cv2.IMREAD_UNCHANGED)

        if self.dataset == 'pascal':
            img_array = cv2.resize(img_array, (256, 256))
            query_array = cv2.resize(query_array, (256, 256))

        img_temp = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        query_temp = cv2.cvtColor(query_array, cv2.COLOR_BGR2GRAY)

        x_img, y_img = center_of_mass(img_temp)
        x_query, y_query = center_of_mass(query_temp)
        x_img = int(x_img)
        y_img = int(y_img)
        x_query = int(x_query)
        y_query = int(y_query)

        # draw filled circle in white on black background as mask
        mask_img = np.zeros_like(img_array)
        mask_query = np.zeros_like(query_array)
        mask_img = cv2.circle(mask_img, (x_img, y_img), self.radius, (255, 255, 255), -1)
        mask_query = cv2.circle(mask_query, (x_query, y_query), self.radius, (255, 255, 255), -1)

        # apply mask to image
        result_img = cv2.bitwise_and(img_array, mask_img)
        result_query = cv2.bitwise_and(query_array, mask_query)

        # save results
        cv2.imwrite(os.path.join(store_path, '{}_rbsd.png'.format(retr_image_name.split('.')[0])), result_img)
        cv2.imwrite(os.path.join(store_path, '{}_rbsd.png'.format(query_image_name.split('.')[0])), result_query)

        list_images = []
        list_images.append(os.path.basename(image_path))
        list_images.append(os.path.basename(query_path))

        df_ = self.df[self.df['file_name'].isin(list_images)]
        moments = df_['moments'].tolist()
        q_sim = cosine_similarity(moments[0].reshape(1, -1), moments[1].reshape(1, -1))
        q_sim = [[(sim[0] + 1) / 2] for sim in q_sim]

        score = q_sim[0]
        score = score[0] * 100
        score = round(score, 2)
        text = 'Below regions were compared to get the similarity score. The similarity between these two region is ' + str(score) + '%'

        # textual = []
        # textual.append(text)

        # visual = []
        # visual.append('img_rbsd.png')
        # visual.append('query_rbsd.png')

        explanation = {}
        explanation['text'] = [text]

        explanation['images'] = [{'name': 'Query Image', 'url': '/media/rbsd/{}_rbsd.png'.format(query_image_name.split('.')[0])},
                             {'name': 'Result Image',
                              'url': '/media/rbsd/{}_rbsd.png'.format(retr_image_name.split('.')[0])}]

        return explanation

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
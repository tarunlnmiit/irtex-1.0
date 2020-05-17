import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import mahotas
import csv
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class RBSDescriptor:
    def __init__(self):
        self.radius = 16
        df = pd.read_csv('moments.csv')
        self.moments = df['moments']
        self.moments = [[float(i) for i in elem.strip('[] ').split()] for elem in self.moments]
        self.labels = df['label']
        self.file_name = df['file_name']

    def similarity(self, query):
        q_sim = []
        #i = 0
        #for moment in self.moments:
            #moment = np.reshape(moment, (1, -1))
            #sim = cosine_similarity(moment, query)
            #if sim > 0.90:
            #    sim_temp = [self.file_name[i], sim, self.labels[i]]
            #    q_sim.append(sim_temp)
            #i += 1
        q_sim = cosine_similarity(self.moments, query)

        return q_sim

    def zernike_moments(self, image):
        return mahotas.features.zernike_moments(image, self.radius).reshape(1, -1)

    def textual_explanation(self):
        return 'text'

    def image_preprocessing(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, (256, 256))
        # img = img + np.random.normal(scale=2, size=img.shape)
        return img


#rbsd = RBSDescriptor()
#img_array = cv2.imread("C:\\Users\\Administrator\\Desktop\\libin_ovgu\\SoSe20\\IRTEX Project\\cifar10\\dog\\image_104.png", cv2.IMREAD_UNCHANGED)
#img_array = rbsd.image_preprocessing(img_array)

#q_moment = rbsd.zernike_moments(img_array)
#sim = rbsd.similarity(q_moment)
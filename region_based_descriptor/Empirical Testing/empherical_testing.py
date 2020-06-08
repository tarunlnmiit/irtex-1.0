import numpy as np
import argparse
import os
import cv2
from tqdm import tqdm
import mahotas
import pandas as pd
import csv
import RBSDescriptor as rb
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
from gluoncv import data, utils
from sklearn.decomposition import PCA
from lpproj import LocalityPreservingProjection

class ZernikeMoments:
    def __init__(self, radius):
        # store the size of the radius that will be
        # used when computing moments
        self.radius = radius

    def describe(self, image):
        # return the Zernike moments for the image
        return mahotas.features.zernike_moments(image, self.radius, degree=12)


def takefirst(elem):
    return elem[0]


test_on = 'pascal'


if test_on == 'cifar':
    feature_csv_path = os.getcwd()
    # feature_csv_path = os.path.join(settings.BASE_DIR, 'region_based_descriptor')
    df = pd.read_pickle(os.path.join(feature_csv_path, 'moments_8_500.pkl'))
    moments_8 = df['moments']
    moments_8 = moments_8.tolist()

    rbsd = rb.RBSDescriptor()
    zernike_toy = rbsd.moments
    labels_toy = rbsd.labels
    print(zernike_toy[0].shape)

    df = pd.read_pickle(os.path.join(feature_csv_path, 'moments_12_500.pkl'))
    moments_12 = df['moments']
    moments_12 = moments_12.tolist()

    df = pd.read_pickle(os.path.join(feature_csv_path, 'moments_16_500.pkl'))
    moments_16 = df['moments']
    moments_16 = moments_16.tolist()

    df = pd.read_pickle(os.path.join(feature_csv_path, 'moments_20_500.pkl'))
    moments_20 = df['moments']
    moments_20 = moments_20.tolist()
    print('On Cifar')
else:
    feature_csv_path = os.getcwd()
    # feature_csv_path = os.path.join(settings.BASE_DIR, 'region_based_descriptor')
    df = pd.read_pickle(os.path.join(feature_csv_path, 'moments_pascal_8_subset.pkl'))
    moments_8 = df['moments']
    moments_8 = moments_8.tolist()

    df = pd.read_pickle(os.path.join(feature_csv_path, 'moments_pascal_12_subset.pkl'))
    moments_12 = df['moments']
    moments_12 = moments_12.tolist()

    df = pd.read_pickle(os.path.join(feature_csv_path, 'moments_pascal_16_subset.pkl'))
    moments_16 = df['moments']
    moments_16 = moments_16.tolist()

    df = pd.read_pickle(os.path.join(feature_csv_path, 'moments_pascal_20_subset.pkl'))
    moments_20 = df['moments']
    moments_20 = moments_20.tolist()
    labels_toy = df['label']
    print('On Pascal')

#dataFile = "C:\\Users\\Administrator\\Desktop\\libin_ovgu\\SoSe20\\IRTEX Project"
#path = os.path.join(dataFile, 'Pascal')

reduced_zm = []
#components = [3, 5, 7, 9, 10, 15]
#components = [3, 9, 15, 20, 25, 30]
components = [20]
for n_comp in components:
    pca = PCA(n_components=n_comp)

    principalComponents = pca.fit_transform(moments_20)
    #print(principalComponents)
    #zernike_toy = principalComponents
    all = []
#    all.append(moments_8)
#    all.append(moments_12)
    #all.append(zernike_toy)
#    all.append(moments_16)
    all.append(moments_20)
    all.append(principalComponents)
    #lpp = LocalityPreservingProjection(n_components=3)
    #lpp_dim = lpp.fit_transform(zernike_toy)
    #zernike_toy = lpp_dim

    topk = [100, 200, 300, 400, 500]
    #topk = [5, 10, 15, 20, 25]

    for moments in all:
        query = moments[0].reshape(1, -1)

        if test_on =='cifar':
         query_label = labels_toy[0]
        else:
            #query_label = eval(labels_toy[0])
            query_label = labels_toy[0]
        print(query_label)
        precision_cs = []
        precision_l1 = []

        total_len = 0

        if test_on =='cifar':
            total_len = 500 #5011  # 4670 #4695 #4729 #4729 #4657 #4727 #4664 #4708 #4689 #4679
        else:
            for i in labels_toy:
                for l in query_label:
                    if l in i: #eval(i):
                        total_len += 1
                        break



        print(total_len)
        # total_len = 6
        recall_cs = []
        recall_l1 = []

        f1_measure_cs = []
        f1_measure_l1 = []

        for k in topk:

            sim_t = []
            sim_l1 = []
            m_to_s = []
            sim_t = cosine_similarity(moments, query)
            sim_l1 = manhattan_distances(moments,query)
            for d in sim_l1:
                sim = 1 / (1 + d)
                m_to_s.append(sim)
            sim_l1 = m_to_s
            #print(sim_l1)
            result = []
            for i,j in zip(sim_t, labels_toy):
                result.append([i,j])

            result_l1 = []
            for i,j in zip(sim_l1, labels_toy):
                result_l1.append([i,j])

            result.sort(key=takefirst, reverse=True)
            result_l1.sort(key=takefirst, reverse=True)

            count = 0
            for i in result[:k]:
                #print('Similarity: {}, Toy Dataset Image: {}'.format(i[0][0], i[1]))
            #    print(i[1])

                if test_on == 'cifar':
                    if query_label in i[1]:
                            count +=1
                else:
                    for l in query_label:
                        if l in i[1]:#eval(i[1]):
                            count += 1
                            break


            print("Cosine Similarity")
            print(count)

            p = count/k
            precision_cs.append(p)
            r = count/total_len
            recall_cs.append(r)
            f1 = 2*p*r / (p + r)
            f1_measure_cs.append(f1)

            count = 0
            for i in result_l1[:k]:
                #print('Similarity: {}, Toy Dataset Image: {}'.format(i[0][0], i[1]))
                if test_on =='cifar':
                    if query_label in i[1]:
                        count +=1
                else:
                    for l in query_label:
                        if l in i[1]: #eval(i[1]):
                            count += 1
                            break


            print("L1 Distance")
            print(count)
            p = count/k
            precision_l1.append(p)
            r = count/total_len
            recall_l1.append(r)
            f1 = 2*p*r / (p + r)
            f1_measure_l1.append(f1)


        print(precision_cs)
        print(precision_l1)
        print(recall_cs)
        print(recall_l1)
        print(f1_measure_cs)
        print(f1_measure_l1)

        plt.figure("F1 Measure")
        plt.xlabel('Top K')
        plt.ylabel('F1 Measure')
        plt.plot(topk, f1_measure_cs, marker='.')
        plt.plot(topk, f1_measure_l1, marker='.')

#print('N_components:', n_comp)
plt.legend(["Cosine Similarity - W/O PCA", "L1 Distance - Similarity- W/O PCA", "Cosine Similarity - W PCA", "L1 Distance - Similarity- W PCA"])
#plt.legend(["Cosine Similarity - d=8", "L1 Distance - Similarity- d=8", "Cosine Similarity - d=12", "L1 Distance - Similarity- d=12",
#            "Cosine Similarity - d=16", "L1 Distance - Similarity- d=16", "Cosine Similarity - d=20", "L1 Distance - Similarity- d=20"])
#plt.legend(["L1 Similarity PCA n=3", "L1 Similarity PCA n=9", "L1 Similarity PCA n=15", "L1 Similarity PCA n=20", "L1 Similarity PCA n=25",
#     "L1 Similarity PCA n=30"])
plt.show()

""""
    plt.figure("Precision")
    plt.xlabel('Top K')
    plt.ylabel('Precision')
    plt.plot(topk, precision_cs, marker='.')
    plt.plot(topk, precision_l1, marker='.')
"""

"""
    plt.figure("Recall")
    plt.xlabel('Top K')
    plt.ylabel('Recall')
    plt.plot(topk, recall_cs, marker='.')
    plt.plot(topk, recall_l1, marker='.')

    plt.legend(["Cosine Similarity", "L1 Distance- Similarity"])

    plt.show()

    plt.figure("F1 Measure")
    plt.xlabel('Top K')
    plt.ylabel('F1 Measure')
    plt.plot(topk, f1_measure_cs, marker='.')
    plt.plot(topk, f1_measure_l1, marker='.')

    plt.legend(["Cosine Similarity", "L1 Distance- Similarity"])

    plt.show()
"""
from django.conf import settings

import argparse
import csv
import cv2
import numpy as np
import os
import pandas as pd
import traceback
from lpproj import LocalityPreservingProjection
from scipy.spatial.distance import hamming
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class ORB:
    def compute(self, img, n_cluster=32, n_components=8, n_neighbors=5):
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Initiate ORB detector
        orb = cv2.ORB_create()

        # find the keypoints with ORB
        kp = orb.detect(img, None)

        # compute the descriptor with ORB
        kp, des = orb.compute(img, kp)

        if des is None:
            return None

        if des.shape[0] < n_cluster:
            return None

        kmeans = KMeans(n_clusters=n_cluster, init='k-means++', random_state=42)
        kmeans.fit(des)
        descriptor = kmeans.cluster_centers_
        # descriptor = descriptor.astype(np.uint8)

        if descriptor.shape[0] < n_neighbors:
            return None

        if n_components is not None:
            lpp = LocalityPreservingProjection(n_components=n_components, n_neighbors=n_neighbors)
            descriptor = lpp.fit_transform(descriptor)
        return descriptor.flatten()


def extract_features(path, output, type, n_cluster, n_components, n_neighbors):
    try:
        fileList = os.listdir(path)
        print('extracting orb for {} images'.format(len(fileList)))
        feature_list = []

        for label in tqdm(fileList):
            for img_file in tqdm(os.listdir(os.path.join(path, label))):
                img = cv2.imread(os.path.join(path, label, img_file))
                computer = ORB()
                descriptor = computer.compute(img, n_cluster, n_components, n_neighbors)

                if descriptor is not None:
                    feature_list.append([img_file, descriptor, label])

        if not os.path.exists(output) and output != '':
            os.makedirs(output)

        if type == 'csv':
            out_file = os.path.join(output, 'orb_pickle/orb_final.csv')

            with open(out_file, 'wt') as file:
                writer = csv.writer(file, delimiter=',', lineterminator='\n')
                writer.writerows([["file_name", "orb", "label"]])
                writer.writerows(feature_list)
                file.close()
        else:
            df = pd.DataFrame(feature_list, columns=["file_name", "orb", "label"])
            pd.to_pickle(df, 'orb_pickle/orb_final_cifar.pkl')
    except Exception as e:
        print(traceback.print_exc())
        exit(0)


def get_similarity_orb(query):
    feature__path = os.path.join(settings.BASE_DIR, 'local_feature_descriptor')
    df = pd.read_pickle(os.path.join(feature__path, 'orb_pickle/orb_final_pascal.pkl'))
    file_name = df['file_name']
    orb = df['orb'].tolist()
    labels = df['label']

    q_sim = cosine_similarity(orb, query)
    q_sim = [[(sim[0] + 1) / 2] for sim in q_sim]

    json_qsim = [{'name': file_name[i], 'similarity': np.float64(q_sim[i][0]), 'label': ', '.join(labels[i]),
                  'url': '/media/voc/{}/{}'.format(labels[i][0], file_name[i])} for i in range(len(q_sim))]

    return json_qsim


def get_similarity_orb_algorithm2(query, images):
    feature__path = os.path.join(settings.BASE_DIR, 'local_feature_descriptor')
    df = pd.read_pickle(os.path.join(feature__path, 'orb_pickle/orb_final_pascal.pkl'))

    df = df[df['file_name'].isin(images)]

    file_name = df['file_name'].tolist()
    orb = df['orb'].tolist()
    labels = df['label'].tolist()

    q_sim = cosine_similarity(orb, query)
    q_sim = [[(sim[0] + 1) / 2] for sim in q_sim]

    json_qsim = [{'name': file_name[i], 'similarity': np.float64(q_sim[i][0]), 'label': ', '.join(labels[i]),
                  'url': '/media/voc/{}/{}'.format(labels[i][0], file_name[i])} for i in range(len(q_sim))]

    return json_qsim


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ORB Extractor')

    parser.add_argument('--path', help='path to images', required=True)
    parser.add_argument('--output', help='output folder')
    parser.add_argument('--type', help='type of output csv or pkl', default='pkl')
    parser.add_argument('--n_cluster', help='number of clusters', default=32)
    parser.add_argument('--n_components', help='number of dimension in LPP', default=None)
    parser.add_argument('--n_neighbors', help='number of neighbors in LPP', default=5)

    args = parser.parse_args()
    path = args.path
    output = args.output
    type = args.type
    n_cluster = int(args.n_cluster)
    n_components = int(args.n_components)
    n_neighbors = int(args.n_neighbors)

    # n_components = None

    if type is None or type not in ['csv', 'pkl']:
        print('='*10)
        print('Invalid output type provided. It should be csv or pkl')
        print('='*10)
        exit(0)
    else:
        extract_features(path, output, type, n_cluster, n_components, n_neighbors)

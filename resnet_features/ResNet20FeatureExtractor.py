from django.conf import settings

from tf2cv.model_provider import get_model as tf2cv_get_model
import tensorflow as tf
import cv2
from skimage.transform import resize
import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse
import traceback


class ResNet20FeatureExtractor:

    def __init__(self):
        self.model = tf2cv_get_model("resnet20_cifar10", pretrained=True, data_format="channels_last")

    def get_model(self):
        return self.model


image_size = 32
model = ResNet20FeatureExtractor().get_model()
code = 'resnet'
# penultimate layer
#model._layers.pop();


def extract_feature_resnet(query_image_path):
    img = cv2.imread(query_image_path)
    img = resize(img, (image_size, image_size))
    img = tf.expand_dims(img, 0)
    feature = model(img)
    #sm = tf.nn.softmax(feature)
    return feature.numpy().reshape(-1)


def get_similarity_resnet(query, dataset):
    feature_csv_path = os.path.join(settings.BASE_DIR, 'resnet_features')
    if dataset == 'cifar':
        df = pd.read_pickle(os.path.join(feature_csv_path, 'cifar_resnet_logits.pkl'))
    if dataset == 'pascal':
        df = pd.read_pickle(os.path.join(feature_csv_path, 'deeplab_pascal.pkl'))

    file_name = df['file_name']
    features = df['resnet'].tolist()
    labels = df['label']

    json_qsim = []
    if dataset == 'cifar':
        q_sim = cosine_similarity(features, query)
        q_sim = [[(sim[0] + 1) / 2] for sim in q_sim]
        for i in range(len(q_sim)):
            for label in labels[i]:
                row = {'name': file_name[i], 'similarity': np.float64(q_sim[i][0]), 'label': label,
                          'url': '/media/cifar10/{}/{}'.format(label, file_name[i])}

                json_qsim.append(row)
    if dataset == 'pascal':
        query = query.reshape(-1, 23)
        query = query[:, 1:-2]
        features = np.array(features)
        features = features[:, 1:-2]
        q_sim = cosine_similarity(features, query)
        q_sim = [[(sim[0] + 1) / 2] for sim in q_sim]
        for i in range(len(q_sim)):
            row = {'name': file_name[i], 'similarity': q_sim[i][0], 'label': ', '.join(labels[i]),
                   'url': '/media/voc/{}/{}'.format(labels[i][0], file_name[i])}

            json_qsim.append(row)

    return json_qsim


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ResNet20 Extractor with Cifar 10 weights')
    parser.add_argument('--stop_at', help='number of items per folder to extract, defaults to all')

    args = parser.parse_args()
    stop_at = args.stop_at
    if not stop_at:
        stop_at = -1

    path = '../media/cifar10'
    feature_list = []
    count = 0

    for label in tqdm(os.listdir(path)):
        for img_file in tqdm(os.listdir(os.path.join(path, label))):
            if count == stop_at:
                count = 0
                break

            row = [img_file,extract_feature_resnet(os.path.join(path, label, img_file)), label]
            feature_list.append(row)
            count += 1

    df = pd.DataFrame(feature_list, columns=["file_name", code, "label"])
    pd.to_pickle(df, code+'.pkl')

    query=extract_feature_resnet('toy_data/airplane_22.png')

    df = pd.read_pickle(code+'.pkl')
    file_name = df['file_name']
    features = df[code]
    labels = df['label']

    results = get_similarity_resnet(query);
    print(results)



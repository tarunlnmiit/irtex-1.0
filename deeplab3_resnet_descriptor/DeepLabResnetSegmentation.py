from tf2cv.model_provider import get_model as tf2cv_get_model
import tensorflow as tf
import cv2
from skimage.transform import resize
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse
import numpy as np



class DeepLabResnetSegmentation:
    def __init__(self):

        self.model = tf2cv_get_model("deeplabv3_resnetd152b_voc", aux=False, pretrained_backbone=True, pretrained=True,
        data_format="channels_last")

    def get_model(self):
        return self.model


image_size = (480, 480)
model = DeepLabResnetSegmentation().get_model()
code = 'deeplab3'



def extract_feature_deeplab(query_image_path):
    img = cv2.imread(query_image_path)
    img = resize(img,image_size)
    img = tf.expand_dims(img, 0)
    m = tf2cv_get_model("deeplabv3_resnetd152b_voc", aux=False, pretrained_backbone=True, pretrained=True,
        data_format="channels_last")
    semantic_seg = m(img)

    output = tf.argmax(semantic_seg, 3)

    indexer = np.zeros(23)
    unique, counts = np.unique(output[0], return_counts=True)
    counts = np.asarray([unique, counts]).T
    for i, v in enumerate(counts):
        indexer[v[0]] = v[1]

    background = indexer[0]
    indexer[0] = 0

    rank = np.argsort(-indexer)[:2]
    indexer[21] = int(rank[0])
    indexer[22] = int(rank[1])
    indexer[0] = background

    return indexer


def get_similarity_deeplab(query, _path=''):
    print('in get sim')
    print(query.shape)
    query = query.reshape(-1, 23)
    print(query.shape)
    query = query[:,1:-2]
    print(query.shape)
    feature_file = 'deeplab_pascal_all'
    features_path = feature_file + '.pkl'
    image_files_path = '../media/pascal'
    if _path is not '':
        features_path = os.path.join(_path, 'deeplab3_resnet_descriptor',feature_file+'.pkl')
        image_files_path = os.path.join(_path, 'media/pascal')

    df = pd.read_pickle(features_path)

    # to prevent results for images for which we do not have the files.
    present_local_pascal_files = os.listdir(image_files_path)
    df = df[df['file_name'].isin(present_local_pascal_files)]

    file_name = df['file_name'].tolist()
    features = df['descriptor'].tolist()
    features = np.array(features)
    features = features[:,1:-2]
    print(features.shape)
    labels = df['classes']

    q_sim = cosine_similarity(features, query)
    json_qsim = [{'name': file_name[i], 'similarity': str(round(q_sim[i][0],4)), 'label': 'label',
                  'url': '/media/pascal/{}'.format(file_name[i])} for i in range(len(q_sim))]

    return json_qsim


if __name__ == "__main__":


    stop_at = -1
    if not stop_at:
        stop_at = -1

    path = '../media/pascal'
    feature_list = []
    count = 0

    for img_file in tqdm(os.listdir(path)):
        if count == stop_at:
            count = 0
            break

        row = [img_file,extract_feature_deeplab(os.path.join(path, img_file)), 0]
        feature_list.append(row)
        count += 1

    df = pd.DataFrame(feature_list, columns=["file_name", code, "label"])
    pd.to_pickle(df, code + '.pkl')
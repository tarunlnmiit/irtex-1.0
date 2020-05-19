from django.conf import settings

import cv2
import sys
import os
import numpy as np
import pandas as pd
import argparse
import math
import csv
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

parser = argparse.ArgumentParser(description='CLD Extractor')

parser.add_argument('path', help='path to images')
parser.add_argument('output', help='output folder')
parser.add_argument('--number', help='number to extract')


class CLDescriptor:

    def __init__(self):
        self.rows = 8
        self.cols = 8
        self.prefix = "CLD"

    def compute(self, img):
        averages = np.zeros((self.rows, self.cols, 3))
        imgH, imgW, _ = img.shape
        for row in range(self.rows):
            for col in range(self.cols):
                slice = img[imgH // self.rows * row: imgH // self.rows * (row + 1),
                            imgW // self.cols * col: imgW // self.cols * (col + 1)]
                average_color_per_row = np.mean(slice, axis=0)
                average_color = np.mean(average_color_per_row, axis=0)
                average_color = np.uint8(average_color)
                averages[row][col][0] = average_color[0]
                averages[row][col][1] = average_color[1]
                averages[row][col][2] = average_color[2]
        icon = cv2.cvtColor(
            np.array(averages, dtype=np.uint8), cv2.COLOR_BGR2YCR_CB)
        y, cr, cb = cv2.split(icon)
        dct_y = cv2.dct(np.float32(y))
        dct_cb = cv2.dct(np.float32(cb))
        dct_cr = cv2.dct(np.float32(cr))
        dct_y_zigzag = []
        dct_cb_zigzag = []
        dct_cr_zigzag = []
        flip = True
        flipped_dct_y = np.fliplr(dct_y)
        flipped_dct_cb = np.fliplr(dct_cb)
        flipped_dct_cr = np.fliplr(dct_cr)
        for i in range(self.rows + self.cols - 1):
            k_diag = self.rows - 1 - i
            diag_y = np.diag(flipped_dct_y, k=k_diag)
            diag_cb = np.diag(flipped_dct_cb, k=k_diag)
            diag_cr = np.diag(flipped_dct_cr, k=k_diag)
            if flip:
                diag_y = diag_y[::-1]
                diag_cb = diag_cb[::-1]
                diag_cr = diag_cr[::-1]
            dct_y_zigzag.append(diag_y)
            dct_cb_zigzag.append(diag_cb)
            dct_cr_zigzag.append(diag_cr)
            flip = not flip
        return np.concatenate(
            [np.concatenate(dct_y_zigzag), np.concatenate(dct_cb_zigzag), np.concatenate(dct_cr_zigzag)])


def get_similarity(query):
    feature_csv_path = os.path.join(settings.BASE_DIR, 'color_layout_descriptor')
    df = pd.read_csv(os.path.join(feature_csv_path, 'cld_1.csv'))
    file_name = df['file_name']
    cld = df['cld']
    cld = [[float(i) for i in elem.strip('[] ').split()] for elem in cld]
    labels = df['label']

    q_sim = cosine_similarity(cld, query)
    json_qsim = [{'name': file_name[i], 'similarity': q_sim[i][0], 'label': labels[i],
                  'url': '/media/cifar10/{}/{}'.format(labels[i], file_name[i])} for i in range(len(q_sim))]

    df = pd.read_csv(os.path.join(feature_csv_path, 'cld_2.csv'))
    file_name = df['file_name']
    cld = df['cld']
    cld = [[float(i) for i in elem.strip('[] ').split()] for elem in cld]
    labels = df['label']

    json_qsim.extend([{'name': file_name[i], 'similarity': q_sim[i][0], 'label': labels[i],
                       'url': '/media/cifar10/{}/{}'.format(labels[i], file_name[i])} for i in range(len(q_sim))])

    q_sim = cosine_similarity(cld, query)

    return json_qsim


def read_image(path):
    return cv2.imread(path)


def extract_features(path, output):

    fileList = os.listdir(path)
    print('extracting cld for {} images'.format(len(fileList)))

    feature_list = []

    for label in tqdm(os.listdir(path)):
        for img_file in tqdm(os.listdir(os.path.join(path, label))):
            img = cv2.imread(os.path.join(path, label, img_file))
            computer = CLDescriptor()
            descriptor = np.around(computer.compute(img), decimals=4)

            row = [img_file, descriptor, label]
            feature_list.append(row)
    if not os.path.exists(output) and output != '':
        os.makedirs(output)
    out_file_1 = os.path.join(output, 'cld_1.csv')
    out_file_2 = os.path.join(output, 'cld_2.csv')

    n = len(feature_list)

    with open(out_file_1, 'wt') as file:
        writer = csv.writer(file, delimiter=',',  lineterminator='\n')
        writer.writerows([["file_name", "cld", "label"]])
        writer.writerows(feature_list[:n//2])
        file.close()

    with open(out_file_2, 'wt') as file:
        writer = csv.writer(file, delimiter=',',  lineterminator='\n')
        writer.writerows([["file_name", "cld", "label"]])
        writer.writerows(feature_list[n//2:])
        file.close()


if __name__ == "__main__":
    args = parser.parse_args()
    path = args.path
    output = args.output

    extract_features(path, output)

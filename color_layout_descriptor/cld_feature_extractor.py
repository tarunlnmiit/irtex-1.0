import cv2
import sys
import os
import numpy as np
from numpy.linalg import norm
import argparse
import math
import csv
from tqdm import tqdm

parser = argparse.ArgumentParser(description='CLD Extractor')

parser.add_argument('path', help='path to images')
parser.add_argument('output', help='output folder')
parser.add_argument('--number', help='number to extract')


class ColorLayoutComputer():

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


def get_similarity(descriptor1, descriptor2):
    descriptor1 = descriptor1.astype(float)
    descriptor2 = descriptor2.astype(float)
    cos_sim = np.dot(descriptor1, descriptor2) / \
        (norm(descriptor1)*norm(descriptor2))
    return cos_sim


def read_image(path):
    return cv2.imread(path)


def extract_features(path, output):

    fileList = os.listdir(path)
    print('extracting cld for {} images'.format(len(fileList)))

    feature_list = [["file_name", "cld", "label"]]

    for label in tqdm(os.listdir(path)):
        for img_file in tqdm(os.listdir(os.path.join(path, label))):
            img = cv2.imread(os.path.join(path, label, img_file))
            computer = ColorLayoutComputer()
            descriptor = computer.compute(img)

            row = [img_file, descriptor, label]
            feature_list.append(row)
    if not os.path.exists(output) and output != '':
        os.makedirs(output)
    out_file = os.path.join(output, 'cld.csv')

    with open(out_file, 'wt') as file:
        writer = csv.writer(file, delimiter=',',  lineterminator='\n')
        writer.writerows(feature_list)
        file.close()


if __name__ == "__main__":
    args = parser.parse_args()
    path = args.path
    output = args.output

    extract_features(path, output)
import numpy as np
import argparse
import os
import cv2
from tqdm import tqdm
import mahotas
import pandas as pd
import csv


class ZernikeMoments:
    def __init__(self, radius):
        # store the size of the radius that will be
        # used when computing moments
        self.radius = radius

    def describe(self, image):
        # return the Zernike moments for the image
        return mahotas.features.zernike_moments(image, self.radius)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLD Extractor')

    parser.add_argument('--path', help='path to images')
    parser.add_argument('--output', help='output folder')
    parser.add_argument('--type', help='type of output csv or pkl')
    args = parser.parse_args()
    path = args.path
    output = args.output
    type = args.type

    if type is None or type not in ['csv', 'pkl']:
        print('='*10)
        print('Invalid output type provided. It should be csv or pkl')
        print('='*10)
        exit(0)
    else:
        desc = ZernikeMoments(16)
        zernike = []

        # dataFile = os.path.join('/Users/tarun/Projects/irtex_env/irtex/media/cifar10/')
        # path = os.path.join(dataFile)

        data_list = [["file_name", "moments", "label"]]

        # Loading the image dataset and converting into zernike moments
        for label in tqdm(os.listdir(path)):
            for img in tqdm(os.listdir(os.path.join(path, label))):
                img_array = cv2.imread(os.path.join(path, label, img), cv2.IMREAD_UNCHANGED)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                # img_array = cv2.resize(img_array, (256, 256))
                # img_array = img_array + np.random.normal(scale=2, size=img_array.shape)
                moments = desc.describe(img_array)

                row = [img, moments, label]
                data_list.append(row)

                zernike.append(moments)
        if type == 'csv':
            # Storing the extracted features into CSV file
            with open(os.path.join(output, 'moments.csv'), 'w', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerows(data_list)
            file.close()
        else:
            columns = data_list.pop(0)
            df = pd.DataFrame(data_list, columns=columns)
            pd.to_pickle(df, os.path.join(output, 'moments.pkl'))

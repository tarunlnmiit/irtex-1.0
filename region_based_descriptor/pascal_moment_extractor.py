import numpy as np
import argparse
import os
import cv2
from tqdm import tqdm
import mahotas
import pandas as pd
import csv
from sklearn.decomposition import PCA


class ZernikeMoments:
    def __init__(self, radius,degree):
        # store the size of the radius that will be
        # used when computing moments
        self.radius = radius
        self.degree = degree

    def describe(self, image):
        # return the Zernike moments for the image
        return mahotas.features.zernike_moments(image, self.radius, degree=self.degree)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RBSD Extractor')

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
        dataset = 'pascal'
        radius = 128
        degree = 20
        desc = ZernikeMoments(radius,degree)
        zernike = []

        # dataFile = os.path.join('/Users/tarun/Projects/irtex_env/irtex/media/cifar10/')
        # path = os.path.join(dataFile)

        data_list = [["file_name", "moments", "label"]]
        pca_moments = [["file_name", "moments", "label"]]
        file_name = []
        labels = []

        df = pd.read_pickle('moments_pascal_updated.pkl')
        file_present = df['file_name']
        # Loading the image dataset and converting into zernike moments
        for label in tqdm(os.listdir(path)):
            for img in tqdm(os.listdir(os.path.join(path, label))):
                if img not in file_present:
                    img_array = cv2.imread(os.path.join(path, label, img), cv2.IMREAD_UNCHANGED)
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                    img_array = cv2.resize(img_array, (256, 256))
                    # img_array = img_array + np.random.normal(scale=2, size=img_array.shape)
                    moments = desc.describe(img_array)

                    row = [img, moments, label]
                    data_list.append(row)

                    file_name.append(img)
                    zernike.append(moments)
                    labels.append(label)

        n_comp = 20
        pca = PCA(n_components=n_comp)
        principalComponents = pca.fit_transform(zernike)
        for i,m in enumerate(principalComponents):
            row = [file_name[i], m, labels[i]]
            pca_moments.append(row)


        if type == 'csv':
            # Storing the extracted features into CSV file
            with open(os.path.join(output, 'moments_pascal.csv'), 'w', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerows(data_list)
            file.close()
            #Storing after applying PCA
            with open(os.path.join(output, 'moments_pascal_pca.csv'), 'w', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerows(pca_moments)
            file.close()
        else:
            columns = data_list.pop(0)
            df = pd.DataFrame(data_list, columns=columns)
            pd.to_pickle(df, os.path.join(output, 'moments_pascal_extra.pkl'))
            #Storing after applying PCA
            columns = pca_moments.pop(0)
            df = pd.DataFrame(pca_moments, columns=columns)
            pd.to_pickle(df, os.path.join(output, 'moments_pascal_extra_pca.pkl'))

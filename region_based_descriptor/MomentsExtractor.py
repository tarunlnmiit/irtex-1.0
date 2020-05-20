from django.conf import settings

import numpy as np
import os
import cv2
from tqdm import tqdm
import mahotas
import csv



class ZernikeMoments:
    def __init__(self, radius):
        # store the size of the radius that will be
        # used when computing moments
        self.radius = radius
    def describe(self, image):
        # return the Zernike moments for the image
        return mahotas.features.zernike_moments(image, self.radius)

desc = ZernikeMoments(16)
zernike = []

dataFile = os.path.join(settings.BASE_DIR, 'media', 'cifar10')
path = os.path.join(dataFile)

data_list = [["file_name", "moments", "label"]]

# Loading the image dataset and converting into zernike moments
for label in tqdm(os.listdir(path)):
	for img in tqdm(os.listdir(os.path.join(path, label))):
		img_array = cv2.imread(os.path.join(path, label, img), cv2.IMREAD_UNCHANGED)
		img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
		#img_array = cv2.resize(img_array, (256, 256))
		#img_array = img_array + np.random.normal(scale=2, size=img_array.shape)
		moments = desc.describe(img_array)

		row = [img, moments, label]
		data_list.append(row)

		zernike.append(moments)


# Storing the extracted features into CSV file
with open('moments.csv', 'w', newline='') as file:
	writer = csv.writer(file, delimiter=',')
	writer.writerows(data_list)

file.close()

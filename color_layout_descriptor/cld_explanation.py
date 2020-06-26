from django.conf import settings

from upload.views import randomString

import cv2
import numpy as np
import matplotlib.pyplot as plt
from .CLDescriptor import CLDescriptor
import os
from pathlib import Path


def get_explanation(query_path, result_path):
    explanation = {}
    random_string = randomString(8)
    media_path = os.path.join(settings.BASE_DIR, 'media')
    save_path = os.path.join(media_path, 'cld')
    cld = CLDescriptor()
    # read file
    query_img = cv2.imread(query_path)
    query_img_descriptor = cld.compute(query_img)
    query_img_icon = iconify(query_img)
    save_loc = os.path.join(save_path, Path(query_path).name)
    plt.imsave(save_loc, query_img_icon)

    result_img = cv2.imread(result_path)
    result_img_descriptor = cld.compute(result_img)
    result_img_icon = iconify(result_img)
    save_loc = os.path.join(save_path, Path(result_path).name)
    plt.imsave(save_loc, result_img_icon)

    sims = get_similarity_by_channel(query_img_descriptor, result_img_descriptor)

    text1 = 'The perception of brightness between the two images are similar by {}'.format(round(sims[0], 2))
    text2 = 'The blue and red components of the colors of the two images are similar by {} and {} respectively '. \
        format(round(sims[1], 2), round(sims[2], 2))
    explanation['text'] = [text1, text2]

    explanation['images'] = [{'name': 'Query Image', 'url': '/media/cld/{}'.format(Path(query_path).name)}, {'name': 'Result Image',
                                                                                     'url': '/media/cld/{}'.format(Path(result_path).name)}]

    return explanation


def iconify(img):
    self_rows = 8
    self_cols = 8
    averages = np.zeros((self_rows, self_cols, 3))
    imgH, imgW, _ = img.shape
    for row in range(self_rows):
        for col in range(self_cols):
            slice = img[imgH // self_rows * row: imgH // self_rows * (row + 1),
                    imgW // self_cols * col: imgW // self_cols * (col + 1)]
            average_color_per_row = np.mean(slice, axis=0)
            average_color = np.mean(average_color_per_row, axis=0)
            average_color = np.uint8(average_color)
            averages[row][col][0] = average_color[0]
            averages[row][col][1] = average_color[1]
            averages[row][col][2] = average_color[2]
    icon = cv2.cvtColor(np.array(averages, dtype=np.uint8), cv2.COLOR_BGR2YCR_CB)
    return icon


def get_similarity_by_channel(descriptor1, descriptor2):
    descriptor1 = descriptor1.reshape(-1, 64)
    descriptor1 = descriptor1 / np.linalg.norm(descriptor1)
    descriptor2 = descriptor2.reshape(-1, 64)
    descriptor2 = descriptor2 / np.linalg.norm(descriptor2)

    sims = []
    for i, layer in enumerate(descriptor1):
        dist = np.linalg.norm(descriptor1[i] - descriptor2[i])
        sims.append(1 / (1 + dist))
    return sims

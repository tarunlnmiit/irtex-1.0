import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from django.conf import settings
from skimage.transform import resize
import os
import tensorflow as tf
from pathlib import Path
from matplotlib import gridspec
from PIL import Image
import matplotlib.image as mpimg
from tf2cv.model_provider import get_model as tf2cv_get_model
from numpy.linalg import norm


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
    label: A 2D array with integer type, storing the segmentation label.

    Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

    Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
    A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person(s)', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
net = tf2cv_get_model("deeplabv3_resnetd152b_voc", aux=False, pretrained_backbone=True, pretrained=True,
                      data_format="channels_last")
net2 = tf2cv_get_model("resnet20_cifar10", pretrained=True, data_format="channels_last")

cifar_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']


def get_cnn_explanation(dataset, query_path, result_path):
    media_path = os.path.join(settings.BASE_DIR, 'media')
    save_path = os.path.join(media_path, 'cnn')
    # if media_path != '':
    #     save_path = os.path.join(media_path, save_path)

    if dataset == 'cifar':
        return get_cifar_explanation(query_path, result_path, save_path)
    else:
        return get_pascal_explanation(query_path, result_path, save_path)


def get_pascal_explanation(query_path, result_path, save_path):
    explanation = {}

    query_img_map = get_pascal_feature_map(query_path, save_path)
    result_img_map = get_pascal_feature_map(result_path, save_path)

    query_index = get_pascal_feature_index(query_img_map)
    result_index = get_pascal_feature_index(result_img_map)

    sim = get_similarity(query_index, result_index)

    text1 = get_pascal_most_prominent_features(query_index, 'Query Image')
    text2 = get_pascal_most_prominent_features(result_index, 'Result Image')
    explanation['text'] = [text1, text2]

    explanation['images'] = [{'name': 'Query Image', 'url': Path(query_path).name}, {'name': 'Result Image',
                                                                                     'url': Path(result_path).name}]
    return explanation


def get_cifar_explanation(query_path, result_path, save_path):
    explanation = {}

    query_prob = generate_cifar_output_probabilities(query_path)
    generate_cifar_explanation_image(save_path, query_path, query_prob)

    result_prob = generate_cifar_output_probabilities(result_path)
    generate_cifar_explanation_image(save_path, result_path, result_prob)

    sim = get_similarity(query_prob, result_prob)

    explanation = {}
    text = 'The distribution of features present in the query and result are similar by {}%'.format(round(sim * 100))
    explanation['text'] = [text]

    explanation['images'] = [{'name': 'Query Image', 'url': Path(query_path).name}, {'name': 'Result Image',
                                                                                     'url': Path(result_path).name}]
    return explanation


def generate_cifar_output_probabilities(img_path):
    im_size = (32, 32)
    image = cv2.imread(img_path)
    image = resize(image, im_size)
    image = tf.expand_dims(image, 0)
    feature = net2(image)
    return tf.nn.softmax(feature.numpy().reshape(-1))


def generate_cifar_explanation_image(save_path, image_path, probabilities):
    save_loc = os.path.join(save_path, Path(image_path).name)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.barh(cifar_labels, probabilities)
    plt.savefig(save_loc, bbox_inches='tight')


def vis_segmentation(image, seg_map, save_loc):
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
        FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.savefig(save_loc, bbox_inches='tight')


def get_pascal_feature_map(image_path, save_path):
    in_size = (480, 480)
    image = mpimg.imread(image_path)
    image = resize(image, in_size)
    img = tf.expand_dims(image, 0)
    img = tf.cast(img, tf.float32)
    semantic_seg = net(img)
    output = tf.argmax(semantic_seg, 3)
    save_loc = os.path.join(save_path, Path(image_path).name)
    vis_segmentation(image, output[0], save_loc)
    return output[0]


def get_pascal_feature_index(output_map):
    indexer = np.zeros(23)
    unique, counts = np.unique(output_map, return_counts=True)
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


def get_pascal_most_prominent_features(indexer, image_name):
    area = 480 * 480

    test = indexer[21]
    index_21 = int(indexer[21])
    index_22 = int(indexer[22])
    area_1 = indexer[index_21]
    area_2 = indexer[index_22]

    ratio_number_1 = round((area_1 / area) * 100)

    ratio_number_2 = round((area_2 / area) * 100)

    text = 'About {}% of {} likely contains {}'.format(ratio_number_1, image_name, LABEL_NAMES[index_21])
    if indexer[22] != 0:
        text = 'About {}% and {}% of {} likely contains  {} and {} respectively'.format(ratio_number_1,
                                                                                        ratio_number_2, image_name,
                                                                                        LABEL_NAMES[index_21],
                                                                                        LABEL_NAMES[index_22])
    return text


def get_similarity(descriptor1, descriptor2):
    cos_sim = np.dot(descriptor1, descriptor2) / \
              (norm(descriptor1) * norm(descriptor2))
    return cos_sim

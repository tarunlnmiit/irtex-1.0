import pickle

from django.http import JsonResponse
from django.conf import settings

from upload.models import QueryImage, QueryImageSimilarity
from upload.serializers import QueryImageSerializer
from upload.views import randomString
from region_based_descriptor.RBSDescriptor import RBSDescriptor, NumpyArrayEncoder
from color_layout_descriptor.CLDescriptor import CLDescriptor, get_similarity_cld, get_similarity_cld_algorithm2
from local_feature_descriptor.orb import ORB, get_similarity_orb, get_similarity_orb_algorithm2
from local_feature_descriptor.sift import SIFT, get_similarity_sift, get_similarity_sift_algorithm2
from segmentation.Segmentation_pascal import extract_features_pascal, get_similarity_segmentation_pascal, get_similarity_segmentation_pascal_algorithm2
from segmentation.NN_segmentation import segmentation_cifar, get_similarity_segmentation_cifar, get_similarity_segmentation_cifar_algorithm2
from vgg16_imagenet_features.VGG16FeatureExractor import extract_feature_vgg, get_similarity_vgg
from resnet_features.ResNet20FeatureExtractor import extract_feature_resnet, get_similarity_resnet
from deeplab3_resnet_descriptor.DeepLabResnetSegmentation import extract_feature_deeplab

import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import traceback
from mxnet import image
from sklearn.tree import DecisionTreeClassifier


def getCombinedResults(request, _id):
    try:
        dataset = request.GET.get('dataset', None)
        if dataset is None:
            response = JsonResponse({
                'error': 'Dataset not selected'
            })
        elif dataset not in ['cifar', 'pascal']:
            response = JsonResponse({
                'error': 'Dataset value incorrect'
            })
        else:
            if dataset == 'cifar':
                weights = [1.15, 1.0, 1.0, 3.1, 1.0]
            if dataset == 'pascal':
                weights = [1.05, 1.0, 1.05, 3.4, 1.0]

            query_url = request.GET.get('query_url', None)
            if query_url:
                query_url = '/'.join(query_url.split('/')[1:])
                image_path = os.path.join(settings.BASE_DIR, query_url)
            else:
                image_instance = QueryImage.objects.get(_id=_id)
                media_path = os.path.join(settings.BASE_DIR, 'media')
                image_path = os.path.join(media_path, str(image_instance.file))

        response = similarityComputationAlgo1(image_path, dataset, weights)

    except Exception as e:
        print(traceback.print_exc())
        response = JsonResponse({
            'error': str(traceback.print_exc())
        })
    finally:
        return response


def getCombinedResultsAlgorithm2(request, _id):
    try:
        dataset = request.GET.get('dataset', None)
        if dataset is None:
            response = JsonResponse({
                'error': 'Dataset not selected'
            })
        elif dataset not in ['cifar', 'pascal']:
            response = JsonResponse({
                'error': 'Dataset value incorrect'
            })
        else:
            if dataset == 'cifar':
                weights = [2.35, 1.1, 1.0, 1.8]
            if dataset == 'pascal':
                weights = [2.725, 1.325, 1.225, 1.225]

            query_url = request.GET.get('query_url', None)
            if query_url:
                query_url = '/'.join(query_url.split('/')[1:])
                image_path = os.path.join(settings.BASE_DIR, query_url)
            else:
                image_instance = QueryImage.objects.get(_id=_id)
                media_path = os.path.join(settings.BASE_DIR, 'media')
                image_path = os.path.join(media_path, str(image_instance.file))

        response = similarityComputationAlgo1(image_path, dataset, weights)
    except Exception as e:
        print(traceback.print_exc())
        response = JsonResponse({
            'error': str(traceback.print_exc())
        })
    finally:
        return response


def getRBSDResults(request, _id):
    try:
        image_instance = QueryImage.objects.get(_id=_id)
        media_path = os.path.join(settings.BASE_DIR, 'media')
        image_path = os.path.join(media_path, str(image_instance.file))

        rbsd = RBSDescriptor()
        img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img_array = rbsd.image_preprocessing(img_array)
        q_moment = rbsd.zernike_moments(img_array)
        sim_rbsd = rbsd.similarity(q_moment)
        sim_rbsd.sort(key=lambda x: x['similarity'], reverse=True)

        result = sim_rbsd[:200]

        response = JsonResponse({
            'result': result,
            'features': ['RBSD']
        })
    except Exception as e:
        print(traceback.print_exc())
        response = JsonResponse({
            'error': str(traceback.print_exc())
        })
    finally:
        return response


def getCLDResults(request, _id):
    try:
        image_instance = QueryImage.objects.get(_id=_id)
        media_path = os.path.join(settings.BASE_DIR, 'media')
        image_path = os.path.join(media_path, str(image_instance.file))

        cld = CLDescriptor()
        img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        descriptor = cld.compute(img_array).reshape(1, -1)

        sim = get_similarity_cld(descriptor)
        sim.sort(key=lambda x: x['similarity'], reverse=True)
        response = JsonResponse({
            'result': sim[:200],
            'features': ['CLD']
        })
    except Exception as e:
        print(traceback.print_exc())
        response = JsonResponse({
            'error': traceback.print_exc()
        })
    finally:
        return response


def getSegmentationResults(request, _id):
    try:
        dataset = request.GET.get('dataset', None)
        if dataset is None:
            response = JsonResponse({
                'error': 'Dataset not selected'
            })
        elif dataset not in ['cifar', 'pascal']:
            response = JsonResponse({
                'error': 'Dataset value incorrect'
            })
        else:
            image_instance = QueryImage.objects.get(_id=_id)
            media_path = os.path.join(settings.BASE_DIR, 'media')
            image_path = os.path.join(media_path, str(image_instance.file))

            if dataset == 'cifar':
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                segmented_query_img = segmentation_cifar(img)
                sim_segmentation = get_similarity_segmentation_cifar(segmented_query_img)
            elif dataset == 'pascal':
                img = image.imread(image_path)
                segmented_query_img_pca = extract_features_pascal(img)
                sim_segmentation = get_similarity_segmentation_pascal(segmented_query_img_pca)

            sim_segmentation.sort(key=lambda x: x['similarity'], reverse=True)

            response = JsonResponse({
                'result': sim_segmentation[:200],
                'features': ['Segmentation']
            })
    except Exception as e:
        print(traceback.print_exc())
        response = JsonResponse({
            'error': traceback.print_exc()
        })
    finally:
        return response


def getLocalResults(request, _id):
    try:
        dataset = request.GET.get('dataset', None)
        if dataset is None:
            response = JsonResponse({
                'error': 'Dataset not selected'
            })
        elif dataset not in ['cifar', 'pascal']:
            response = JsonResponse({
                'error': 'Dataset value incorrect'
            })
        else:
            image_instance = QueryImage.objects.get(_id=_id)
            media_path = os.path.join(settings.BASE_DIR, 'media')
            image_path = os.path.join(media_path, str(image_instance.file))

            if dataset == 'cifar':
                features = ['SIFT']
                sift = SIFT()
                img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                descriptor = sift.compute(img_array, 16, None).reshape(1, -1)

                sim = get_similarity_sift(descriptor)
                sim.sort(key=lambda x: x['similarity'], reverse=True)
            elif dataset == 'pascal':
                features = ['ORB']
                orb = ORB()
                img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                descriptor = orb.compute(img_array, 16, 8).reshape(1, -1)

                sim = get_similarity_orb(descriptor)
                sim.sort(key=lambda x: x['similarity'], reverse=True)

            response = JsonResponse({
                'result': sim[:200],
                'features': features
            })
    except Exception as e:
        print(traceback.print_exc())
        response = JsonResponse({
            'error': traceback.print_exc()
        })
    finally:
        return response


def getVGG16Results(request, _id):
    try:
        image_instance = QueryImage.objects.get(_id=_id)
        media_path = os.path.join(settings.BASE_DIR, 'media')
        image_path = os.path.join(media_path, str(image_instance.file))
        feature = extract_feature_vgg(image_path)
        sim = get_similarity_vgg(feature,settings.BASE_DIR)

        sim.sort(key=lambda x: x['similarity'], reverse=True)
        print(len(sim))
        response = JsonResponse({
            'result': sim[:200],
            'cld': [],
            'rbsd':[],
            'features': ['VGG', 'CLD', 'RBSD']
        })
    except Exception as e:
        print(traceback.print_exc())
        response = JsonResponse({
            'error': traceback.print_exc()
        })
    return response


def getResnet20Results(request, _id):
    try:
        dataset = request.GET.get('dataset', None)
        if dataset is None:
            response = JsonResponse({
                'error': 'Dataset not selected'
            })
        elif dataset not in ['cifar', 'pascal']:
            response = JsonResponse({
                'error': 'Dataset value incorrect'
            })
        else:
            image_instance = QueryImage.objects.get(_id=_id)
            media_path = os.path.join(settings.BASE_DIR, 'media')
            image_path = os.path.join(media_path, str(image_instance.file))
            if dataset == 'cifar':
                df = pd.read_pickle(os.path.join(settings.BASE_DIR, 'resnet_features/cifar_resnet_logits.pkl'))
            elif dataset == 'pascal':
                df = pd.read_pickle(os.path.join(settings.BASE_DIR, 'resnet_features/deeplab_pascal.pkl'))

            descriptor = df.loc[df['file_name'] == image_instance.file].iloc[0, 1].reshape(1, -1)
            sim_resnet = get_similarity_resnet(descriptor, dataset)

            sim_resnet.sort(key=lambda x: x['similarity'], reverse=True)

            response = JsonResponse({
                'result': sim_resnet[:200],
                'cld': [],
                'rbsd': [],
                'features': ['VGG', 'CLD', 'RBSD']
            })
    except Exception as e:
        print(traceback.print_exc())
        response = JsonResponse({
            'error': traceback.print_exc()
        })
    return response


def image_data(request, _id):
    image_instance = QueryImage.objects.get(_id=_id)
    serializer = QueryImageSerializer(image_instance)

    return JsonResponse({
        'name': serializer.data['file'].split('/')[-1],
        'url': serializer.data['file'],
    })


def getLocalTextExplanations(request):
    try:
        query_id = request.GET.get('query_id', None)
        result_url = request.GET.get('result_url', None)
        dataset = request.GET.get('dataset', None)

        if dataset is None:
            response = JsonResponse({
                'error': 'Dataset not selected'
            })
        elif dataset not in ['cifar', 'pascal']:
            response = JsonResponse({
                'error': 'Dataset value incorrect'
            })
        else:
            if not query_id or not result_url:
                response = JsonResponse({
                    'error': 'Dataset not selected'
                })

        media_path = os.path.join(settings.BASE_DIR, 'media')

        query_image_instance = QueryImage.objects.get(_id=query_id)
        query_image_path = os.path.join(media_path, str(query_image_instance.file))

        # if dataset == 'cifar':
        #     result_image_path = os.path.join(media_path, 'cifar10/{}'.format(result_url))
        # elif dataset == 'pascal':
        #     result_image_path = os.path.join(media_path, 'voc/{}'.format(result_url))

        # Load the query image
        query_image = cv2.imread(query_image_path)
        query_image_org = cv2.resize(query_image, (256, 256), interpolation=cv2.INTER_AREA)
        query_image = cv2.cvtColor(query_image_org, cv2.COLOR_BGR2RGB)
        query_image = cv2.cvtColor(query_image, cv2.COLOR_RGB2GRAY)

        # Load the result image
        result_url = '/'.join(result_url.split('/')[1:])
        result_image_path = os.path.join(settings.BASE_DIR, result_url)
        result_image = cv2.imread(result_image_path)
        result_image_org = cv2.resize(result_image, (256, 256), interpolation=cv2.INTER_AREA)
        result_image = cv2.cvtColor(result_image_org, cv2.COLOR_BGR2RGB)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2GRAY)

        random_string = randomString(8)
        match_image_path = os.path.join(media_path, 'matches/{}.jpg'.format(random_string))

        if dataset == 'cifar':
            sift = cv2.xfeatures2d.SIFT_create()

            train_keypoints, train_descriptor = sift.detectAndCompute(query_image, None)
            test_keypoints, test_descriptor = sift.detectAndCompute(result_image, None)

            # Create a Brute Force Matcher object.
            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

            # Perform the matching between the ORB descriptors of the training image and the test image
            matches = bf.match(train_descriptor, test_descriptor)

            # The matches with shorter distance are the ones we want.
            matches = sorted(matches, key=lambda x: x.distance)

            result = cv2.drawMatches(query_image_org, train_keypoints, result_image_org, test_keypoints, matches[:5], result_image_org,
                                     flags=2)

            cv2.imwrite(match_image_path, result)
        elif dataset == 'pascal':
            orb = cv2.ORB_create()

            train_keypoints, train_descriptor = orb.detectAndCompute(query_image, None)
            test_keypoints, test_descriptor = orb.detectAndCompute(result_image, None)

            # Create a Brute Force Matcher object.
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Perform the matching between the ORB descriptors of the training image and the test image
            matches = bf.match(train_descriptor, test_descriptor)

            # The matches with shorter distance are the ones we want.
            matches = sorted(matches, key=lambda x: x.distance)

            result = cv2.drawMatches(query_image_org, train_keypoints, result_image_org, test_keypoints, matches[:5], result_image_org,
                                     flags=2)

            cv2.imwrite(match_image_path, result)
        response = JsonResponse({
            'message': 'Key points are interesting or stand out points in an image like corners, '
                       'edges that do not vary with change in image scale and rotation. '
                       'The number of key points detected in query image are {}, '
                       'the number of key points in this result image are {} '
                       'and the number of matching key points are {}'.format(len(train_keypoints), len(test_keypoints), len(matches)),
            'url': '/media/matches/{}.jpg'.format(random_string)
        })
    except Exception as e:
        print(traceback.print_exc())
        response = JsonResponse({
            'error': traceback.print_exc()
        })
    finally:
        return response


def randomQueries(request):
    try:
        dataset = request.GET.get('dataset', None)

        if dataset is None:
            response = JsonResponse({
                'error': 'Dataset not selected'
            })
        elif dataset not in ['cifar', 'pascal']:
            response = JsonResponse({
                'error': 'Dataset value incorrect'
            })
        else:
            media_path = os.path.join(settings.BASE_DIR, 'media')
            if dataset == 'cifar':
                path = os.path.join(media_path, 'cifar10')
                media_val = 'cifar10'
            if dataset == 'pascal':
                path = os.path.join(media_path, 'voc')
                media_val = 'voc'

            map = {}
            labels = os.listdir(path)
            for label in labels:
                label_path = os.path.join(path, label)
                map[label] = random.sample(os.listdir(label_path), 10)
                map[label] = ['/media/{}/{}/{}'.format(media_val, label, item) for item in map[label]]
        response = JsonResponse(map)
    except Exception as e:
        print(traceback.print_exc())
        response = JsonResponse({
            'error': traceback.print_exc()
        })
    finally:
        return response


def similarityComputationAlgo1(image_path, dataset, weights):
    # RBSD
    rbsd = RBSDescriptor(dataset)
    img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img_array = rbsd.image_preprocessing(img_array)
    q_moment = rbsd.zernike_moments(img_array)
    sim_rbsd = rbsd.similarity(q_moment)
    sim_rbsd.sort(key=lambda x: x['similarity'], reverse=True)

    # CLD
    cld = CLDescriptor()
    img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    descriptor = cld.compute(img_array).reshape(1, -1)

    sim_cld = get_similarity_cld(descriptor, dataset)
    sim_cld.sort(key=lambda x: x['similarity'], reverse=True)

    # Segmentation
    if dataset == 'cifar':
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        segmented_query_img = segmentation_cifar(img)
        sim_segmentation = get_similarity_segmentation_cifar(segmented_query_img)
    elif dataset == 'pascal':
        img = image.imread(image_path)
        segmented_query_img_pca = extract_features_pascal(img)
        sim_segmentation = get_similarity_segmentation_pascal(segmented_query_img_pca)

    sim_segmentation.sort(key=lambda x: x['similarity'], reverse=True)

    # Resnet
    if dataset == 'cifar':
        df = pd.read_pickle(os.path.join(settings.BASE_DIR, 'resnet_features/cifar_resnet_logits.pkl'))
        if len(str(image_instance.file).split('_')) > 2:
            file_name = '_'.join(str(image_instance.file).split('_')[:2]) + '.png'
        else:
            file_name = image_instance.file
        descriptor = df.loc[df['file_name'] == file_name].iloc[0, 1].reshape(1, -1)
    elif dataset == 'pascal':
        df = pd.read_pickle(os.path.join(settings.BASE_DIR, 'resnet_features/deeplab_pascal.pkl'))
        # descriptor = df.loc[df['file_name'] == image_instance.file].iloc[0, 1][1:-2].reshape(1, -1)
        descriptor = extract_feature_deeplab(image_path)

    sim_resnet = get_similarity_resnet(descriptor, dataset)

    sim_resnet.sort(key=lambda x: x['similarity'], reverse=True)

    # Local
    if dataset == 'cifar':
        features = ['SIFT']
        sift = SIFT()
        img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        descriptor = sift.compute(img_array, 16, None).reshape(1, -1)

        sim_local = get_similarity_sift(descriptor)
        # sim_local.sort(key=lambda x: x['similarity'], reverse=True)
    elif dataset == 'pascal':
        features = ['ORB']
        orb = ORB()
        img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        descriptor = orb.compute(img_array, 16, 8).reshape(1, -1)

        sim_local = get_similarity_orb(descriptor)

    sim_local.sort(key=lambda x: x['similarity'], reverse=True)

    combined = {}
    len_features = 5

    for item in sim_cld:
        name = item['name']
        combined[name] = {}
        combined[name]['name'] = name
        combined[name]['url'] = item['url']
        combined[name]['label'] = item['label']
        combined[name]['similarity_list'] = [0] * len_features
        combined[name]['similarity_list'][0] = item['similarity']

    for item in sim_rbsd:
        name = item['name']
        if name not in combined:
            combined[name] = {}
            combined[name]['name'] = name
            combined[name]['url'] = item['url']
            combined[name]['label'] = item['label']
            combined[name]['similarity_list'] = [0] * len_features
            combined[name]['similarity_list'][1] = item['similarity']
        else:
            combined[name]['similarity_list'][1] = item['similarity']

    for item in sim_segmentation:
        name = item['name']
        if name not in combined:
            combined[name] = {}
            combined[name]['name'] = name
            combined[name]['url'] = item['url']
            combined[name]['label'] = item['label']
            combined[name]['similarity_list'] = [0] * len_features
            combined[name]['similarity_list'][2] = item['similarity']
        else:
            combined[name]['similarity_list'][2] = item['similarity']

    for item in sim_resnet:
        name = item['name']
        if name not in combined:
            combined[name] = {}
            combined[name]['name'] = name
            combined[name]['url'] = item['url']
            combined[name]['label'] = item['label']
            combined[name]['similarity_list'] = [0] * len_features
            combined[name]['similarity_list'][3] = item['similarity']
        else:
            combined[name]['similarity_list'][3] = item['similarity']
            if dataset == 'pascal':
                combined[name]['label'] = item['label']

    for item in sim_local:
        name = item['name']
        if name not in combined:
            combined[name] = {}
            combined[name]['name'] = name
            combined[name]['url'] = item['url']
            combined[name]['label'] = item['label']
            combined[name]['similarity_list'] = [0] * len_features
            combined[name]['similarity_list'][4] = item['similarity']
        else:
            combined[name]['similarity_list'][4] = item['similarity']

    for k, v in combined.items():
        # temp = []
        # for sim_value in combined[k]['similarity_list']:
        #     if sim_value != 0:
        #         temp.append(sim_value)
        #     else:
        #         if isinstance(sim_value, int):
        #             continue
        #         else:
        #             temp.append(sim_value)
        # combined[k]['similarity'] = np.average(temp)
        s = 0
        for i, similarity in enumerate(combined[k]['similarity_list']):
            s += (weights[i] * similarity)
        combined[k]['similarity'] = s / np.sum(weights)
        combined[k]['similarity_list'] = [combined[k]['similarity']] + combined[k]['similarity_list']

    combined = list(combined.values())
    combined.sort(key=lambda x: x['similarity'], reverse=True)
    # sim = list(sim_final.values())
    # sim.sort(key=lambda x: x['similarity'], reverse=True)

    generateRulesAlgo1(combined)

    response = JsonResponse({
        'result': combined[:200],
        'cld': sim_cld[:200],
        'rbsd': sim_rbsd[:200],
        'segmentation': sim_segmentation[:200],
        'resnet': sim_resnet[:200],
        'local': sim_local[:200],
        'features': ['Combined', 'CLD', 'RBSD', 'Segmentation', 'Resnet', 'Local']
    })
    return response


def similarityComputationAlgo2(image_path, dataset, weights):
    # Resnet
    if dataset == 'cifar':
        df = pd.read_pickle(os.path.join(settings.BASE_DIR, 'resnet_features/cifar_resnet_logits.pkl'))
        if len(str(image_instance.file).split('_')) > 2:
            file_name = '_'.join(str(image_instance.file).split('_')[:2]) + '.png'
        else:
            file_name = image_instance.file
        descriptor = df.loc[df['file_name'] == file_name].iloc[0, 1].reshape(1, -1)
    elif dataset == 'pascal':
        df = pd.read_pickle(os.path.join(settings.BASE_DIR, 'resnet_features/deeplab_pascal.pkl'))
        # descriptor = df.loc[df['file_name'] == image_instance.file].iloc[0, 1][1:-2].reshape(1, -1)
        descriptor = extract_feature_deeplab(image_path)

    sim_resnet = get_similarity_resnet(descriptor, dataset)

    sim_resnet.sort(key=lambda x: x['similarity'], reverse=True)

    final_result_images = [item['name'] for item in sim_resnet[:200]]

    # RBSD
    rbsd = RBSDescriptor(dataset)
    img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img_array = rbsd.image_preprocessing(img_array)
    q_moment = rbsd.zernike_moments(img_array)
    sim_rbsd = rbsd.similarity_algorithm2(final_result_images, q_moment)
    sim_rbsd.sort(key=lambda x: x['similarity'], reverse=True)

    # CLD
    cld = CLDescriptor()
    img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    descriptor = cld.compute(img_array).reshape(1, -1)

    sim_cld = get_similarity_cld_algorithm2(descriptor, dataset, final_result_images)
    sim_cld.sort(key=lambda x: x['similarity'], reverse=True)

    # Segmentation
    if dataset == 'cifar':
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        segmented_query_img = segmentation_cifar(img)
        sim_segmentation = get_similarity_segmentation_cifar_algorithm2(segmented_query_img, final_result_images)
    elif dataset == 'pascal':
        img = image.imread(image_path)
        segmented_query_img_pca = extract_features_pascal(img)
        sim_segmentation = get_similarity_segmentation_pascal_algorithm2(segmented_query_img_pca, final_result_images)

    sim_segmentation.sort(key=lambda x: x['similarity'], reverse=True)

    # Local
    if dataset == 'cifar':
        features = ['SIFT']
        sift = SIFT()
        img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        descriptor = sift.compute(img_array, 16, None).reshape(1, -1)

        sim_local = get_similarity_sift_algorithm2(descriptor, final_result_images)
        # sim_local.sort(key=lambda x: x['similarity'], reverse=True)
    elif dataset == 'pascal':
        features = ['ORB']
        orb = ORB()
        img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        descriptor = orb.compute(img_array, 16, 8).reshape(1, -1)

        sim_local = get_similarity_orb_algorithm2(descriptor, final_result_images)

    sim_local.sort(key=lambda x: x['similarity'], reverse=True)

    combined = {}
    len_features = 4

    for item in sim_cld:
        name = item['name']
        combined[name] = {}
        combined[name]['name'] = name
        combined[name]['url'] = item['url']
        combined[name]['label'] = item['label']
        combined[name]['similarity_list'] = [0] * len_features
        combined[name]['similarity_list'][0] = item['similarity']

    for item in sim_rbsd:
        name = item['name']
        if name not in combined:
            combined[name] = {}
            combined[name]['name'] = name
            combined[name]['url'] = item['url']
            combined[name]['label'] = item['label']
            combined[name]['similarity_list'] = [0] * len_features
            combined[name]['similarity_list'][1] = item['similarity']
        else:
            combined[name]['similarity_list'][1] = item['similarity']

    for item in sim_segmentation:
        name = item['name']
        if name not in combined:
            combined[name] = {}
            combined[name]['name'] = name
            combined[name]['url'] = item['url']
            combined[name]['label'] = item['label']
            combined[name]['similarity_list'] = [0] * len_features
            combined[name]['similarity_list'][2] = item['similarity']
        else:
            combined[name]['similarity_list'][2] = item['similarity']

    for item in sim_local:
        name = item['name']
        if name not in combined:
            combined[name] = {}
            combined[name]['name'] = name
            combined[name]['url'] = item['url']
            combined[name]['label'] = item['label']
            combined[name]['similarity_list'] = [0] * len_features
            combined[name]['similarity_list'][3] = item['similarity']
        else:
            combined[name]['similarity_list'][3] = item['similarity']

    for k, v in combined.items():
        # temp = []
        # for sim_value in combined[k]['similarity_list']:
        #     if sim_value != 0:
        #         temp.append(sim_value)
        #     else:
        #         if isinstance(sim_value, int):
        #             continue
        #         else:
        #             temp.append(sim_value)
        # combined[k]['similarity'] = np.average(temp)
        s = 0
        for i, similarity in enumerate(combined[k]['similarity_list']):
            s += (weights[i] * similarity)
        combined[k]['similarity'] = s / np.sum(weights)
        combined[k]['similarity_list'] = [combined[k]['similarity']] + combined[k]['similarity_list']

    combined = list(combined.values())
    combined.sort(key=lambda x: x['similarity'], reverse=True)
    # sim = list(sim_final.values())
    # sim.sort(key=lambda x: x['similarity'], reverse=True)

    generateRulesAlgo2(combined)

    response = JsonResponse({
        'result': combined[:200],
        'cld': sim_cld[:200],
        'rbsd': sim_rbsd[:200],
        'segmentation': sim_segmentation[:200],
        # 'resnet': sim_resnet[:200],
        'local': sim_local[:200],
        'features': ['Combined', 'CLD', 'RBSD', 'Segmentation', 'Local']
    })
    return response


def generateRulesAlgo1(result):
    dataset = []
    for index, item in enumerate(result):
        if index < 200:
            label = 1
        else:
            label = 0
        temp = [item['name']]
        for sim in item['similarity_list'][1:]:
            temp.append(sim)
        temp.append(label)
        dataset.append(temp)

    df = pd.DataFrame(dataset, columns=['file_name', 'cld', 'rbsd', 'seg', 'resnet', 'local', 'label'])
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    features = df.columns[1:-1]
    features.tolist()

    clf = DecisionTreeClassifier(max_depth=5, random_state=42, criterion='entropy')
    clf.fit(X, y)

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    classes = clf.tree_.value.tolist()

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)

    for i in range(n_nodes):
        if is_leaves[i]:
            if classes[i][0][0] > classes[i][0][1]:
                label = 0
            else:
                label = 1
                print("%slabel = %s" % (node_depth[i] * "\t", label))
        else:
            print("%s if %s <= %s"
                  % (node_depth[i] * "\t",
                     features.tolist()[feature[i]],
                     threshold[i],
                     ))
    print()


def generateRulesAlgo2(result):
    dataset = []
    for index, item in enumerate(result):
        if index < 200:
            label = 1
        else:
            label = 0
        temp = [item['name']]
        for sim in item['similarity_list'][1:]:
            temp.append(sim)
        temp.append(label)
        dataset.append(temp)

    df = pd.DataFrame(dataset, columns=['file_name', 'cld', 'rbsd', 'seg', 'local', 'label'])
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    features = df.columns[1:-1]
    features.tolist()

    clf = DecisionTreeClassifier(max_depth=5, random_state=42, criterion='entropy')
    clf.fit(X, y)

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    classes = clf.tree_.value.tolist()

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)

    for i in range(n_nodes):
        if is_leaves[i]:
            if classes[i][0][0] > classes[i][0][1]:
                label = 0
            else:
                label = 1
                print("%slabel = %s" % (node_depth[i] * "\t", label))
        else:
            print("%s if %s <= %s"
                  % (node_depth[i] * "\t",
                     features.tolist()[feature[i]],
                     threshold[i],
                     ))
    print()
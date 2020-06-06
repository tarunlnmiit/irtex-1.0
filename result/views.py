from django.http import JsonResponse
from django.conf import settings

from upload.models import QueryImage, QueryImageSimilarity
from upload.serializers import QueryImageSerializer
from region_based_descriptor.RBSDescriptor import RBSDescriptor, NumpyArrayEncoder
from color_layout_descriptor.CLDescriptor import CLDescriptor, get_similarity_cld
from local_feature_descriptor.orb import ORB, get_similarity_orb
from local_feature_descriptor.sift import SIFT, get_similarity_sift
from vgg16_imagenet_features.VGG16FeatureExractor import extract_feature_vgg,get_similarity_vgg
from resnet20_cifar_10_features.ResNet20FeatureExtractor import extract_feature_resnet,get_similarity_resnet

import cv2
import json
import numpy as np
import os
import traceback
from collections import OrderedDict


def getCombinedResults(request, _id):
    try:
        dataset = request.GET.get('dataset', None)
        dataset = 'cifar'
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

            final_sim = OrderedDict()

            rbsd = RBSDescriptor()
            img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            img_array = rbsd.image_preprocessing(img_array)
            q_moment = rbsd.zernike_moments(img_array)
            sim_rbsd = rbsd.similarity(q_moment)
            sim_rbsd.sort(key=lambda x: x['similarity'], reverse=True)

            cld = CLDescriptor()
            img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            descriptor = cld.compute(img_array).reshape(1, -1)

            sim_cld = get_similarity_cld(descriptor)
            sim_cld.sort(key=lambda x: x['similarity'], reverse=True)

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

            for item in sim_cld:
                name = item['name']
                combined[name] = {}
                combined[name]['name'] = name
                combined[name]['url'] = item['url']
                combined[name]['label'] = item['label']
                combined[name]['similarity_list'] = [0] * 3
                combined[name]['similarity_list'][0] = item['similarity']

            for item in sim_rbsd:
                name = item['name']
                if name not in combined:
                    combined[name] = {}
                    combined[name]['name'] = name
                    combined[name]['url'] = item['url']
                    combined[name]['label'] = item['label']
                    combined[name]['similarity_list'] = [0] * 3
                    combined[name]['similarity_list'][1] = item['similarity']
                else:
                    combined[name]['similarity_list'][1] = item['similarity']

            for item in sim_local:
                name = item['name']
                if name not in combined:
                    combined[name] = {}
                    combined[name]['name'] = name
                    combined[name]['url'] = item['url']
                    combined[name]['label'] = item['label']
                    combined[name]['similarity_list'] = [0] * 3
                    combined[name]['similarity_list'][2] = item['similarity']
                else:
                    combined[name]['similarity_list'][2] = item['similarity']

            for k, v in combined.items():
                combined[k]['similarity'] = np.average(combined[k]['similarity_list'])

            combined = list(combined.values())
            combined.sort(key=lambda x: x['similarity'], reverse=True)
            # sim = list(sim_final.values())
            # sim.sort(key=lambda x: x['similarity'], reverse=True)

            response = JsonResponse({
                'result': combined[:200],
                'cld': sim_cld[:200],
                'rbsd': sim_rbsd[:200],
                'local': sim_local[:200],
                'features': ['Combined CLD & RBSD', 'CLD', 'RBSD', 'Local']
            })
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
        image_instance = QueryImage.objects.get(_id=_id)
        media_path = os.path.join(settings.BASE_DIR, 'media')
        image_path = os.path.join(media_path, str(image_instance.file))
        feature = extract_feature_resnet(image_path)
        sim = get_similarity_resnet(feature,settings.BASE_DIR)

        sim.sort(key=lambda x: x['similarity'], reverse=True)
        print(len(sim))
        response = JsonResponse({
            'result': sim[:200],
            'cld': [],
            'rbsd': [],
            'features': ['RESNET', 'CLD', 'RBSD']
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

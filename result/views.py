from django.http import JsonResponse
from django.conf import settings

from upload.models import QueryImage, QueryImageSimilarity
from upload.serializers import QueryImageSerializer
from region_based_descriptor.RBSDescriptor import RBSDescriptor
from color_layout_descriptor.CLDescriptor import CLDescriptor, get_similarity

import cv2
import numpy as np
import os
import traceback
from collections import OrderedDict


def getCombinedResults(request, _id):
    try:
        image_instance = QueryImage.objects.get(_id=_id)
        media_path = os.path.join(settings.BASE_DIR, 'media')
        image_path = os.path.join(media_path, str(image_instance.file))

        rbsd = RBSDescriptor()
        img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img_array = rbsd.image_preprocessing(img_array)
        q_moment = rbsd.zernike_moments(img_array)
        sim_rbsd = rbsd.similarity(q_moment)
        # sim_rbsd.sort(key=lambda x: x['similarity'], reverse=True)

        cld = CLDescriptor()
        img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        descriptor = np.around(cld.compute(img_array), decimals=4).reshape(1, -1)

        sim_cld = get_similarity(descriptor)
        # sim_cld.sort(key=lambda x: x['similarity'], reverse=True)

        sim = OrderedDict()

        for item in sim_rbsd:
            sim[item['name']] = item

        for item in sim_cld:
            if item['name'] in sim:
                # print(sim[item['name']], sim[item['name']]['similarity'], item['name'], item['similarity'])
                sim[item['name']]['similarity'] = np.average([sim[item['name']]['similarity'], item['similarity']])

        sim = list(sim.values())
        sim.sort(key=lambda x: x['similarity'], reverse=True)
        result = sim[:200]

        response = JsonResponse({
            'result': result
        })

        # TODO this code if server side pagination is needed
        # if request.GET.get('first', False) and request.GET.get('first') == '1':
        #     rbsd = RBSDescriptor()
        #     img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        #     img_array = rbsd.image_preprocessing(img_array)
        #     q_moment = rbsd.zernike_moments(img_array)
        #     sim_rbsd = rbsd.similarity(q_moment)
        #     # sim_rbsd.sort(key=lambda x: x['similarity'], reverse=True)
        #
        #     cld = CLDescriptor()
        #     img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        #     descriptor = np.around(cld.compute(img_array), decimals=4).reshape(1, -1)
        #
        #     sim_cld = get_similarity(descriptor)
        #     # sim_cld.sort(key=lambda x: x['similarity'], reverse=True)
        #
        #     sim = OrderedDict()
        #
        #     for item in sim_rbsd:
        #         sim[item['name']] = item
        #
        #     for item in sim_cld:
        #         if item['name'] in sim:
        #             # print(sim[item['name']], sim[item['name']]['similarity'], item['name'], item['similarity'])
        #             sim[item['name']]['similarity'] = np.average([sim[item['name']]['similarity'], item['similarity']])
        #
        #     sim = list(sim.values())
        #     sim.sort(key=lambda x: x['similarity'], reverse=True)
        #     result = sim[:10]
        #
        #     query_image_similarity_instance = QueryImageSimilarity()
        #     query_image_similarity_instance.query_image_id = _id
        #     query_image_similarity_instance.similarities = sim
        #     query_image_similarity_instance.save()
        #
        #     response = JsonResponse({
        #         'result': result
        #     })
        # elif request.GET.get('first', False) and request.GET.get('first') == '0' and request.GET.get('page', False):
        #     page = int(request.GET.get('page'))
        #     query_image_similarity_instance = QueryImageSimilarity.objects.get(query_image_id=_id)
        #     f = (10 * page) - 10
        #     l = (page * 10)
        #     result = query_image_similarity_instance.similarities[f:l]
        #     response = JsonResponse({
        #         'result': result
        #     })
        # else:
        #     response = JsonResponse({
        #         'error': 'All query parameters not received'
        #     })
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
            'result': result
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
        descriptor = np.around(cld.compute(img_array), decimals=4).reshape(1, -1)

        sim = get_similarity(descriptor)
        sim.sort(key=lambda x: x['similarity'], reverse=True)
        print(len(sim))
        response = JsonResponse({
            'result': sim[:200]
        })
    except Exception as e:
        print(traceback.print_exc())
        response = JsonResponse({
            'error': traceback.print_exc()
        })
    finally:
        return response


def image_data(request, _id):
    image_instance = QueryImage.objects.get(_id=_id)
    serializer = QueryImageSerializer(image_instance)

    return JsonResponse({
        'name': serializer.data['file'].split('/')[-1],
        'url': serializer.data['file'],
    })
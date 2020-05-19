from django.http import JsonResponse
from django.conf import settings

from upload.models import QueryImage
from upload.serializers import QueryImageSerializer
from region_based_descriptor.RBSDescriptor import RBSDescriptor
from color_layout_descriptor.CLDescriptor import CLDescriptor, get_similarity

import cv2
import numpy as np
import os
import traceback


def getRBSDResults(request, _id):
    try:
        image_instance = QueryImage.objects.get(_id=_id)
        media_path = os.path.join(settings.BASE_DIR, 'media')
        image_path = os.path.join(media_path, str(image_instance.file))

        rbsd = RBSDescriptor()
        img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img_array = rbsd.image_preprocessing(img_array)

        q_moment = rbsd.zernike_moments(img_array)
        sim = rbsd.similarity(q_moment)
        sim.sort(key=lambda x: x['similarity'], reverse=True)
        print(type(sim))
    except Exception as e:
        print(traceback.print_exc())
    return JsonResponse({
        'result': sim[:10]
    })


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
    except Exception as e:
        print(traceback.print_exc())
    return JsonResponse({
        'result': sim[:10]
    })


def image_data(request, _id):
    image_instance = QueryImage.objects.get(_id=_id)
    serializer = QueryImageSerializer(image_instance)

    return JsonResponse({
        'name': serializer.data['file'].split('/')[-1],
        'url': serializer.data['file'],
    })
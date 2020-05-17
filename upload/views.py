from django.http import HttpResponse, JsonResponse
from django.conf import settings
from django.core import files
from rest_framework.parsers import JSONParser, FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status

from .models import QueryImage
from .serializers import QueryImageSerializer

import os
import requests
import random
import string
import tempfile


def randomString(stringLength=6):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


class QueryImageUploadView(APIView):
    parser_class = (FileUploadParser,)

    def get(self, request, **kwargs):
        if request.GET.get('id', False):
            id = request.GET.get('id')
            snippet = QueryImage.objects.get(_id=id)
            serializer = QueryImageSerializer(snippet)
            return JsonResponse(serializer.data, safe=False)
        snippets = QueryImage.objects.all()
        serializer = QueryImageSerializer(snippets, many=True)
        return JsonResponse(serializer.data, safe=False)

    def post(self, request, *args, **kwargs):
        if request.data.get('url', False):
            url = request.data.get('url')

            request = requests.get(url, stream=True)
            file_name = 'image' + randomString() + '.'  + url.split('.')[-1]
            lf = tempfile.NamedTemporaryFile()

            for block in request.iter_content(1024 * 8):
                if not block:
                    break
                lf.write(block)

            image = QueryImage()
            image.url = url
            image.file.save(file_name, files.File(lf))
            query_image_serializer = QueryImageSerializer(image)
            return Response(query_image_serializer.data, status=status.HTTP_201_CREATED)
        else:
            query_image_serializer = QueryImageSerializer(data=request.data)

            if query_image_serializer.is_valid():
                query_image_serializer.save()
                return Response(query_image_serializer.data, status=status.HTTP_201_CREATED)
            else:
                return Response(query_image_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    '''doesnt work yet'''
    # def delete(self, request):
    #     if request.GET.get('id', False):
    #         id = request.GET.get('id')
    #         print('here')
    #         snippet = QueryImage.objects.get(_id=id)
    #         snippet.delete()
    #         serializer = QueryImageSerializer(snippet)
    #         return JsonResponse(serializer.data, safe=False)
    #     else:
    #         return JsonResponse({
    #             'message': 'No ID received'
    #         })

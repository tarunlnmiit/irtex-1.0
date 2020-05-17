from django.shortcuts import render
from django.http import JsonResponse


def getResults(request, _id):
    print(_id)
    return JsonResponse({
        'string': 'result'
    })
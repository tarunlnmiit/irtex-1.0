from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from .models import UserData
from .serializers import UserDataSerializer


@csrf_exempt
def user_list(request):
    """
    List all code snippets, or create a new user's data.
    """
    if request.method == 'GET':
        snippets = UserData.objects.all()
        serializer = UserDataSerializer(snippets, many=True)
        return JsonResponse(serializer.data, safe=False)
    elif request.method == 'POST':
        data = JSONParser().parse(request)
        serializer = UserDataSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data, status=201)
        return JsonResponse(serializer.errors, status=400)


@csrf_exempt
def user_detail(request, _id):
    """
        Retrieve, update or delete a user's data.
        """
    try:
        snippet = UserData.objects.get(_id=_id)
    except UserData.DoesNotExist:
        return HttpResponse(status=404)

    if request.method == 'GET':
        serializer = UserDataSerializer(snippet)
        return JsonResponse(serializer.data)
    elif request.method == 'PUT':
        data = JSONParser().parse(request)
        serializer = UserDataSerializer(snippet, data=data)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data)
        return JsonResponse(serializer.errors, status=400)
    elif request.method == 'DELETE':
        snippet.delete()
        return HttpResponse(status=204)


def hello(request):
    return JsonResponse({
        'string': 'hello'
    })
from .models import QueryImage
from rest_meets_djongo.serializers import DjongoModelSerializer


class QueryImageSerializer(DjongoModelSerializer):
    class Meta:
        model = QueryImage
        fields = '__all__'

from .models import UserData
from rest_meets_djongo.serializers import DjongoModelSerializer


class UserDataSerializer(DjongoModelSerializer):
    class Meta:
        model = UserData
        fields = '__all__'

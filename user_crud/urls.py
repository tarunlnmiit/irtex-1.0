from django.urls import path
from . import views

urlpatterns = [
    path('userdata/', views.user_list),
    path('userdata/<str:_id>/', views.user_detail),
    path('/', views.hello),
]

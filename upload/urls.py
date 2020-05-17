from django.urls import path
from . import views

urlpatterns = [
    path('', views.QueryImageUploadView.as_view()),
]
from django.urls import path
from . import views

urlpatterns = [
    path('cld/<str:_id>', views.getCLDResults),
    path('rbsd/<str:_id>', views.getRBSDResults),
    path('segmentation/<str:_id>', views.getSegmentationResults),
    path('<str:_id>', views.getCombinedResults),
    path('vgg/<str:_id>', views.getVGG16Results),
    path('resnet/<str:_id>', views.getResnet20Results),
]
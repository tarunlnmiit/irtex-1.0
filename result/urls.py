from django.urls import path
from . import views

urlpatterns = [
    path('cld/<str:_id>', views.getCLDResults),
    path('rbsd/<str:_id>', views.getRBSDResults),
    path('segmentation/<str:_id>', views.getSegmentationResults),
    path('vgg/<str:_id>', views.getVGG16Results),
    path('resnet/<str:_id>', views.getResnet20Results),
    path('explain/local/', views.getLocalTextExplanations),
    path('explain/cld/', views.getCLDTextExplanations),
    path('explain/rbsd/', views.getRBSDTextExplanations),
    path('explain/segmentation/', views.getSegTextExplanations),
    path('explain/global/', views.getGlobalTextExplanations),
    path('queries', views.randomQueries),
    path('algo2/<str:_id>', views.getCombinedResultsAlgorithm2),
    path('algo2/<str:_id>', views.getCombinedResultsAlgorithm2),
    path('url', views.getResults),
    path('compare', views.storeCompareUserClicks),
    path('<str:_id>', views.getCombinedResults),
]
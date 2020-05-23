from django.urls import path
from . import views

urlpatterns = [
    path('cld/<str:_id>', views.getCLDResults),
    path('rbsd/<str:_id>', views.getRBSDResults),
    path('<str:_id>', views.getCombinedResults),
]
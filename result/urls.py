from django.urls import path
from . import views

urlpatterns = [
    path('rbsd/<str:_id>', views.getRBSDResults),
    path('cld/<str:_id>', views.getCLDResults),
]
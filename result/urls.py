from django.urls import path
from . import views

urlpatterns = [
    path('<str:_id>', views.getRBSDResults),
    path('cld/<str:_id>', views.getCLDResults),
]
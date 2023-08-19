from django.urls import path
from . import views

urlpatterns = [
    path('heart_beat/', views.health_check, name='health_check'),
]

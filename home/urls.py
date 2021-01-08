from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('nhandang/', views.find, name = "nhandang"),
]
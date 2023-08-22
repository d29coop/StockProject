from django.urls import path
from . import views
urlpatterns = [
    path('', views.home, name='index'),
    path('input', views.input),
    path('output', views.output),
]
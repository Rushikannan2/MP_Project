from django.urls import path
from . import views

app_name = 'graphapp'

urlpatterns = [
    path('', views.home, name='home'),
    path('solve/', views.solve, name='solve'),
]

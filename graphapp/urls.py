from django.urls import path
from . import views

app_name = 'graphapp'

urlpatterns = [
    path('', views.home, name='home'),
    path('graphical/', views.graphical_view, name='graphical'),
    path('solve/', views.solve, name='solve'),
    path('transportation/', views.transportation_view, name='transportation'),
    path('solve-transportation/', views.solve_transportation, name='solve_transportation'),
    path('simplex/', views.simplex_view, name='simplex'),
    path('solve-simplex/', views.solve_simplex, name='solve_simplex'),
    path('applications/', views.applications, name='applications'),
]
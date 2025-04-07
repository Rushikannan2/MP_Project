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
    path('integer-programming/', views.integer_programming, name='integer_programming'),
    path('fractional-programming/', views.fractional_programming, name='fractional_programming'),
]
from django.urls import path
from .views import *
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('signup/', signup_view, name='signup'),
    path('login/', CustomLoginView.as_view(), name='login'),  # Custom login view
    # path('dashboard/', dashboard_view, name='dashboard'),    # Dashboard view
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('', views.home, name='home'),
    path('profile/', views.profile_management, name='profile_management'),
    path('logout/', custom_logout_view, name='custom_logout'),
    path('reports/', views.report_list, name='report_list'),
     path('report/<int:report_id>/', report_detail_view, name='report_detail'),
    path('reports/download/<int:report_id>/', views.download_report, name='download_report'),
    path('reports/generate/', views.generate_report, name='generate_report'),
    path('profile/', profile_management, name='profile_management'),
    path('upload/', upload_image, name='upload_image'),
    path('results/', views.detection_results, name='detection_results'),
    path('download_report/', views.download_report_view, name='download_report'),
    path('visualizations/', combined_visualization_view, name='combined_visualization'),
]



# grow_builder/urls.py
from django.urls import path
from . import views # Import views from the current directory

# This is often helpful to namespace URLs, especially if you have multiple apps
app_name = 'grow_builder'

urlpatterns = [
    # URL for the main builder page (e.g., /grow-builder/)
    # The empty string '' means it matches the base URL included from the project urls.py
    path('', views.grow_room_builder_view, name='builder_page'),

    # URL for the API endpoint (e.g., /grow-builder/api/calculate-lights/)
    path('api/calculate-lights/', views.calculate_lights_api, name='calculate_lights_api'),
]
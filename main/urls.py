from django.urls import path
from main import views

urlpatterns = [
    path('', views.index, name='index'),  # Use views.index and name it 'index'
    path('technology/', views.technology, name='technology'),
    path('applications/', views.applications, name='applications'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('contact/success/', views.contact_success, name='contact_success'),
    path('api/simulation/', views.run_simulation, name='run_simulation'),
    path('simulation/', views.simulation_view, name='simulation_view'),
    path('agenticai/', views.agenticai, name='agenticai'),
    path('coolingsystem/', views.cooling_system, name='cooling_system'),
]
from django.urls import path
from main import views
from .views import blog_admin_panel
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'),
    path('technology/', views.technology, name='technology'),
    path('blog/', views.blog, name='blog'),
    path('blogs/', views.blog_index, name='blog_index'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('contact/success/', views.contact_success, name='contact_success'),
    # Updated simulation endpoint URL to use an underscore to match the module name
    path('api/ml_simulation/', views.run_simulation, name='run_simulation'),
    path('simulation/', views.simulation_view, name='simulation_view'),
    path('agenticai/', views.agenticai, name='agenticai'),
    path('coolingsystem/', views.cooling_system, name='cooling_system'),
    path('cea/', views.cea, name='cea'),
    path('lighting_simulation/', views.lighting_simulation, name='lighting_simulation'),
    path('blog/upfp/', views.upfp, name='blog/upfp'),
    path('dialux_simulation/', views.dialux_simulation, name='dialux_simulation'),
    path('ajax/toggle-like/', views.toggle_like, name='toggle_like'),
    path('ajax/add-comment/', views.add_comment, name='add_comment'),
    path('ajax/log-share/', views.log_share, name='log_share'),
    path('blog-admin/', blog_admin_panel, name='blog_admin_panel'),
    path('blog-admin/create/', views.create_blog_post, name='create_blog_post'),
    path('blog/edit/<slug:slug>/', views.edit_blog_post, name='edit_blog_post'),
    path('blog/delete/<slug:slug>/', views.delete_blog_post, name='delete_blog_post'),
    path('blogs/<slug:slug>/', views.blog_detail, name='blog_detail'),
    path('blog/approve/<slug:slug>/', views.approve_blog_post, name='approve_blog_post'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

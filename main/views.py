from django.shortcuts import render, redirect, get_object_or_404
from django.core.mail import send_mail, EmailMessage
from django.conf import settings
from django.contrib import messages
from .forms import ContactForm
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .ml_simulation import run_ml_simulation  # Import from ml_simulation.py
from django.contrib.auth.decorators import login_required
from .blog_system import BlogPost, Like, Share  # our model file
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.http import require_POST
from .forms import BlogPostForm
from django.http import HttpResponse
from .blog_system import BlogPost
import mammoth
from django.contrib.auth.decorators import user_passes_test
import io

def index(request):
    return render(request, 'main/index.html')

def technology(request):
    return render(request, 'main/technology.html')

def blog(request):
    return render(request, 'main/blog.html')

def about(request):
    return render(request, 'main/about.html')

def contact(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            email = form.cleaned_data['email']
            subject = form.cleaned_data['subject']
            message = form.cleaned_data['message']

            email_body = f"Name: {name}\nEmail: {email}\n\n{message}"

            send_mail(
                subject,
                email_body,
                settings.DEFAULT_FROM_EMAIL,
                [settings.EMAIL_HOST_USER],
                fail_silently=False,
            )

            return redirect('contact_success')
        else:
            print(form.errors)
    else:
        form = ContactForm()

    return render(request, 'main/contact.html', {'form': form})

def contact_success(request):
    return render(request, 'main/contact_success.html')

@csrf_exempt  # For testing; ensure proper CSRF handling in production
def run_simulation(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            floor_width = data.get('floor_width')
            floor_length = data.get('floor_length')
            target_ppfd = data.get('target_ppfd')

            if floor_width is None or floor_length is None or target_ppfd is None:
                return JsonResponse(
                    {'error': 'Missing required parameters: floor_width, floor_length, target_ppfd'},
                    status=400
                )

            # Updated call: use keyword names that match the function's signature.
            simulation_results = run_ml_simulation(
                floor_width_ft=floor_width,
                floor_length_ft=floor_length,
                target_ppfd=target_ppfd
            )
            return JsonResponse(simulation_results)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=400)

def simulation_view(request):
    return render(request, 'main/simulation.html')

def agenticai(request):
    return render(request, 'main/agenticai.html')

def cooling_system(request):
    return render(request, 'main/cooling_system.html')

def cea(request):
    return render(request, 'main/cea.html')

def lighting_simulation(request):
    return render(request, 'main/lighting_simulation.html')

def upfp(request):
    return render(request, 'main/blog/upfp.html')

def dialux_simulation(request):
    return render(request, 'main/dialux_simulation.html')

@login_required  # ensure only authorized users can access
def blog_admin_panel(request):
    posts = BlogPost.objects.all().order_by('-created_at')
    return render(request, 'main/blog_admin_panel.html', {'posts': posts})

@require_POST
@login_required
def toggle_like(request):
    post_slug = request.POST.get('slug')
    if not post_slug:
        return HttpResponseBadRequest("Missing post slug.")
    
    post = get_object_or_404(BlogPost, slug=post_slug)
    like, created = Like.objects.get_or_create(post=post, user=request.user)
    if not created:
        # Already liked, so remove like
        like.delete()
        liked = False
    else:
        liked = True

    return JsonResponse({
        'liked': liked,
        'like_count': post.likes.count()
    })

@require_POST
@login_required
def log_share(request):
    post_slug = request.POST.get('slug')
    platform = request.POST.get('platform')
    if not post_slug or not platform:
        return HttpResponseBadRequest("Missing parameters.")
    
    post = get_object_or_404(BlogPost, slug=post_slug)
    share = Share.objects.create(post=post, user=request.user, platform=platform)
    
    return JsonResponse({
        'share_id': share.id,
        'platform': share.platform,
        'share_count': post.shares.count()
    })

import mammoth

@login_required
def create_blog_post(request):
    if request.method == 'POST':
        form = BlogPostForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save(commit=False)
            doc_file = form.cleaned_data.get('content_doc')
            if doc_file:
                file_bytes = doc_file.read()
                file_io = io.BytesIO(file_bytes)
                style_map = """
p[style-name='BoldBlueTitle'] => p[style='margin: 0 !important; line-height: 1.4 !important;']
r[style-name='BoldBlueTitle'] => span[style='display: block; color: #0073b1; font-weight: bold; font-size: 26px; margin: 0 !important; line-height: 1.4 !important;']

p[style-name='BoldBlue'] => p[style='color: #0073b1; font-weight: bold; font-family: "Avenir Next", "Helvetica Neue", sans-serif; font-size: 20px; margin-bottom: 5px;']
r[style-name='BoldBlue'] => span.bold-blue[style='color: #0073b1; font-weight: bold; font-size: 20px;']

p[style-name='TitleText'] => p[style='font-family: "Avenir Next", "Helvetica Neue", sans-serif; font-size: 20px; margin-top: 0px !important;']
r[style-name='TitleText'] => span.title-text[style='font-family: "Avenir Next", "Helvetica Neue", sans-serif; font-size: 20px;']

p[style-name='SectionText'] => p[style='margin: 0 !important; line-height: 1.4 !important;']
r[style-name='SectionText'] => span[style='display: block; font-family: "Avenir Next", "Helvetica Neue", sans-serif; font-size: 20px; margin: 0 !important; line-height: 1.4 !important;']

p[style-name='BlueSectionText'] => p[style='color: #0073b1; font-family: "Avenir Next", "Helvetica Neue", sans-serif; font-size: 20px; margin-top: 0px !important;']
r[style-name='BlueSectionText'] => span.blue-section-text[style='color: #0073b1; font-family: "Avenir Next", "Helvetica Neue", sans-serif; font-size: 20px;']

p[style-name='BoldBlackTitle'] => p[style='font-family: "Avenir Next", "Helvetica Neue", sans-serif; font-weight: bold; font-size: 20px; margin-top: 0px !important;']
r[style-name='BoldBlackTitle'] => span.bold-black-title[style='font-family: "Avenir Next", "Helvetica Neue", sans-serif; font-weight: bold; font-size: 20px;']
"""
                result = mammoth.convert_to_html(file_io, style_map=style_map)
                instance.content = result.value
                from django.core.files.base import ContentFile
                instance.content_doc = ContentFile(file_bytes, name=doc_file.name)
            instance.created_by = request.user
            instance.save()
            return redirect('blog_admin_panel')
    else:
        form = BlogPostForm()
    return render(request, 'main/blog_create.html', {'form': form})

@login_required
def edit_blog_post(request, slug):
    post = get_object_or_404(BlogPost, slug=slug)
    if request.method == 'POST':
        form = BlogPostForm(request.POST, request.FILES, instance=post)
        if form.is_valid():
            # Only override content if a new file was uploaded.
            if 'content_doc' in request.FILES:
                doc_file = form.cleaned_data.get('content_doc')
                if doc_file:
                    style_map = "p[style-name='SubText'] => p[style='color: #0078BE']"
                    result = mammoth.convert_to_html(doc_file, style_map=style_map)
                    html = result.value
                    form.instance.content = html
            # Otherwise, leave the CKEditor content as-is.
            form.save()
            return redirect('blog_admin_panel')
    else:
        form = BlogPostForm(instance=post)
    return render(request, 'main/blog_edit.html', {'form': form, 'post': post})


@login_required
def delete_blog_post(request, slug):
    post = get_object_or_404(BlogPost, slug=slug)
    if request.method == 'POST':
        post.delete()
        return redirect('blog_admin_panel')
    return render(request, 'main/blog_confirm_delete.html', {'post': post})

def blog_index(request):
    if request.user.is_superuser:
        posts = BlogPost.objects.all().order_by('-created_at')
    else:
        posts = BlogPost.objects.filter(approved=True).order_by('-created_at')
    return render(request, 'main/blog_index.html', {'posts': posts})

def blog_detail(request, slug):
    post = get_object_or_404(BlogPost, slug=slug)
    return render(request, 'main/blog_detail.html', {'post': post})

def is_approver(user):
    return user.username in ['austin22', 'amrouse123']

@user_passes_test(is_approver)
def approve_blog_post(request, slug):
    post = get_object_or_404(BlogPost, slug=slug)
    # Only allow approval if not already approved.
    if not post.approved:
        post.approved = True
        post.save()
        # Send notification email to the creator.
        send_mail(
            "Your Blog Post Has Been Approved",
            f"Hello, your blog post titled '{post.title}' has been approved and is now live on the website.",
            "austin@luminousphotonics.com",
            [post.created_by.email],
            fail_silently=False,
        )
    return redirect('blog_admin_panel')


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
from .blog_system import BlogPost, Comment, Like, Share  # our model file
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.http import require_POST
from .forms import BlogPostForm
from django.http import HttpResponse
from .blog_system import BlogPost
import mammoth
from django.contrib.auth.decorators import user_passes_test

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
def add_comment(request):
    post_slug = request.POST.get('slug')
    comment_text = request.POST.get('comment')
    if not post_slug or not comment_text:
        return HttpResponseBadRequest("Missing parameters.")
    
    post = get_object_or_404(BlogPost, slug=post_slug)
    comment = Comment.objects.create(post=post, user=request.user, text=comment_text)
    
    return JsonResponse({
        'comment_id': comment.id,
        'comment_text': comment.text,
        'username': comment.user.username,
        'created_at': comment.created_at.strftime("%Y-%m-%d %H:%M")
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
            # Process Word document conversion if uploaded...
            doc_file = form.cleaned_data.get('content_doc')
            if doc_file:
                style_map = [
                    "p[style-name='BlueText'] => p[style='color: #0073b1']",
                ]
                result = mammoth.convert_to_html(doc_file, style_map=style_map)
                html = result.value
                form.instance.content = html
            # Set the creator to the currently logged in user
            form.instance.created_by = request.user
            form.save()
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
            doc_file = form.cleaned_data.get('content_doc')
            if doc_file:
                style_map = "p[style-name='SubText'] => p[style='color: #0078BE']"
                result = mammoth.convert_to_html(doc_file, style_map=style_map)
                html = result.value
                form.instance.content = html
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

def is_austin(user):
    return user.username == 'austin22'

@user_passes_test(is_austin)
def approve_blog_post(request, slug):
    post = get_object_or_404(BlogPost, slug=slug)
    # Only allow approval if not already approved.
    if not post.approved:
        post.approved = True
        post.save()
        # Send notification email to the creator (assumed to be 'amrouse123')
        send_mail(
            "Your Blog Post Has Been Approved",
            f"Hello, your blog post titled '{post.title}' has been approved and is now live on the website.",
            "austin@luminousphotonics.com",
            [post.created_by.email],
            fail_silently=False,
        )
    return redirect('blog_admin_panel')

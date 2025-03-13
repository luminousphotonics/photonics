from django.shortcuts import render, redirect, get_object_or_404
from django.core.mail import send_mail, EmailMessage
from django.conf import settings
from django.contrib import messages
from .forms import ContactForm
import os
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .ml_simulation import run_ml_simulation  # Import from ml_simulation.py
from django.contrib.auth.decorators import login_required
from .blog_system import BlogPost, Like, Share  # our model file
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.http import require_POST
from .forms import BlogPostForm
from .blog_system import BlogPost
import mammoth
from django.contrib.auth.decorators import user_passes_test
import io
from .ml_simulation import run_ml_simulation
import queue
import threading
import uuid
from django_redis import get_redis_connection

# Use Redis keys with a prefix for clarity.
REDIS_PREFIX = "simulation_job"

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

@csrf_exempt
def run_simulation(request):
    if request.method in ['GET', 'POST']:
        try:
            if request.method == 'POST':
                data = json.loads(request.body)
                print("DEBUG: Received POST data:", data)
                floor_width = data.get('floor_width')
                floor_length = data.get('floor_length')
                target_ppfd = data.get('target_ppfd')
                floor_height = data.get('floor_height', 3.0)
            else:
                print("DEBUG: GET parameters:", request.GET)
                floor_width = request.GET.get('floor_width')
                floor_length = request.GET.get('floor_length')
                target_ppfd = request.GET.get('target_ppfd')
                floor_height = request.GET.get('floor_height', 3.0)

            if floor_width is None or floor_length is None or target_ppfd is None:
                return JsonResponse({'error': 'Missing required parameters'}, status=400)

            simulation_results = run_ml_simulation(
                floor_width_ft=floor_width,
                floor_length_ft=floor_length,
                target_ppfd=target_ppfd,
                floor_height_ft=floor_height
            )
            print("DEBUG: Simulation completed. Results:", simulation_results)
            return JsonResponse(simulation_results)
        except Exception as e:
            print("ERROR in run_simulation:", e)
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
def toggle_like(request):
    post_slug = request.POST.get('slug')
    if not post_slug:
        return HttpResponseBadRequest("Missing post slug.")
    
    post = get_object_or_404(BlogPost, slug=post_slug)
    
    # For authenticated users, use the database
    if request.user.is_authenticated:
        like, created = Like.objects.get_or_create(post=post, user=request.user)
        if not created:
            like.delete()
            liked = False
        else:
            liked = True
    else:
        # For anonymous users, track likes in the session
        liked_posts = request.session.get('liked_posts', [])
        if post_slug in liked_posts:
            liked_posts.remove(post_slug)
            liked = False
        else:
            liked_posts.append(post_slug)
            liked = True
        request.session['liked_posts'] = liked_posts

    # Note: The like_count here only reflects database likes.
    # To incorporate anonymous likes, you'll need additional handling.
    return JsonResponse({
        'liked': liked,
        'like_count': post.likes.count()
    })

@require_POST
def log_share(request):
    post_slug = request.POST.get('slug')
    platform = request.POST.get('platform')
    if not post_slug or not platform:
        return HttpResponseBadRequest("Missing parameters.")
    
    post = get_object_or_404(BlogPost, slug=post_slug)
    
    if request.user.is_authenticated:
        share = Share.objects.create(post=post, user=request.user, platform=platform)
    else:
        # For anonymous users, track shares in the session.
        shares = request.session.get('shares', {})
        share_key = f"{post_slug}_{platform}"
        shares[share_key] = shares.get(share_key, 0) + 1
        request.session['shares'] = shares
        share = None  # No database record for anonymous share
    
    # Calculate total share count: combine DB records with session count.
    session_shares = request.session.get('shares', {})
    anonymous_count = session_shares.get(f"{post_slug}_{platform}", 0)
    db_count = post.shares.filter(platform=platform).count()
    total_count = db_count + anonymous_count
    
    response_data = {
        'platform': platform,
        'share_count': total_count
    }
    if share:
        response_data['share_id'] = share.id
    return JsonResponse(response_data)

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

def miro(request):
    return render(request, 'main/miro.html')

# ----------------------------
# New Endpoints for Asynchronous Heavy Payload Transfer
# ----------------------------

# Global job store to track each simulation job.
# Each job is stored as a dictionary with keys: 'progress', 'status', and 'result'
simulation_job_store = {}

@csrf_exempt
def simulation_start(request):
    """
    Starts a simulation asynchronously using Redis to store progress.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            floor_width = float(data.get('floor_width', 12))
            floor_length = float(data.get('floor_length', 12))
            target_ppfd = float(data.get('target_ppfd', 1250))
            floor_height = float(data.get('floor_height', 3.0))
        except Exception as e:
            return JsonResponse({'error': 'Invalid parameters: ' + str(e)}, status=400)
        
        job_id = str(uuid.uuid4())
        redis_conn = get_redis_connection("default")
        # Initialize keys in Redis.
        redis_conn.delete(f"{REDIS_PREFIX}:{job_id}:progress")
        redis_conn.set(f"{REDIS_PREFIX}:{job_id}:status", "running")
        redis_conn.delete(f"{REDIS_PREFIX}:{job_id}:result")
        
        # Define a callback that pushes progress messages into Redis.
        def progress_callback(message: str):
            redis_conn.rpush(f"{REDIS_PREFIX}:{job_id}:progress", message)
        
        def run_job():
            try:
                result = run_ml_simulation(
                    floor_width_ft=floor_width,
                    floor_length_ft=floor_length,
                    target_ppfd=target_ppfd,
                    floor_height=floor_height,
                    progress_callback=progress_callback,
                    side_by_side=True
                )
                redis_conn.set(f"{REDIS_PREFIX}:{job_id}:result", json.dumps(result))
                redis_conn.set(f"{REDIS_PREFIX}:{job_id}:status", "done")
            except Exception as e:
                progress_callback("ERROR: " + str(e))
                redis_conn.set(f"{REDIS_PREFIX}:{job_id}:status", "done")
                redis_conn.set(f"{REDIS_PREFIX}:{job_id}:result", json.dumps({"error": str(e)}))
        
        thread = threading.Thread(target=run_job)
        thread.start()
        return JsonResponse({'job_id': job_id})
    return JsonResponse({'error': 'POST request required.'}, status=400)

def simulation_progress(request, job_id):
    """
    Returns current progress and status from Redis for the given job.
    """
    redis_conn = get_redis_connection("default")
    status = redis_conn.get(f"{REDIS_PREFIX}:{job_id}:status")
    if status is None:
        return JsonResponse({'error': 'Job not found'}, status=404)
    # Decode bytes to string.
    status = status.decode("utf-8")
    # Get all progress messages.
    progress = redis_conn.lrange(f"{REDIS_PREFIX}:{job_id}:progress", 0, -1)
    # Convert each item from bytes to string.
    progress = [msg.decode("utf-8") for msg in progress]
    return JsonResponse({
        'status': status,
        'progress': progress
    })

def simulation_result(request, job_id):
    """
    Returns the final simulation result from Redis once the job is complete.
    """
    redis_conn = get_redis_connection("default")
    status = redis_conn.get(f"{REDIS_PREFIX}:{job_id}:status")
    if status is None or status.decode("utf-8") != "done":
        return JsonResponse({'error': 'Job not found or not complete.'}, status=404)
    result_data = redis_conn.get(f"{REDIS_PREFIX}:{job_id}:result")
    if result_data is None:
        return JsonResponse({'error': 'Result not available.'}, status=404)
    result = json.loads(result_data.decode("utf-8"))
    return JsonResponse(result)

def simulation_status(request, job_id):
    """
    Returns only the status of the simulation job ("running" or "done") from Redis.
    """
    redis_conn = get_redis_connection("default")
    status = redis_conn.get(f"simulation_job:{job_id}:status")
    if status is None:
        return JsonResponse({'error': 'Job not found'}, status=404)
    return JsonResponse({'status': status.decode("utf-8")})


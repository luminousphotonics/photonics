from django.shortcuts import render, redirect
from django.core.mail import send_mail, EmailMessage
from django.conf import settings
from django.contrib import messages
from .forms import ContactForm
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .ml_simulation import run_ml_simulation  # Import from ml_simulation.py

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

from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.conf import settings
from django.contrib import messages
from .forms import ContactForm
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .lighting_optimization import run_simulation as run_simulation_optimization

def index(request):
    return render(request, 'main/index.html')

def technology(request):
    return render(request, 'main/technology.html')

def applications(request):
    return render(request, 'main/applications.html')

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
                email,  # From email (form's email)
                [settings.EMAIL_HOST_USER],  # To email (your email from settings)
                fail_silently=False,  # Set to True in production after debugging
            )

            return redirect('contact_success')

        else:
            print(form.errors)

    else:
        form = ContactForm()

    return render(request, 'main/contact.html', {'form': form}) # Assuming your app is named 'main'

def contact_success(request):
    return render(request, 'main/contact_success.html')  # Correct path for 'main' app

@csrf_exempt  # For testing; handle CSRF appropriately in production
def run_simulation(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            floor_width = data.get('floor_width')
            floor_length = data.get('floor_length')
            target_ppfd = data.get('target_ppfd')
            # floor_height isn’t used by the optimization directly but might be used elsewhere
            perimeter_reflectivity = data.get('perimeter_reflectivity', 0.3)

            # Validate data if needed

            # Use the aliased function
            simulation_results = run_simulation_optimization(
                floor_width=floor_width,
                floor_length=floor_length,
                target_ppfd=target_ppfd,
                perimeter_reflectivity=perimeter_reflectivity,
                verbose=False  # Adjust verbosity as needed
            )

            return JsonResponse(simulation_results)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=400)

def simulation_view(request):
    return render(request, 'main/simulation.html')

def agenticai(request):
    return render(request, 'main/agenticai.html')
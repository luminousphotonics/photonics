from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.conf import settings
from django.contrib import messages
from .forms import ContactForm
import os

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
            message = form.cleaned_data['message']

            subject = f'New Contact Form Submission from {name}'
            full_message = f'Name: {name}\nEmail: {email}\n\nMessage:\n{message}'

            try:
                send_mail(
                    subject,
                    full_message,
                    email,  # From email
                    [os.environ.get('EMAIL_HOST_USER')],  # To email
                    fail_silently=False,
                )
                messages.success(request, 'Your message has been sent successfully!')
                return redirect('contact')
            except Exception as e:
                messages.error(request, 'There was an error sending your message. Please try again later.')
    else:
        form = ContactForm()

    return render(request, 'main/contact.html', {'form': form})
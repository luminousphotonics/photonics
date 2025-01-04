from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.conf import settings
from django.contrib import messages

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
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = request.POST.get('message')

        if name and email and message:
            try:
                send_mail(
                    f'Contact Form Submission from {name}',  # Subject
                    message,  # Message body
                    email,  # From email (user's email)
                    [settings.EMAIL_HOST_USER],  # To email (your company email)
                    fail_silently=False,
                )
                messages.success(request, 'Your message has been sent successfully!')
                return redirect('contact')  # Redirect to the contact page after sending
            except Exception as e:
                messages.error(request, f'An error occurred: {e}')
                return redirect('contact')
        else:
            messages.error(request, 'Please fill in all fields.')
            return redirect('contact')

    return render(request, 'main/contact.html')
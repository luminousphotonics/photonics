# myapp/forms.py
from django import forms
from .blog_system import BlogPost

class ContactForm(forms.Form):
    name = forms.CharField(max_length=100, widget=forms.TextInput(attrs={'placeholder': 'Your Name'}))
    email = forms.EmailField(widget=forms.EmailInput(attrs={'placeholder': 'Your Email'}))
    subject = forms.CharField(max_length=100, widget=forms.TextInput(attrs={'placeholder': 'Subject'}))
    message = forms.CharField(widget=forms.Textarea(attrs={'placeholder': 'Your Message'}))

class BlogPostForm(forms.ModelForm):
    content_doc = forms.FileField(
        required=False,
        help_text="Upload a Word document (.docx) for the post content."
    )

    class Meta:
        model = BlogPost
        fields = ['title', 'slug', 'content', 'title_image']  # 'content' will eventually hold the HTML

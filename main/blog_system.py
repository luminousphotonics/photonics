from django.db import models
from django.contrib.auth import get_user_model
from django_ckeditor_5.fields import CKEditor5Field  # Updated import

User = get_user_model()

class BlogPost(models.Model):
    title = models.CharField(max_length=255)
    slug = models.SlugField(unique=True)
    content = CKEditor5Field('Content', config_name='default', blank=True, null=True)
    title_image = models.ImageField(upload_to='blog_images/')
    content_doc = models.FileField(upload_to='blog_docs/', blank=True, null=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name="blog_posts")
    approved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # New fields for SEO
    meta_keywords = models.CharField(
        max_length=255, blank=True, null=True,
        help_text="Enter 3-5 keywords separated by commas"
    )
    meta_title = models.CharField(
        max_length=80, blank=True, null=True,
        help_text="SEO title tag for browser tab (50-60 characters)"
    )
    meta_description = models.TextField(
        blank=True, null=True,
        help_text="Meta description for SERP (a brief page summary)"
    )

    def __str__(self):
        return self.title


class Like(models.Model):
    post = models.ForeignKey(BlogPost, related_name="likes", on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('post', 'user')

    def __str__(self):
        return f"Like by {self.user} on {self.post}"

class Share(models.Model):
    PLATFORM_CHOICES = [
        ('linkedin', 'LinkedIn'),
        ('facebook', 'Facebook'),
        ('twitter', 'Twitter'),
    ]
    post = models.ForeignKey(BlogPost, related_name="shares", on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    platform = models.CharField(max_length=50, choices=PLATFORM_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Share on {self.platform} by {self.user} for {self.post}"

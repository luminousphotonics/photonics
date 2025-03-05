from django.db import models
from django.contrib.auth.models import User
from django.contrib.auth import get_user_model
User = get_user_model()

class BlogPost(models.Model):
    title = models.CharField(max_length=255)
    slug = models.SlugField(unique=True)
    content = models.TextField()
    title_image = models.ImageField(upload_to='blog_images/')
    created_by = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="blog_posts",
        default=2  # manually setting default to 2 (assuming that's amrouse123)
    )
    approved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title

class Comment(models.Model):
    post = models.ForeignKey(BlogPost, related_name="comments", on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Comment by {self.user} on {self.post}"

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

// blog_interactions.js

// Helper function to get CSRF token from cookies
function getCSRFToken() {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
      const cookies = document.cookie.split(';');
      for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        if (cookie.substring(0, 10) === 'csrftoken=') {
          cookieValue = decodeURIComponent(cookie.substring(10));
          break;
        }
      }
    }
    return cookieValue;
  }
  
  // Toggle like functionality
  function toggleLike(slug) {
    fetch("/ajax/toggle-like/", {
      method: "POST",
      headers: {
        "X-CSRFToken": getCSRFToken(),
        "Content-Type": "application/x-www-form-urlencoded"
      },
      body: "slug=" + slug
    })
    .then(response => response.json())
    .then(data => {
      // Update the like count on the page
      const likeCount = document.getElementById("like-count");
      if (likeCount) {
        likeCount.innerText = data.like_count;
      }
    })
    .catch(err => console.error(err));
  }
  
  // Add comment functionality
  function addComment(slug) {
    const commentInput = document.getElementById("comment-input");
    const commentText = commentInput.value.trim();
    if (!commentText) return;
    fetch("/ajax/add-comment/", {
      method: "POST",
      headers: {
        "X-CSRFToken": getCSRFToken(),
        "Content-Type": "application/x-www-form-urlencoded"
      },
      body: "slug=" + slug + "&comment=" + encodeURIComponent(commentText)
    })
    .then(response => response.json())
    .then(data => {
      // Append the new comment to the comment list
      const commentList = document.getElementById("comment-list");
      if (commentList) {
        const newComment = document.createElement("div");
        newComment.innerText = data.username + ": " + data.comment_text;
        commentList.appendChild(newComment);
        commentInput.value = "";
      }
    })
    .catch(err => console.error(err));
  }
  
  function logShare(slug, platform) {
    // Log the share via AJAX
    fetch("/ajax/log-share/", {
      method: "POST",
      headers: {
        "X-CSRFToken": getCSRFToken(),
        "Content-Type": "application/x-www-form-urlencoded"
      },
      body: "slug=" + slug + "&platform=" + platform
    })
    .then(response => response.json())
    .then(data => {
      console.log("Logged share on " + data.platform + ". Total shares: " + data.share_count);
    })
    .catch(err => console.error(err));
  
    // Then open the sharing URL
    const currentUrl = encodeURIComponent(window.location.href);
    let shareUrl = "";
  
    switch (platform) {
      case "linkedin":
        shareUrl = "https://www.linkedin.com/shareArticle?mini=true&url=" + currentUrl;
        break;
      case "facebook":
        shareUrl = "https://www.facebook.com/sharer/sharer.php?u=" + currentUrl;
        break;
      case "x": // formerly Twitter
        shareUrl = "https://twitter.com/intent/tweet?url=" + currentUrl;
        break;
      default:
        return;
    }
  
    window.open(shareUrl, '_blank', 'width=600,height=400');
  }
  
  
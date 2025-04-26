// scripts.js

// PDF opener
function openPDF() {
    const url = document.getElementById('pdfButton2').dataset.pdfUrl;
    window.open(url, '_blank');
  }
  
// ANimate BG
function initAnimatedBG(bgId, canvasId) {
    const section = document.getElementById(bgId);
    const canvas  = document.getElementById(canvasId);
    const ctx     = canvas.getContext('2d');
    let width, height, points, target;
  
    function onResize() {
      width  = section.clientWidth;
      height = section.clientHeight;
      canvas.width  = width;
      canvas.height = height;
    }
  
    function dist(a, b) {
      return (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
    }
  
    function drawLines(p) {
      if (!p.active) return;
      p.closest.forEach(c => {
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(c.x, c.y);
        ctx.strokeStyle = `rgba(232,185,36,${p.active})`;
        ctx.stroke();
      });
    }
  
    function drawCircle(c) {
      if (!c.active) return;
      ctx.beginPath();
      ctx.arc(c.pos.x, c.pos.y, c.radius, 0, 2 * Math.PI);
      ctx.fillStyle = `rgba(232,185,36,${c.active})`;
      ctx.fill();
    }
  
    function animate() {
      ctx.clearRect(0, 0, width, height);
      points.forEach(p => {
        const d = dist(target, p);
        if      (d <  4000)  { p.active = 0.3;  p.circle.active = 0.6; }
        else if (d < 20000)  { p.active = 0.1;  p.circle.active = 0.3; }
        else if (d < 40000)  { p.active = 0.02; p.circle.active = 0.1; }
        else                 { p.active = 0;    p.circle.active = 0;   }
        drawLines(p);
        drawCircle(p.circle);
      });
      requestAnimationFrame(animate);
    }
  
    function init() {
      onResize();
      target = { x: width / 2, y: height / 2 };
      points = [];
      const gridX = Math.floor(width / 20);
      const gridY = Math.floor(height / 20);
  
      for (let i = 0; i < gridX; i++) {
        for (let j = 0; j < gridY; j++) {
          const px = i * 20 + Math.random() * 20;
          const py = j * 20 + Math.random() * 20;
          points.push({ x: px, y: py, originX: px, originY: py });
        }
      }
  
      points.forEach(p => {
        p.closest = points
          .filter(o => o !== p)
          .sort((a, b) => dist(p, a) - dist(p, b))
          .slice(0, 5);
        p.circle = { pos: p, radius: 2 + Math.random() * 2, active: 0 };
        gsap.to(p, {
          duration: 1 + Math.random(),
          x: p.originX - 50 + Math.random() * 100,
          y: p.originY - 50 + Math.random() * 100,
          ease: 'circ.inOut',
          repeat: -1,
          yoyo: true
        });
      });
  
      window.addEventListener('resize', onResize);
      window.addEventListener('mousemove', e => {
        target.x = e.clientX;
        target.y = e.clientY;
      });
  
      animate();
    }
  
    init();
  }
  
  document.addEventListener('DOMContentLoaded', () => {
    initAnimatedBG('animated-bg-1', 'animated-canvas-1');
    initAnimatedBG('animated-bg-2', 'animated-canvas-2');
  });
  

// Lazy Loading for Background Images
const lazyBackgrounds = document.querySelectorAll(".lazy-load-bg");

const lazyLoadBackground = (target) => {
    const io = new IntersectionObserver((entries, observer) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                const bgUrl = entry.target.getAttribute("data-bg");
                entry.target.style.backgroundImage = `url(${bgUrl})`;
                entry.target.classList.add("visible"); // Add visible class for fade-in effect
                observer.unobserve(entry.target); // Stop observing after loading
            }
        });
    });

    io.observe(target);
};

lazyBackgrounds.forEach(lazyLoadBackground);

// Lazy Loading for Images and Videos
const lazyMedia = document.querySelectorAll(".lazy-load");

const lazyLoad = (target) => {
    const io = new IntersectionObserver((entries, observer) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                const element = entry.target;
                if (element.tagName === "IMG") {
                    element.src = element.dataset.src;
                    element.classList.add("visible"); // Add visible class for fade-in effect
                } else if (element.tagName === "VIDEO") {
                    const source = element.querySelector("source");
                    source.src = source.dataset.src;
                    element.load();
                    element.classList.add("visible"); // Add visible class for fade-in effect
                }
                observer.unobserve(element); // Stop observing after loading
            }
        });
    });

    io.observe(target);
};

lazyMedia.forEach(lazyLoad);

// Initialize Slick Carousel
$(document).ready(function () {
    $('.carousel').slick({
        dots: true,
        infinite: true,
        speed: 500,
        slidesToShow: 4,
        slidesToScroll: 1,
        adaptiveHeight: false,
        autoplay: true,
        autoplaySpeed: 3000,
        arrows: true,
        pauseOnHover: false,
        centerMode: true,
        variableWidth: false, // Ensure this is false
    });

    // Adjust Carousel Height
    function adjustCarouselHeight() {
        var maxHeight = 0;
        $('.carousel .slick-slide').each(function () {
            if ($(this).height() > maxHeight) {
                maxHeight = $(this).height();
            }
        });
        $('.carousel').height(maxHeight);
    }

    // Adjust height on initialization
    adjustCarouselHeight();

    // Adjust height on window resize
    $(window).on('resize', function () {
        adjustCarouselHeight();
    });
});

function toggleLiterature() {
    var literatureDiv = document.getElementById('academic-literature');
    if (literatureDiv.style.display === "none" || literatureDiv.style.display === "") {
      literatureDiv.style.display = "block";
    } else {
      literatureDiv.style.display = "none";
    }
  }

// Scroll Animations for Hidden Elements
const animElements = document.querySelectorAll('.hidden, .hero, .feature, .info-box, .cta');

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
        } else {
            entry.target.classList.remove('visible'); // Remove class when out of view
        }
    });
}, { threshold: 0.1 });

animElements.forEach(el => observer.observe(el));

// Back to Top Button Functionality
const backToTopBtn = document.getElementById('backToTopBtn');

window.addEventListener('scroll', function () {
    if (window.scrollY > 300) {
        backToTopBtn.classList.add('show');
    } else {
        backToTopBtn.classList.remove('show');
    }
});

backToTopBtn.addEventListener('click', function (e) {
    e.preventDefault();
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
});

// Home Button Functionality
const homeButton = document.getElementById('homeButton');

window.addEventListener('scroll', function () {
    if (window.scrollY > 300) {
        homeButton.classList.add('show');
    } else {
        homeButton.classList.remove('show');
    }
});

homeButton.addEventListener('click', function (e) {
    e.preventDefault();
    window.location.href = "/";
});

// Info Box Interactivity with Learn More Buttons
const infoBoxes = document.querySelectorAll('.info-box');
infoBoxes.forEach(box => {
    box.addEventListener('click', function (event) {
        // If click is on the Learn More button, handle navigation
        if (event.target.classList.contains('learn-more-button')) {
            event.stopPropagation(); // Prevent parent click handler from firing
            const dataInfo = this.getAttribute('data-info');
            let targetUrl = '';

            switch (dataInfo) {
                case 'agentic-ai':
                    targetUrl = document.getElementById('agentic-ai-link').href;
                    break;
                case 'lighting-simulations':
                    targetUrl = document.getElementById('lighting-simulations-link').href;
                    break;
                case 'ml-optimization':
                    targetUrl = document.getElementById('ml-optimization-link').href;
                    break;
                case 'cooling-system':
                    targetUrl = document.getElementById('cooling-system-link').href;
                    break;
            }

            window.location.href = targetUrl;
        } else {
            // Check if the clicked box is already active
            const isActive = this.classList.contains('active');

            // Close other info boxes only if the clicked box is not active
            if (!isActive) {
                infoBoxes.forEach(otherBox => {
                    if (otherBox !== this) {
                        otherBox.classList.remove('active');
                    }
                });
            }

            // Toggle active state on the clicked box
            this.classList.toggle('active');
        }
    });
});

// Intersection Observer for Info Panel
const infoPanel = document.querySelector('.info-panel');

const infoPanelObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            infoPanelObserver.unobserve(entry.target); // Stop observing once visible
        }
    });
}, { threshold: 0.2 }); // Adjust threshold as needed

infoPanelObserver.observe(infoPanel);

(() => {
  const canvas = document.getElementById('photonics-canvas');
  const ctx    = canvas.getContext('2d');
  let width, height, time = 0;

  function resize() {
    width = canvas.width  = window.innerWidth;
    height= canvas.height = window.innerHeight;
  }
  window.addEventListener('resize', resize);
  resize();

  // generate some drifting spark particles
  const sparks = Array.from({length:120}, () => ({
    x: Math.random()*width,
    y: Math.random()*height,
    size: 1+Math.random()*2,
    speed: 0.2+Math.random()*0.5,
    hue: Math.random()*360
  }));

  function draw() {
    ctx.clearRect(0,0,width,height);
    ctx.globalCompositeOperation = 'lighter';

    // draw two sine-ribbon layers
    [0.002, 0.0012].forEach((freq, idx) => {
      const amp = (idx+1)*40 + 20;
      const speed = (idx+1)*0.3;
      ctx.lineWidth = 3 + idx*2;
      const grad = ctx.createLinearGradient(0,0,width,0);
      grad.addColorStop(0, `hsla(${(time*10)%360},100%,50%,0.4)`);
      grad.addColorStop(1, `hsla(${(time*10+180)%360},100%,50%,0.4)`);
      ctx.strokeStyle = grad;
      ctx.beginPath();
      for (let x=0; x<=width; x+=10) {
        const y = height/2 
          + Math.sin((x*freq) + time*speed) * amp
          + Math.cos((x*freq*1.3) + time*speed*0.7) * (amp/2);
        idx===0
          ? ctx.lineTo(x, y)
          : ctx.lineTo(x, y + 100);
      }
      ctx.stroke();
    });

    // draw sparks
    sparks.forEach(s => {
      s.y -= s.speed;
      if (s.y < -10) {
        s.y = height+10;
        s.x = Math.random()*width;
      }
      ctx.fillStyle = `hsla(${(s.hue + time*5)%360},100%,70%,0.6)`;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.size, 0, Math.PI*2);
      ctx.fill();
    });

    time += 0.5;
    requestAnimationFrame(draw);
  }

  draw();
})();

//Image Container Slide-In
document.addEventListener("DOMContentLoaded", () => {
  const obs = new IntersectionObserver((entries, o) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add("visible");
        o.unobserve(e.target);
      }
    });
  }, { threshold: 0.3 });

  document.querySelectorAll(".slide-images-container")
          .forEach(el => obs.observe(el));
});

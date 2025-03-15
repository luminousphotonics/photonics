import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

# Define layers for "Diamond: 61" pattern
layers = [
    [(0, 0)],
    [(-1, 0), (1, 0), (0, -1), (0, 1)],
    [(-1, -1), (1, -1), (-1, 1), (1, 1),
     (-2, 0), (2, 0), (0, -2), (0, 2)],
    [(-2, -1), (2, -1), (-2, 1), (2, 1),
     (-1, -2), (1, -2), (-1, 2), (1, 2),
     (-3, 0), (3, 0), (0, -3), (0, 3)],
    [(-2, -2), (2, -2), (-2, 2), (2, 2),
     (-3, -1), (3, -1), (-3, 1), (3, 1),
     (-1, -3), (1, -3), (-1, 3), (1, 3),
     (-4, 0), (4, 0), (0, -4), (0, 4)],
    [(-3, -2), (3, -2), (-3, 2), (3, 2),
     (-2, -3), (2, -3), (-2, 3), (2, 3),
     (-4, -1), (4, -1), (-4, 1), (4, 1),
     (-1, -4), (1, -4), (-1, 4), (1, 4),
     (-5, 0), (5, 0), (0, -5), (0, 5)]
]

# Array dimensions and spacing
light_array_width_ft = 10.0
light_array_height_ft = 10.0
spacing_x = light_array_width_ft / 7.2
spacing_y = light_array_height_ft / 7.2

center_x, center_y = 0.0, 0.0
theta = math.radians(45)  # Rotate by 45 degrees

# Build LED list with assigned intensities (center LED from layer 0)
leds = []
leds.append({'pos': (center_x, center_y), 'I': 1})

for layer_index, layer in enumerate(layers):
    if layer_index == 0:
        continue  # Center already added
    intensity = 1 + layer_index  # Increase intensity for outer layers
    for dot in layer:
        # Apply spacing and rotation
        x_offset = spacing_x * dot[0]
        y_offset = spacing_y * dot[1]
        rotated_x = x_offset * math.cos(theta) - y_offset * math.sin(theta)
        rotated_y = x_offset * math.sin(theta) + y_offset * math.cos(theta)
        pos = (center_x + rotated_x, center_y + rotated_y)
        leds.append({'pos': pos, 'I': intensity})

# Parameters for target plane (above the LED array)
h = 5.0   # height above the array
epsilon = 1e-6  # avoid division by zero

# Create grid for the target plane where PPFD is computed
grid_range = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(grid_range, grid_range)
PPFD = np.zeros_like(X)

# Compute PPFD from each LED using a combined inverse square and cosine drop-off model
for led in leds:
    led_x, led_y = led['pos']
    I_led = led['I']
    dx = X - led_x
    dy = Y - led_y
    distance = np.sqrt(dx**2 + dy**2 + h**2) + epsilon
    PPFD += I_led * h / (distance**3)

# 3D visualization of the PPFD surface
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, PPFD, cmap='viridis', edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='PPFD (a.u.)')

ax.set_title('3D PPFD Distribution with Rotated COB LED Array')
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_zlabel('PPFD (a.u.)')

# Helper: compute PPFD at a single (x,y) point
def compute_ppfd_at_point(x, y, leds, h, epsilon=1e-6):
    total = 0
    for led in leds:
        led_x, led_y = led['pos']
        d = np.sqrt((x - led_x)**2 + (y - led_y)**2 + h**2) + epsilon
        total += led['I'] * h / (d**3)
    return total

# Overlay LED positions by computing their PPFD on the target plane
led_xs, led_ys, led_zs = [], [], []
for led in leds:
    x, y = led['pos']
    led_xs.append(x)
    led_ys.append(y)
    led_zs.append(compute_ppfd_at_point(x, y, leds, h))

ax.scatter(led_xs, led_ys, led_zs, color='red', s=50, label='LED Positions')
ax.legend()

plt.show()

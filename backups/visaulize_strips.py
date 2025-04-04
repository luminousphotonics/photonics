#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt

# ==============================================================================
# Configuration (from automated_predictor.py)
# ==============================================================================
NUM_LAYERS_TARGET = 5      # Example: Visualize for N=5 layers
SIM_W = 10.9728             # 40 ft in meters
SIM_L = 5.4864             # 40 ft in meters
SIM_H = 0.9144              # 3 ft in meters (Height primarily for Z, less critical for XY plot)

# ==============================================================================
# COB Position Logic (Adapted from build_cob_positions)
# ==============================================================================

def get_transformed_cob_positions(W, L, H, num_total_layers):
    """
    Generates the final transformed COB positions based on the original script's logic.
    Returns a list of tuples: [(px, py, pz, layer), ...]
    and the parameters used for transformation (needed for strips).
    """
    n = num_total_layers - 1 # Max layer index
    positions = []
    # Abstract diamond coordinates
    positions.append((0, 0, H, 0)) # Center COB, layer 0
    for i in range(1, n + 1): # Layers 1 to n
        for x in range(-i, i + 1):
            y_abs = i - abs(x)
            # Add point if it's on the diamond perimeter for layer i
            if abs(x) + y_abs == i:
                 if y_abs == 0:
                     positions.append((x, 0, H, i))
                 else:
                     positions.append((x, y_abs, H, i))
                     positions.append((x, -y_abs, H, i))

    # Transformation parameters
    theta = math.radians(45)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    center_x, center_y = W / 2, L / 2
    # Ensure scale avoids division by zero if n=0 (num_total_layers=1)
    scale_x = W / (n * math.sqrt(2)) if n > 0 else W / 2
    scale_y = L / (n * math.sqrt(2)) if n > 0 else L / 2

    transformed_positions = []
    for (ax, ay, ah, layer) in positions: # Abstract x, y, h
        # Rotate
        rx = ax * cos_t - ay * sin_t
        ry = ax * sin_t + ay * cos_t
        # Scale and Translate
        px = center_x + rx * scale_x
        py = center_y + ry * scale_y
        # Use consistent Z height (slightly below ceiling)
        pz = H * 0.98
        transformed_positions.append((px, py, pz, layer))

    # Store transformation params needed for strips
    transform_params = {
        'center_x': center_x, 'center_y': center_y,
        'scale_x': scale_x, 'scale_y': scale_y,
        'cos_t': cos_t, 'sin_t': sin_t,
        'H': H
    }

    return transformed_positions, transform_params

def get_ordered_strip_layer_coords(layer_index, transform_params):
    """
    Generates the transformed coordinates for a specific layer's perimeter IN ORDER,
    suitable for drawing connecting strips. Returns list of (px, py, pz) tuples.
    """
    if layer_index == 0:
        return [] # No strips for the center layer

    i = layer_index
    ordered_abstract_coords = []
    H = transform_params['H']

    # Generate abstract coordinates in clockwise order (example)
    # Top-Right to Top-Left (Quadrant 1 changing to 2)
    for x in range(i, 0, -1):
        ordered_abstract_coords.append((x, i - x, H, i))
    # Top-Left to Bottom-Left (Quadrant 2 changing to 3)
    for x in range(0, -i, -1):
         ordered_abstract_coords.append((x, i + x, H, i)) # y = i - abs(x) = i - (-x) = i + x
    # Bottom-Left to Bottom-Right (Quadrant 3 changing to 4)
    for x in range(-i, 0, 1):
         ordered_abstract_coords.append((x, -i - x, H, i)) # y = -(i - abs(x)) = -(i - (-x)) = -i - x
    # Bottom-Right to Top-Right (Quadrant 4 changing to 1)
    for x in range(0, i, 1):
         ordered_abstract_coords.append((x, -i + x, H, i)) # y = -(i - abs(x)) = -(i - x) = -i + x


    # Apply the same transformation
    center_x = transform_params['center_x']
    center_y = transform_params['center_y']
    scale_x = transform_params['scale_x']
    scale_y = transform_params['scale_y']
    cos_t = transform_params['cos_t']
    sin_t = transform_params['sin_t']

    ordered_transformed_coords = []
    for (ax, ay, ah, layer) in ordered_abstract_coords:
        rx = ax * cos_t - ay * sin_t
        ry = ax * sin_t + ay * cos_t
        px = center_x + rx * scale_x
        py = center_y + ry * scale_y
        pz = H * 0.98
        ordered_transformed_coords.append((px, py, pz))

    return ordered_transformed_coords

# ==============================================================================
# Visualization
# ==============================================================================

print(f"Generating visualization for N={NUM_LAYERS_TARGET} layers...")

# 1. Get COB positions and transformation details
all_cobs_transformed, transform_params = get_transformed_cob_positions(SIM_W, SIM_L, SIM_H, NUM_LAYERS_TARGET)
cob_array = np.array(all_cobs_transformed) # Easier slicing

# 2. Setup Plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal', adjustable='box')
ax.set_title(f'COB and LED Strip Layout (N={NUM_LAYERS_TARGET})')
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_xlim(0, SIM_W)
ax.set_ylim(0, SIM_L)
ax.grid(True, linestyle='--', alpha=0.6)

# 3. Plot COBs (color-coded by layer)
cmap_cobs = plt.cm.viridis
num_layers = NUM_LAYERS_TARGET
unique_layers = sorted(list(set(cob_array[:, 3].astype(int))))
colors_cobs = cmap_cobs(np.linspace(0, 1, num_layers))

for layer_idx in unique_layers:
    layer_mask = cob_array[:, 3] == layer_idx
    ax.scatter(cob_array[layer_mask, 0], cob_array[layer_mask, 1],
               color=colors_cobs[layer_idx], edgecolors='black', s=50,
               label=f'COB Layer {layer_idx}', zorder=3)

# 4. Plot LED Strips (color-coded by layer, different style)
cmap_strips = plt.cm.plasma # Use a different colormap for strips
colors_strips = cmap_strips(np.linspace(0, 1, num_layers))

strip_handles = [] # For legend

for layer_idx in range(1, num_layers): # Strips only for layers 1 to N-1
    ordered_points = get_ordered_strip_layer_coords(layer_idx, transform_params)

    if not ordered_points: continue # Should not happen for layer > 0

    num_points = len(ordered_points)
    for i in range(num_points):
        p1 = ordered_points[i]
        p2 = ordered_points[(i + 1) % num_points] # Wrap around connection

        line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                 color=colors_strips[layer_idx], linewidth=2.5, linestyle='-',
                 alpha=0.8, zorder=2) # Draw strips behind COBs

    # Add only one representative line per layer to the legend handles
    if num_points > 0 and layer_idx not in [h.get_label() for h in strip_handles]:
         # Create a proxy artist for the legend
         strip_handles.append(plt.Line2D([0], [0], color=colors_strips[layer_idx], lw=2.5, label=f'Strip Layer {layer_idx}'))


# 5. Add Legend
handles, labels = ax.get_legend_handles_labels()
# Combine COB handles and strip handles
ax.legend(handles=handles + strip_handles, fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5))

# 6. Show Plot
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
plt.show()

print("Visualization complete.")
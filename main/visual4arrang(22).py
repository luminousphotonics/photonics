#!/usr/bin/env python3
import math
import matplotlib.pyplot as plt
import argparse

# Constants
NUM_LAYERS = 5

def build_diamond_61_pattern(width_ft, length_ft):
    layers_raw = [
        [(0, 0)], # Central lighting element (Layer 1 - 5 Lighting elements)
        [(-1, 0), (1, 0), (0, -1), (0, 1)], # (Layer 1 - 5 Lighting elements)
        [(-1, -1), (1, -1), (-1, 1), (1, 1), # (Layer 2 - 8 lighting elements)
         (-2, 0), (2, 0), (0, -2), (0, 2)], # (Layer 2 - 8 Lighting elements)
        [(-2, -1), (2, -1), (-2, 1), (2, 1), # (Layer 3 - 12 Lighting elements)
         (-1, -2), (1, -2), (-1, 2), (1, 2), # (Layer 3 - 12 Lighting elements)
         (-3, 0), (3, 0), (0, -3), (0, 3)], # (Layer 3 - 12 Lighting elements)
        [(-2, -2), (2, -2), (-2, 2), (2, 2),  # (Layer 4 - 16 Lighting elements)
         (-3, -1), (3, -1), (-3, 1), (3, 1), # (Layer 4 - 16 Lighting elements)
         (-1, -3), (1, -3), (-1, 3), (1, 3), # (Layer 4 - 16 Lighting elements)
         (-4, 0), (4, 0), (0, -4), (0, 4)], # (Layer 4 - 16 Lighting elements)
        [(-3, -2), (3, -2), (-3, 2), (3, 2), # (Layer 5 - 20 Lighting elements)
         (-2, -3), (2, -3), (-2, 3), (2, 3), # (Layer 5 - 20 Lighting elements)
         (-4, -1), (4, -1), (-4, 1), (4, 1), # (Layer 5 - 20 Lighting elements)
         (-1, -4), (1, -4), (-1, 4), (1, 4), # (Layer 5 - 20 Lighting elements)
         (-5, 0), (5, 0), (0, -5), (0, 5)] # (Layer 5 - 20 Lighting elements)
    ]
    spacing_x = width_ft / 3.5
    spacing_y = length_ft / 3.5
    theta = math.radians(45)

    layers_scaled = []
    for layer in layers_raw:
        coords_layer = []
        for (ix, iy) in layer:
            x_offset = ix * spacing_x
            y_offset = iy * spacing_y
            rotated_x = x_offset * math.cos(theta) - y_offset * math.sin(theta)
            rotated_y = x_offset * math.sin(theta) + y_offset * math.cos(theta)
            coords_layer.append((rotated_x, rotated_y))
        layers_scaled.append(coords_layer)
    return layers_scaled

def generate_light_sources(floor_width_ft, floor_length_ft):
    diamond_layers_ft = build_diamond_61_pattern(floor_width_ft, floor_length_ft)
    merged_layer = diamond_layers_ft[0] + diamond_layers_ft[1]
    remaining_layers = diamond_layers_ft[2:6]
    combined_layers_ft = [merged_layer] + remaining_layers
    return combined_layers_ft

def visualize_light_sources(floor_width_ft, floor_length_ft):
    # Generate light source positions using the updated logic in feet
    light_layers = generate_light_sources(floor_width_ft, floor_length_ft)

    # Set up matplotlib plot
    plt.figure(figsize=(8, 8))
    layer_colors = ["red", "blue", "green", "orange", "purple"]

    for idx, layer in enumerate(light_layers):
        if layer:  # If the layer has emitters
            xs = [coord[0] for coord in layer]
            ys = [coord[1] for coord in layer]
            plt.scatter(xs, ys, color=layer_colors[idx % len(layer_colors)], label=f"Layer {idx+1}", s=80, alpha=0.8)

    plt.title(f"Diamond: 61 Light Source Arrangement\nFloor: {floor_width_ft}ft x {floor_length_ft}ft")
    plt.xlabel("X Position (feet)")
    plt.ylabel("Y Position (feet)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize light source arrangement in feet.")
    parser.add_argument('--floor_width', type=float, default=12.0, help='Floor width in feet.')
    parser.add_argument('--floor_length', type=float, default=12.0, help='Floor length in feet.')
    args = parser.parse_args()

    visualize_light_sources(args.floor_width, args.floor_length)

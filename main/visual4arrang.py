#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Constants
FT_TO_M = 0.3048  # Feet to meters conversion
NUM_LAYERS = 5    # Number of light layers

def generate_light_sources(num_layers, center_x, center_y, spacing_x, spacing_y):
    """
    Generates light source coordinates for a given number of concentric square layers.
    """
    light_sources = []
    for n in range(1, num_layers + 1):
        layer_sources = []
        if n == 1:
            # Layer 1: Central emitter plus 4 surrounding lights
            layer_sources.append((center_x, center_y))  # Central emitter
            layer_sources.append((center_x + spacing_x, center_y))  # Right
            layer_sources.append((center_x - spacing_x, center_y))  # Left
            layer_sources.append((center_x, center_y + spacing_y))  # Top
            layer_sources.append((center_x, center_y - spacing_y))  # Bottom
        else:
            # Layers 2 to num_layers: Concentric squares around the center
            for x in range(-n, n + 1):
                for y in range(-n, n + 1):
                    if abs(x) == n or abs(y) == n:
                        layer_sources.append((center_x + x * spacing_x, center_y + y * spacing_y))
        light_sources.append(layer_sources)
    return light_sources

def visualize_light_sources(floor_width_ft, floor_length_ft):
    # Convert feet to meters
    floor_width_m = floor_width_ft * FT_TO_M
    floor_length_m = floor_length_ft * FT_TO_M

    # Calculate spacing based on number of layers
    spacing_x = floor_width_m / (2 * NUM_LAYERS)
    spacing_y = floor_length_m / (2 * NUM_LAYERS)

    # Calculate center of the floor
    center_x = floor_width_m / 2
    center_y = floor_length_m / 2

    # Generate light sources arrangement
    light_sources = generate_light_sources(NUM_LAYERS, center_x, center_y, spacing_x, spacing_y)

    # Plotting the light source arrangement
    plt.figure(figsize=(8, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple']  # Colors for different layers
    for layer_index, layer in enumerate(light_sources):
        if layer:  # Check if layer has any lights
            xs, ys = zip(*layer)
        else:
            xs, ys = [], []
        plt.scatter(xs, ys, color=colors[layer_index % len(colors)], label=f'Layer {layer_index + 1}')

    plt.title(f'Light Source Arrangement for {floor_width_ft}ft x {floor_length_ft}ft Floor')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize light source arrangement.")
    parser.add_argument('--floor_width', type=float, default=12.0, help='Floor width in feet.')
    parser.add_argument('--floor_length', type=float, default=15.0, help='Floor length in feet.')
    args = parser.parse_args()

    visualize_light_sources(args.floor_width, args.floor_length)

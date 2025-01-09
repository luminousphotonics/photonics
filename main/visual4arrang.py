import matplotlib.pyplot as plt
import numpy as np

# Assuming the following functions and constants are imported from your simulation script
# from your_simulation_module import generate_light_sources, NUM_LAYERS, FT_TO_M, HEIGHT_FROM_FLOOR

def visualize_light_sources(floor_width_ft, floor_length_ft):
    # Convert feet to meters
    floor_width_m = floor_width_ft * FT_TO_M
    floor_length_m = floor_length_ft * FT_TO_M

    # Define spacing based on how your simulation calculates it
    spacing_x = floor_width_m / (2 * NUM_LAYERS)
    spacing_y = floor_length_m / (2 * NUM_LAYERS)
    
    # Calculate the center of the floor
    center_x = floor_width_m / 2
    center_y = floor_length_m / 2

    # Generate light sources arrangement
    light_sources = generate_light_sources(NUM_LAYERS, center_x, center_y, spacing_x, spacing_y)

    # Plotting
    plt.figure(figsize=(8, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple']  # Colors for different layers
    for layer_index, layer in enumerate(light_sources):
        xs, ys = zip(*layer) if layer else ([], [])
        plt.scatter(xs, ys, color=colors[layer_index % len(colors)], label=f'Layer {layer_index + 1}')
    
    plt.title(f'Light Source Arrangement for {floor_width_ft}ft x {floor_length_ft}ft Floor')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# Example usage: visualize for a 12ft x 15ft room
visualize_light_sources(12.0, 15.0)

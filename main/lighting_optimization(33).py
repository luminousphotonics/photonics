#!/usr/bin/env python3
"""
Lighting Optimization Script with Diamond Pattern Arrangement and 5 Layers

This script optimizes lighting intensities across 5 layers following a diamond pattern layout to achieve
a uniform Photosynthetic Photon Flux Density (PPFD) distribution on the floor, 
considering both direct light and reflected light from perimeter walls based on their reflectivity.

Usage:
    python lighting_optimization_diamond.py [--floor_width WIDTH] [--floor_length LENGTH]
                                            [--target_ppfd PPFD] [--perimeter_reflectivity REFLECTIVITY]
                                            [--verbose]

Author: Your Name
Date: 2025-01-04
"""

import math
import numpy as np
from scipy.optimize import minimize
import argparse
import sys

# --- Constants and Parameters ---
NUM_LAYERS = 5
NUM_MEASUREMENT_POINTS = 800
DEFAULT_FLOOR_WIDTH = 12.0
DEFAULT_FLOOR_LENGTH = 12.0
HEIGHT_FROM_FLOOR = 2.5
DEFAULT_TARGET_PPFD = 1000.0
LUMENS_TO_PPFD_CONVERSION = 7
MAX_LUMENS = 20000.0
MIN_LUMENS = 2000.0
DEFAULT_PERIMETER_REFLECTIVITY = 0.0

# --- Diamond Pattern Light Source Generation ---
def build_diamond_61_pattern(width_ft, length_ft):
    layers_raw = [
        [(0, 0)],  # Central lighting element (Layer 1 - 5 Lighting elements)
        [(-1, 0), (1, 0), (0, -1), (0, 1)],  # Layer 1 additional emitters
        [(-1, -1), (1, -1), (-1, 1), (1, 1),  # Layer 2
         (-2, 0), (2, 0), (0, -2), (0, 2)],
        [(-2, -1), (2, -1), (-2, 1), (2, 1),  # Layer 3
         (-1, -2), (1, -2), (-1, 2), (1, 2),
         (-3, 0), (3, 0), (0, -3), (0, 3)],
        [(-2, -2), (2, -2), (-2, 2), (2, 2),  # Layer 4
         (-3, -1), (3, -1), (-3, 1), (3, 1),
         (-1, -3), (1, -3), (-1, 3), (1, 3),
         (-4, 0), (4, 0), (0, -4), (0, 4)],
        [(-3, -2), (3, -2), (-3, 2), (3, 2),  # Layer 5
         (-2, -3), (2, -3), (-2, 3), (2, 3),
         (-4, -1), (4, -1), (-4, 1), (4, 1),
         (-1, -4), (1, -4), (-1, 4), (1, 4),
         (-5, 0), (5, 0), (0, -5), (0, 5)]
    ]
    # Scale factors based on floor dimensions
    spacing_x = width_ft / 7.2
    spacing_y = length_ft / 7.2
    theta = math.radians(45)

    layers_scaled = []
    for layer in layers_raw:
        coords_layer = []
        for (ix, iy) in layer:
            # Scale and rotate coordinates
            x_offset = ix * spacing_x
            y_offset = iy * spacing_y
            rotated_x = x_offset * math.cos(theta) - y_offset * math.sin(theta)
            rotated_y = x_offset * math.sin(theta) + y_offset * math.cos(theta)
            coords_layer.append((rotated_x + width_ft/2, rotated_y + length_ft/2))  
            # Translate to center floor coordinates
        layers_scaled.append(coords_layer)
    return layers_scaled

# --- Determine the number of active layers based on floor dimensions ---
def determine_active_layers(floor_width, floor_length):
    layer_dimensions = [
        (0, 0), 
        (3, 3), 
        (5, 5), 
        (8, 8), 
        (11, 11), 
        (14, 14)
    ]
    active_layers = 0
    for i in range(1, len(layer_dimensions)):
        required_width, required_length = layer_dimensions[i]
        if floor_width >= required_width and floor_length >= required_length:
            active_layers = i
        else:
            break
    return active_layers

def generate_light_sources(floor_width, floor_length):
    # Use the diamond pattern to generate layers
    diamond_layers = build_diamond_61_pattern(floor_width, floor_length)
    # Merge first two raw layers for Layer 1 as per initial description:
    merged_layer1 = diamond_layers[0] + diamond_layers[1]
    # Layers 2 to 5 remain as defined
    layers_2_to_5 = diamond_layers[2:6]
    # Combine into final list of layers
    combined_layers = [merged_layer1] + layers_2_to_5
    return combined_layers

# --- Lambertian Emission Model ---
def lambertian_emission(intensity, distance, z):
    if distance == 0:
        return intensity
    return (intensity * z) / ((distance ** 2 + z ** 2) ** 1.5)

# --- Calculate PPFD at a Point ---
def calculate_ppfd_at_point(point, light_sources, light_intensities_by_layer, height_from_floor):
    total_ppfd = 0
    for layer_index, layer_sources in enumerate(light_sources):
        layer_intensity = light_intensities_by_layer[layer_index]
        for source in layer_sources:
            distance = math.sqrt(
                (source[0] - point[0]) ** 2 +
                (source[1] - point[1]) ** 2 +
                height_from_floor ** 2
            )
            total_ppfd += lambertian_emission(layer_intensity, distance, height_from_floor)
    return total_ppfd

# --- Generate Measurement Points ---
def generate_measurement_points(floor_width, floor_length, num_points):
    measurement_points = []
    points_per_dimension = int(math.sqrt(num_points))
    step_x = floor_width / (points_per_dimension + 1)
    step_y = floor_length / (points_per_dimension + 1)
    for i in range(1, points_per_dimension + 1):
        for j in range(1, points_per_dimension + 1):
            x = i * step_x
            y = j * step_y
            measurement_points.append((x, y))
    return measurement_points

# --- Calculate Reflected PPFD from Perimeter ---
def calculate_reflected_ppfd(measurement_points, light_sources, light_intensities_by_layer, 
                             height_from_floor, floor_width, floor_length, perimeter_reflectivity):
    reflected_ppfd = np.zeros(len(measurement_points))
    reflected_fraction = 0.1
    edge_threshold = min(floor_width, floor_length) * 0.05  
    total_reflected_light = 0
    for idx, point in enumerate(measurement_points):
        near_left = point[0] <= edge_threshold
        near_right = point[0] >= (floor_width - edge_threshold)
        near_bottom = point[1] <= edge_threshold
        near_top = point[1] >= (floor_length - edge_threshold)
        is_near_wall = near_left or near_right or near_bottom or near_top

        if is_near_wall:
            direct_ppfd = calculate_ppfd_at_point(point, light_sources, light_intensities_by_layer, height_from_floor)
            reflected_light = direct_ppfd * reflected_fraction
            total_reflected_light += reflected_light

    total_reflected_light *= perimeter_reflectivity
    if len(measurement_points) > 0:
        reflected_ppfd += total_reflected_light / len(measurement_points)

    return reflected_ppfd

# --- Calculate Intensity and Variance ---
def calculate_intensity(floor_width, floor_length, height_from_floor, light_intensities_by_layer, 
                        perimeter_reflectivity, measurement_points):
    light_sources = generate_light_sources(floor_width, floor_length)

    direct_ppfd = np.array([
        calculate_ppfd_at_point(point, light_sources, light_intensities_by_layer, height_from_floor)
        for point in measurement_points
    ])

    reflected_ppfd = calculate_reflected_ppfd(
        measurement_points, light_sources, light_intensities_by_layer,
        height_from_floor, floor_width, floor_length, perimeter_reflectivity
    )

    total_ppfd = direct_ppfd + reflected_ppfd
    mean_intensity = np.mean(total_ppfd)
    mad = np.mean(np.abs(total_ppfd - mean_intensity))

    return total_ppfd, mad, mean_intensity

# --- Objective Function for Optimization ---
def objective_function(light_intensities_by_layer, floor_width, floor_length, height_from_floor,
                       target_ppfd, perimeter_reflectivity, measurement_points):
    _, mad, mean_intensity = calculate_intensity(
        floor_width, floor_length, height_from_floor, light_intensities_by_layer, 
        perimeter_reflectivity, measurement_points
    )
    print(f"Objective - MAD: {mad:.2f}, PPFD Penalty: {(mean_intensity - target_ppfd)**2:.2f}")
    ppfd_penalty = (mean_intensity - target_ppfd) ** 2
    return mad * 1.0 + ppfd_penalty * 0.3

# --- Optimization Process ---
def optimize_lighting(floor_width, floor_length, height_from_floor,
                      initial_intensities_by_layer, target_ppfd, 
                      perimeter_reflectivity, measurement_points, verbose=False):
    # Determine the number of active layers based on floor dimensions
    active_layers = determine_active_layers(floor_width, floor_length)

    # Set bounds: For active layers use normal bounds, inactive layers fixed to 0
    bounds = []
    for i in range(NUM_LAYERS):
        if i < active_layers:
            bounds.append((MIN_LUMENS / LUMENS_TO_PPFD_CONVERSION, MAX_LUMENS / LUMENS_TO_PPFD_CONVERSION))
        else:
            bounds.append((0, 0))  # Fix intensity for inactive layers

    # Initialize intensities: set nonzero for active layers, zero for others
    for i in range(active_layers, NUM_LAYERS):
        initial_intensities_by_layer[i] = 0.0

    if verbose:
        print("Starting optimization...")
        print(f"Initial intensities (PPFD): {initial_intensities_by_layer}")
        print(f"Active layers based on floor dimensions: {active_layers}")
        print(f"Perimeter Reflectivity: {perimeter_reflectivity}")

    result = minimize(
        objective_function,
        initial_intensities_by_layer,
        args=(
            floor_width,
            floor_length,
            height_from_floor,
            target_ppfd,
            perimeter_reflectivity,
            measurement_points
        ),
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 10000, 'disp': verbose}
    )

    if not result.success:
        error_msg = f"Optimization did not converge: {result.message}"
        print(error_msg, file=sys.stderr)
        raise Exception(error_msg)

    optimized_intensities_by_layer = result.x
    _, mad, optimized_ppfd = calculate_intensity(
        floor_width, floor_length, height_from_floor, optimized_intensities_by_layer, 
        perimeter_reflectivity, measurement_points
    )

    if verbose:
        print(f"Optimization completed with MAD: {mad:.2f} and Optimized PPFD: {optimized_ppfd:.2f}")

    return optimized_intensities_by_layer, mad, optimized_ppfd

# --- Run Simulation ---
def run_simulation(floor_width, floor_length, target_ppfd, perimeter_reflectivity, verbose=False):
    initial_intensities_by_layer = [3000.0 / LUMENS_TO_PPFD_CONVERSION] * NUM_LAYERS
    measurement_points = generate_measurement_points(floor_width, floor_length, NUM_MEASUREMENT_POINTS)

    if verbose:
        print(f"Running simulation with floor width {floor_width}m, floor length {floor_length}m,")
        print(f"target PPFD {target_ppfd} µmol/m²/s, and perimeter reflectivity {perimeter_reflectivity}")

    optimized_intensities, mad, optimized_ppfd = optimize_lighting(
        floor_width, floor_length, HEIGHT_FROM_FLOOR,
        initial_intensities_by_layer, target_ppfd, perimeter_reflectivity, measurement_points, verbose=verbose
    )

    optimized_lumens_by_layer = [intensity * LUMENS_TO_PPFD_CONVERSION for intensity in optimized_intensities]

    if verbose:
        print("\nOptimized Lumens (grouped by layer):")
        for i, lumens in enumerate(optimized_lumens_by_layer, start=1):
            print(f"  Layer {i}: {lumens:.2f} lumens")
        print(f"Minimized MAD: {mad:.2f} µmol/m²/s")
        print(f"Optimized PPFD: {optimized_ppfd:.2f} µmol/m²/s")

    return {
        "optimized_lumens_by_layer": optimized_lumens_by_layer,
        "mad": mad,
        "optimized_ppfd": optimized_ppfd,
        "floor_width": floor_width,
        "floor_length": floor_length,
        "target_ppfd": target_ppfd,
        "floor_height": HEIGHT_FROM_FLOOR,
        "perimeter_reflectivity": perimeter_reflectivity
    }

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Optimize lighting intensities using diamond pattern for uniform PPFD.")
    parser.add_argument('--floor_width', type=float, default=DEFAULT_FLOOR_WIDTH,
                        help='Width of the floor area (meters).')
    parser.add_argument('--floor_length', type=float, default=DEFAULT_FLOOR_LENGTH,
                        help='Length of the floor area (meters).')
    parser.add_argument('--target_ppfd', type=float, default=DEFAULT_TARGET_PPFD,
                        help='Target PPFD (µmol/m²/s).')
    parser.add_argument('--perimeter_reflectivity', type=float, default=DEFAULT_PERIMETER_REFLECTIVITY,
                        help='Perimeter wall reflectivity (0 to 1).')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity')
    args = parser.parse_args()

    if args.floor_width <= 0 or args.floor_length <= 0:
        print("Error: Floor dimensions must be positive numbers.", file=sys.stderr)
        sys.exit(1)
    if args.target_ppfd <= 0:
        print("Error: Target PPFD must be a positive number.", file=sys.stderr)
        sys.exit(1)
    if not (0.0 <= args.perimeter_reflectivity <= 1.0):
        print("Error: Perimeter reflectivity must be between 0 and 1.", file=sys.stderr)
        sys.exit(1)

    try:
        results = run_simulation(
            args.floor_width,
            args.floor_length,
            args.target_ppfd,
            args.perimeter_reflectivity,
            verbose=args.verbose
        )
        # Further processing or output of results as needed.
    except Exception as e:
        print(f"Simulation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

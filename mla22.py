#!/usr/bin/env python3
"""
Lighting Optimization Script

This script optimizes lighting intensities across multiple layers to achieve
a uniform Photosynthetic Photon Flux Density (PPFD) distribution on the floor.

Usage:
    python lighting_optimization.py [--floor_width WIDTH] [--floor_length LENGTH]
                                     [--target_ppfd PPFD] [--verbose]

Example:
    python lighting_optimization.py --floor_width 12.0 --floor_length 15.0 --target_ppfd 1200

Author: OpenAI ChatGPT
Date: 2025-01-04
"""

import math
import numpy as np
from scipy.optimize import minimize
import argparse
import sys

# --- Constants and Parameters ---
NUM_LAYERS = 6
NUM_MEASUREMENT_POINTS = 225
DEFAULT_FLOOR_WIDTH = 12.0
DEFAULT_FLOOR_LENGTH = 12.0  # Using length now
HEIGHT_FROM_FLOOR = 2.5  # Using height from floor now
BEAM_ANGLE_DEG = 120
DEFAULT_TARGET_PPFD = 1000.0
LUMENS_TO_PPFD_CONVERSION = 13
MAX_LUMENS = 13500.0  # Maximum luminous output constraint
MIN_LUMENS = 3000.0  # Minimum luminous output constraint

# --- Centered Square Layer Arrangement ---
def generate_light_sources(num_layers, center_x, center_y, spacing_x, spacing_y):
    """
    Generates light source coordinates for concentric square layers.
    """
    light_sources = []
    for n in range(num_layers):
        layer_sources = []
        for x in range(-n, n + 1):
            for y in range(-n, n + 1):
                if abs(x) == n or abs(y) == n:  # On the edge of the square
                    layer_sources.append((center_x + x * spacing_x, center_y + y * spacing_y))
        light_sources.append(layer_sources)
    return light_sources

# --- Lambertian Emission Model ---
def lambertian_emission(intensity, distance, z):
    """
    Calculates radiant intensity at a point from a Lambertian source.
    """
    if distance == 0:
        return intensity
    return (intensity * z) / ((distance ** 2 + z ** 2) ** 1.5)

# --- Calculate PPFD at a Point ---
def calculate_ppfd_at_point(point, light_sources, light_intensities_by_layer, height_from_floor):
    """
    Calculates total PPFD at a point from all light sources.
    """
    total_ppfd = 0
    for layer_index, layer_sources in enumerate(light_sources):
        layer_intensity = light_intensities_by_layer[layer_index]
        for source in layer_sources:
            distance = math.sqrt((source[0] - point[0]) ** 2 + (source[1] - point[1]) ** 2 + height_from_floor ** 2)
            total_ppfd += lambertian_emission(layer_intensity, distance, height_from_floor)
    return total_ppfd

# --- Generate Measurement Points ---
def generate_measurement_points(floor_width, floor_length, num_points):
    """
    Generates measurement points within the floor area.
    """
    measurement_points = []
    points_per_dimension = int(math.sqrt(num_points))
    step_x = floor_width / (points_per_dimension + 1)
    step_y = floor_length / (points_per_dimension + 1)  # Use floor_length
    for i in range(1, points_per_dimension + 1):
        for j in range(1, points_per_dimension + 1):
            x = i * step_x
            y = j * step_y
            measurement_points.append((x, y))
    return measurement_points

# --- Calculate Intensity and Variance ---
def calculate_intensity(floor_width, floor_length, height_from_floor, light_array_width, light_array_height, light_intensities_by_layer, pattern_option=None):
    """
    Calculates photon distribution and variance given input parameters.
    """
    spacing_x = light_array_width / 10
    spacing_y = light_array_height / 10
    light_sources = generate_light_sources(NUM_LAYERS, floor_width / 2, floor_length / 2, spacing_x, spacing_y)
    measurement_points = generate_measurement_points(floor_width, floor_length, NUM_MEASUREMENT_POINTS)

    intensities = np.array([
        calculate_ppfd_at_point(point, light_sources, light_intensities_by_layer, height_from_floor)
        for point in measurement_points
    ])

    mean_intensity = np.mean(intensities)
    mad = np.mean(np.abs(intensities - mean_intensity))

    return intensities, mad, mean_intensity

# --- Objective Function for Optimization ---
def objective_function(light_intensities_by_layer, floor_width, floor_length, height_from_floor, light_array_width, light_array_height, target_ppfd):
    """
    The function to be minimized, balancing uniformity and target PPFD.
    """
    _, mad, mean_intensity = calculate_intensity(
        floor_width, floor_length, height_from_floor, light_array_width, light_array_height, light_intensities_by_layer, None
    )

    # Penalize deviation from target PPFD
    ppfd_penalty = (mean_intensity - target_ppfd) ** 2

    # Combine MAD and PPFD penalties
    return mad + 0.1 * ppfd_penalty

# --- Optimization Process ---
def optimize_lighting(floor_width, floor_length, height_from_floor, light_array_width, light_array_height, initial_intensities_by_layer, target_ppfd, verbose=False):
    """
    Optimizes light intensities to minimize MAD, considering the target PPFD.
    """
    bounds = [(MIN_LUMENS / LUMENS_TO_PPFD_CONVERSION, MAX_LUMENS / LUMENS_TO_PPFD_CONVERSION)] * NUM_LAYERS

    if verbose:
        print("Starting optimization...")
        print(f"Initial intensities: {initial_intensities_by_layer}")

    result = minimize(
        objective_function,
        initial_intensities_by_layer,
        args=(floor_width, floor_length, height_from_floor, light_array_width, light_array_height, target_ppfd),
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 500, 'disp': verbose}
    )

    if not result.success:
        print("Optimization did not converge:", result.message, file=sys.stderr)
        sys.exit(1)

    optimized_intensities_by_layer = result.x
    _, mad, optimized_ppfd = calculate_intensity(
        floor_width, floor_length, height_from_floor, light_array_width, light_array_height, optimized_intensities_by_layer, None
    )

    return optimized_intensities_by_layer, mad, optimized_ppfd

# --- Run Simulation ---
def run_simulation(floor_width, floor_length, target_ppfd, verbose=False):
    """
    Main function to run the optimization.
    """
    light_array_width = floor_width
    light_array_height = floor_length
    initial_intensities_by_layer = [200.0] * NUM_LAYERS  # Starting with 200 PPFD per layer

    if verbose:
        print(f"Running simulation with floor width {floor_width}m, floor length {floor_length}m, target PPFD {target_ppfd} µmol/m²/s")

    optimized_intensities, mad, optimized_ppfd = optimize_lighting(
        floor_width, floor_length, HEIGHT_FROM_FLOOR, light_array_width, light_array_height,
        initial_intensities_by_layer, target_ppfd, verbose=verbose
    )

    # Convert optimized intensities to lumens
    optimized_lumens_by_layer = [intensity * LUMENS_TO_PPFD_CONVERSION for intensity in optimized_intensities]

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
        "floor_length": floor_length,  # Now returning floor_length
        "target_ppfd": target_ppfd,
        "floor_height": HEIGHT_FROM_FLOOR,  # Add floor_height to the return object
    }

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Optimize lighting intensities for uniform PPFD distribution.")
    parser.add_argument('--floor_width', type=float, default=DEFAULT_FLOOR_WIDTH, help='Width of the floor area (meters). Default is 10.0')
    parser.add_argument('--floor_length', type=float, default=DEFAULT_FLOOR_LENGTH, help='Length of the floor area (meters). Default is 10.0')
    parser.add_argument('--target_ppfd', type=float, default=DEFAULT_TARGET_PPFD, help='Target PPFD (µmol/m²/s). Default is 1000.0')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity')
    args = parser.parse_args()

    run_simulation(args.floor_width, args.floor_length, args.target_ppfd, verbose=args.verbose)

if __name__ == '__main__':
    main()

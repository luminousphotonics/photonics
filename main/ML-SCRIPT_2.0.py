#!/usr/bin/env python3
"""
Lighting Optimization Script with Diamond Pattern Arrangement and multiple Layers

This script optimizes lighting intensities across separate concentric square layers following 
a centered square number integer sequence pattern layout to achieve a uniform Illuminance (LUX) 
distribution on the floor. It uses Root Mean Squared Error (RMSE) for optimization.

Usage:
    python lighting_optimization_diamond.py [--floor_width WIDTH] [--floor_length LENGTH]
                                            [--target_lux LUX] [--perimeter_reflectivity REFLECTIVITY]
                                            [--verbose]

Author: Bard
Date: 2025-01-04
"""

import math
import numpy as np
from scipy.optimize import minimize
import argparse
import sys

# --- Constants and Parameters ---
NUM_LAYERS = 10
NUM_MEASUREMENT_POINTS = 800
DEFAULT_FLOOR_WIDTH = 12
DEFAULT_FLOOR_LENGTH = 12
HEIGHT_FROM_FLOOR = 2.5
DEFAULT_TARGET_LUX = 2000
LUMENS_TO_PPFD_CONVERSION = 1  # For LUX calculations, set conversion factor to 1
MAX_LUMENS = 20000.0
MIN_LUMENS = 2000.0
DEFAULT_PERIMETER_REFLECTIVITY = 0.03

# --- Diamond Pattern Light Source Generation ---
def build_diamond_61_pattern(width_ft, length_ft):
    layers_raw = [
        [(0, 0)],
        [(-1, 0), (1, 0), (0, -1), (0, 1)],
        [(-1, -1), (1, -1), (-1, 1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)],
        [(-2, -1), (2, -1), (-2, 1), (2, 1), (-1, -2), (1, -2), (-1, 2), (1, 2), (-3, 0), (3, 0), (0, -3), (0, 3)],
        [(-2, -2), (2, -2), (-2, 2), (2, 2), (-3, -1), (3, -1), (-3, 1), (3, 1), (-1, -3), (1, -3), (-1, 3), (1, 3), (-4, 0), (4, 0), (0, -4), (0, 4)],
        [(-3, -2), (3, -2), (-3, 2), (3, 2), (-2, -3), (2, -3), (-2, 3), (2, 3),
         (-4, -1), (4, -1), (-4, 1), (4, 1), (-1, -4), (1, -4), (-1, 4), (1, 4),
         (-5, 0), (5, 0), (0, -5), (0, 5)],
        [(-3, -3), (3, -3), (-3, 3), (3, 3), (-4, -2), (4, -2), (-4, 2), (4, 2),
         (-2, -4), (2, -4), (-2, 4), (2, 4), (-5, -1), (5, -1), (-5, 1), (5, 1),
         (-1, -5), (1, -5), (-1, 5), (1, 5), (-6, 0), (6, 0), (0, -6), (0, 6)],
        [(-4, -3), (4, -3), (-4, 3), (4, 3), (-3, -4), (3, -4), (-3, 4), (3, 4),
         (-5, -2), (5, -2), (-5, 2), (5, 2), (-2, -5), (2, -5), (-2, 5), (2, 5),
         (-6, -1), (6, -1), (-6, 1), (6, 1), (-1, -6), (1, -6), (-1, 6), (1, 6),
         (-7, 0), (7, 0), (0, -7), (0, 7)],
        [(-4, -4), (4, -4), (-4, 4), (4, 4), (-5, -3), (5, -3), (-5, 3), (5, 3),
         (-3, -5), (3, -5), (-3, 5), (3, 5), (-6, -2), (6, -2), (-6, 2), (6, 2),
         (-2, -6), (2, -6), (-2, 6), (2, 6), (-7, -1), (7, -1), (-7, 1), (7, 1),
         (-1, -7), (1, -7), (-1, 7), (1, 7), (-8, 0), (8, 0), (0, -8), (0, 8)],
        [(-5, -4), (5, -4), (-5, 4), (5, 4), (-4, -5), (4, -5), (-4, 5), (4, 5),
         (-6, -3), (6, -3), (-6, 3), (6, 3), (-3, -6), (3, -6), (-3, 6), (3, 6),
         (-7, -2), (7, -2), (-7, 2), (7, 2), (-2, -7), (2, -7), (-2, 7), (2, 7),
         (-8, -1), (8, -1), (-8, 1), (8, 1), (-1, -8), (1, -8), (-1, 8), (1, 8),
         (-9, 0), (9, 0), (0, -9), (0, 9)],
        [(-5, -5), (5, -5), (-5, 5), (5, 5), (-6, -4), (6, -4), (-6, 4), (6, 4),
         (-4, -6), (4, -6), (-4, 6), (4, 6), (-7, -3), (7, -3), (-7, 3), (7, 3),
         (-3, -7), (3, -7), (-3, 7), (3, 7), (-8, -2), (8, -2), (-8, 2), (8, 2),
         (-2, -8), (2, -8), (-2, 8), (2, 8), (-9, -1), (9, -1), (-9, 1), (9, 1),
         (-1, -9), (1, -9), (-1, 9), (1, 9), (-10, 0), (10, 0), (0, -10), (0, 10)]
    ]
    spacing_x = width_ft / 7.2
    spacing_y = length_ft / 7.2
    theta = math.radians(45)

    layers_scaled = []
    for layer in layers_raw:
        coords_layer = []
        for (ix, iy) in layer:
            x_offset = ix * spacing_x
            y_offset = iy * spacing_y
            rotated_x = x_offset * math.cos(theta) - y_offset * math.sin(theta)
            rotated_y = x_offset * math.sin(theta) + y_offset * math.cos(theta)
            coords_layer.append((rotated_x + width_ft/2, rotated_y + length_ft/2))
        layers_scaled.append(coords_layer)
    return layers_scaled

def determine_active_layers(floor_width, floor_length):
    layer_dimensions = [
        (0, 0), (3, 3), (5, 5), (8, 8), (11, 11),
        (12, 12), (17, 17), (20, 20), (23, 23), (26, 26), (29, 29)
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
    return build_diamond_61_pattern(floor_width, floor_length)

def lambertian_emission(intensity, distance, z):
    if distance == 0:
        return intensity
    return (intensity * z) / ((distance ** 2 + z ** 2) ** 1.5)

def calculate_lux_at_point(point, light_sources, light_intensities_by_layer, height_from_floor):
    total_lux = 0
    for layer_index, layer_sources in enumerate(light_sources):
        if layer_index < len(light_intensities_by_layer):
            layer_intensity = light_intensities_by_layer[layer_index]
            for source in layer_sources:
                distance = math.sqrt(
                    (source[0] - point[0]) ** 2 +
                    (source[1] - point[1]) ** 2 +
                    height_from_floor ** 2
                )
                total_lux += lambertian_emission(layer_intensity, distance, height_from_floor)
    return total_lux

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

# --- Reflection Functions ---
def calculate_wall_reflection(measurement_point, floor_width, floor_length, perimeter_reflectivity, direct_lux, decay_constant=1.0):
    x_m, y_m = measurement_point
    reflected_lux_exp = 0.0
    reflected_lux_inv = 0.0

    distances = {
        'left': x_m,
        'right': floor_width - x_m,
        'bottom': y_m,
        'top': floor_length - y_m
    }

    for distance in distances.values():
        reflected_light_exp = direct_lux * perimeter_reflectivity * math.exp(-distance / decay_constant)
        reflected_lux_exp += reflected_light_exp

        reflected_light_inv = direct_lux * perimeter_reflectivity / (distance + 1)**2
        reflected_lux_inv += reflected_light_inv

    return reflected_lux_exp, reflected_lux_inv

def calculate_reflected_light(measurement_points, floor_width, floor_length, perimeter_reflectivity, direct_lux_array, num_iterations=3, decay_constant=1.0):
    reflected_intensity_exp = np.zeros(len(measurement_points))
    reflected_intensity_inv = np.zeros(len(measurement_points))

    for iteration in range(num_iterations):
        temp_intensity_exp = np.zeros(len(measurement_points))
        temp_intensity_inv = np.zeros(len(measurement_points))

        for i, point in enumerate(measurement_points):
            direct_or_updated_lux = direct_lux_array[i] if iteration == 0 else direct_lux_array[i] + reflected_intensity_exp[i]
            refl_exp, refl_inv = calculate_wall_reflection(point, floor_width, floor_length, perimeter_reflectivity, direct_or_updated_lux, decay_constant)
            temp_intensity_exp[i] += refl_exp
            temp_intensity_inv[i] += refl_inv

        reflected_intensity_exp += temp_intensity_exp
        reflected_intensity_inv += temp_intensity_inv

    return reflected_intensity_inv

# --- Calculation and Optimization Functions ---
def calculate_intensity_and_rmse(floor_width, floor_length, height_from_floor, light_intensities_by_layer, 
                                 perimeter_reflectivity, measurement_points):
    light_sources = generate_light_sources(floor_width, floor_length)

    direct_lux = np.array([
        calculate_lux_at_point(point, light_sources, light_intensities_by_layer, height_from_floor)
        for point in measurement_points
    ])

    reflected_lux = calculate_reflected_light(
        measurement_points, floor_width, floor_length, perimeter_reflectivity,
        direct_lux, num_iterations=5, decay_constant=1.5
    )

    total_lux = direct_lux + reflected_lux
    mean_intensity = np.mean(total_lux)
    rmse = np.sqrt(np.mean((total_lux - mean_intensity) ** 2))

    return total_lux, rmse, mean_intensity

def objective_function(light_intensities_by_layer, floor_width, floor_length, height_from_floor,
                       target_lux, perimeter_reflectivity, measurement_points, layer_weights, phase=1):
    active_layers = determine_active_layers(floor_width, floor_length)
    adjusted_intensities = list(light_intensities_by_layer)
    for i in range(active_layers, len(adjusted_intensities)):
        adjusted_intensities[i] = 0.0

    _, rmse, mean_intensity = calculate_intensity_and_rmse(
        floor_width, floor_length, height_from_floor, adjusted_intensities, 
        perimeter_reflectivity, measurement_points
    )

    if mean_intensity <= target_lux:
        lux_penalty = (target_lux - mean_intensity) ** 2
    else:
        overshoot_multiplier = 300
        lux_penalty = overshoot_multiplier * (mean_intensity - target_lux) ** 2

    intensity_dev_penalty = 0
    for i in range(active_layers):
        intensity = adjusted_intensities[i]
        target_intensity = MAX_LUMENS * layer_weights[i]
        intensity_dev_penalty += (intensity - target_intensity)**2

    rmse_scaled = rmse * 0.6
    lux_penalty_scaled = lux_penalty * 0.2 if phase == 1 else 0
    intensity_dev_penalty_scaled = intensity_dev_penalty * 0.2

    objective_value = rmse_scaled + lux_penalty_scaled + intensity_dev_penalty_scaled
    print(f"Phase: {phase}, RMSE: {rmse_scaled:.2f}, LUX Penalty: {lux_penalty_scaled:.2f}, Intensity Deviation Penalty: {intensity_dev_penalty_scaled:.2f}")
    return objective_value

def optimize_lighting(floor_width, floor_length, height_from_floor,
                      initial_intensities_by_layer, target_lux, 
                      perimeter_reflectivity, measurement_points, verbose=False):

    active_layers = determine_active_layers(floor_width, floor_length)

    if verbose:
        print("Starting Optimization")

    bounds = []
    for i in range(NUM_LAYERS):
        if i < active_layers:
            bounds.append((MIN_LUMENS, MAX_LUMENS))
        else:
            bounds.append((0, 0))

    initial_guess = initial_intensities_by_layer
    constraints = []

    layer_weights = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0]

    result = minimize(
        objective_function,
        initial_guess,
        args=(
            floor_width,
            floor_length,
            height_from_floor,
            target_lux,
            perimeter_reflectivity,
            measurement_points,
            layer_weights,
            1
        ),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 2000, 'disp': verbose, 'ftol': 1e-6},
    )

    if not result.success:
        print(f"Warning: Optimization did not converge: {result.message}", file=sys.stderr)
    optimized_intensities = result.x

    _, rmse, optimized_lux = calculate_intensity_and_rmse(
        floor_width, floor_length, height_from_floor, optimized_intensities, 
        perimeter_reflectivity, measurement_points
    )

    if verbose:
        print(f"Optimization completed with RMSE: {rmse:.2f} and Optimized LUX: {optimized_lux:.2f}")

    return optimized_intensities, rmse, optimized_lux

def run_simulation(floor_width, floor_length, target_lux, perimeter_reflectivity, verbose=False):
    average_total_lumens = (MIN_LUMENS + MAX_LUMENS) / 2.0

    if NUM_LAYERS == 5:
        lumen_distribution_percentages = [0.0226, 0.0904, 0.1357, 0.0724, 0.6787]
        initial_intensities_first_five = [
            average_total_lumens * pct
            for pct in lumen_distribution_percentages
        ]
        remaining_layers = [0.0] * (NUM_LAYERS - 5)
        initial_intensities_by_layer = initial_intensities_first_five + remaining_layers
    elif NUM_LAYERS == 10:
        lumen_distribution_percentages = [0.0226, 0.0904, 0.1357, 0.0724]
        sum_first_four = sum(lumen_distribution_percentages)
        remainder = 1.0 - sum_first_four
        equal_share = remainder / 6.0

        initial_intensities = []
        for pct in lumen_distribution_percentages:
            initial_intensities.append(average_total_lumens * pct)
        for _ in range(6):
            initial_intensities.append(average_total_lumens * equal_share)
        initial_intensities_by_layer = initial_intensities
    else:
        initial_intensities_by_layer = [3000.0] * NUM_LAYERS

    measurement_points = generate_measurement_points(floor_width, floor_length, NUM_MEASUREMENT_POINTS)

    if verbose:
        print(f"Running simulation with floor width {floor_width}m, floor length {floor_length}m,")
        print(f"target LUX {target_lux} lm/m², and perimeter reflectivity {perimeter_reflectivity}")

    optimized_intensities, rmse, optimized_lux = optimize_lighting(
        floor_width, floor_length, HEIGHT_FROM_FLOOR,
        initial_intensities_by_layer, target_lux, DEFAULT_PERIMETER_REFLECTIVITY, measurement_points, verbose=verbose
    )

    optimized_lumens_by_layer = [intensity for intensity in optimized_intensities]

    print("\nOptimized Lumens (grouped by layer):")
    for i, lumens in enumerate(optimized_lumens_by_layer, start=1):
        print(f"  Layer {i}: {lumens:.2f} lumens")
    print(f"Minimized RMSE: {rmse:.2f} lm/m²")
    print(f"Optimized LUX: {optimized_lux:.2f} lm/m²")

    return {
        "optimized_lumens_by_layer": optimized_lumens_by_layer,
        "rmse": rmse,
        "optimized_lux": optimized_lux,
        "floor_width": floor_width,
        "floor_length": floor_length,
        "target_lux": target_lux,
        "floor_height": HEIGHT_FROM_FLOOR,
        "perimeter_reflectivity": perimeter_reflectivity
    }

def main():
    parser = argparse.ArgumentParser(description="Optimize lighting intensities using diamond pattern for uniform LUX.")
    parser.add_argument('--floor_width', type=float, default=DEFAULT_FLOOR_WIDTH,
                        help='Width of the floor area (meters).')
    parser.add_argument('--floor_length', type=float, default=DEFAULT_FLOOR_LENGTH,
                        help='Length of the floor area (meters).')
    parser.add_argument('--target_lux', type=float, default=DEFAULT_TARGET_LUX,
                        help='Target LUX (lm/m²).')
    parser.add_argument('--perimeter_reflectivity', type=float, default=DEFAULT_PERIMETER_REFLECTIVITY,
                        help='Perimeter wall reflectivity (0 to 1).')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity')
    args = parser.parse_args()

    if args.floor_width <= 0 or args.floor_length <= 0:
        print("Error: Floor dimensions must be positive numbers.", file=sys.stderr)
        sys.exit(1)
    if args.target_lux <= 0:
        print("Error: Target LUX must be a positive number.", file=sys.stderr)
        sys.exit(1)
    if not (0.0 <= args.perimeter_reflectivity <= 1.0):
        print("Error: Perimeter reflectivity must be between 0 and 1.", file=sys.stderr)
        sys.exit(1)

    try:
        results = run_simulation(
            args.floor_width,
            args.floor_length,
            args.target_lux,
            args.perimeter_reflectivity,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"Simulation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

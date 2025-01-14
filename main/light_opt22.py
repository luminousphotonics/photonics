"""
Lighting Optimization Script with Diamond Pattern Arrangement and multiple Layers

This script optimizes lighting intensities across separate concentric square layers following 
a centered square number integer sequence pattern layout to achieve a uniform Photosynthetic Photon Flux Density 
(PPFD) distribution on the floor.

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
NUM_LAYERS = 10
NUM_MEASUREMENT_POINTS = 800
DEFAULT_FLOOR_WIDTH = 30
DEFAULT_FLOOR_LENGTH = 30
HEIGHT_FROM_FLOOR = 2.5
DEFAULT_TARGET_PPFD = 1250.0
LUMENS_TO_PPFD_CONVERSION = 5
MAX_LUMENS = 20000.0
MIN_LUMENS = 2000.0
DEFAULT_PERIMETER_REFLECTIVITY = 0.0

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
        (14, 14), (17, 17), (20, 20), (23, 23), (26, 26), (29, 29)
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

def calculate_ppfd_at_point(point, light_sources, light_intensities_by_layer, height_from_floor):
    total_ppfd = 0
    for layer_index, layer_sources in enumerate(light_sources):
        if layer_index < len(light_intensities_by_layer):
            layer_intensity = light_intensities_by_layer[layer_index]
            for source in layer_sources:
                distance = math.sqrt(
                    (source[0] - point[0]) ** 2 +
                    (source[1] - point[1]) ** 2 +
                    height_from_floor ** 2
                )
                total_ppfd += lambertian_emission(layer_intensity, distance, height_from_floor)
    return total_ppfd

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

def objective_function(light_intensities_by_layer, floor_width, floor_length, height_from_floor,
                       target_ppfd, perimeter_reflectivity, measurement_points, layer_weights, phase=1):
    # Determine active layers based on current floor dimensions
    active_layers = determine_active_layers(floor_width, floor_length)

    # Create adjusted intensities list where inactive layers are zeroed
    adjusted_intensities = list(light_intensities_by_layer)
    for i in range(active_layers, len(adjusted_intensities)):
        adjusted_intensities[i] = 0.0

    # Calculate intensity, MAD, and mean PPFD using adjusted intensities
    _, mad, mean_intensity = calculate_intensity(
        floor_width, floor_length, height_from_floor, adjusted_intensities, 
        perimeter_reflectivity, measurement_points
    )

    # Use piecewise penalty for PPFD: higher penalty for overshoot
    if mean_intensity <= target_ppfd:
        ppfd_penalty = (target_ppfd - mean_intensity) ** 2
    else:
        # Increase penalty for overshooting: adjust multiplier as needed
        overshoot_multiplier = 300  
        ppfd_penalty = overshoot_multiplier * (mean_intensity - target_ppfd) ** 2

    # Calculate intensity deviation penalty only over active layers
    intensity_dev_penalty = 0
    for i in range(active_layers):
        intensity = adjusted_intensities[i]
        target_intensity = (MAX_LUMENS / LUMENS_TO_PPFD_CONVERSION) * layer_weights[i]
        intensity_dev_penalty += (intensity - target_intensity)**2

    mad_scaled = mad * 0.6
    ppfd_penalty_scaled = ppfd_penalty * 0.2 if phase == 1 else 0
    intensity_dev_penalty_scaled = intensity_dev_penalty * 0.2

    objective_value = mad_scaled + ppfd_penalty_scaled + intensity_dev_penalty_scaled
    print(f"Phase: {phase}, MAD: {mad_scaled:.2f}, PPFD Penalty: {ppfd_penalty_scaled:.2f}, Intensity Deviation Penalty: {intensity_dev_penalty_scaled:.2f}")
    return objective_value


def optimize_lighting(floor_width, floor_length, height_from_floor,
                      initial_intensities_by_layer, target_ppfd, 
                      perimeter_reflectivity, measurement_points, verbose=False):

    active_layers = determine_active_layers(floor_width, floor_length)

    if verbose:
        print("Starting Optimization")

    bounds = []
    for i in range(NUM_LAYERS):
        if i < active_layers:
            bounds.append((MIN_LUMENS / LUMENS_TO_PPFD_CONVERSION, MAX_LUMENS / LUMENS_TO_PPFD_CONVERSION))
        else:
            bounds.append((0, 0))

    initial_guess = initial_intensities_by_layer
    constraints = []

    # Define layer weights to favor outer layers. Inner layers get lower weights.
    layer_weights = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0] 

    result = minimize(
        objective_function,
        initial_guess,
        args=(
            floor_width,
            floor_length,
            height_from_floor,
            target_ppfd,
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

    _, mad, optimized_ppfd = calculate_intensity(
        floor_width, floor_length, height_from_floor, optimized_intensities, 
        perimeter_reflectivity, measurement_points
    )

    if verbose:
        print(f"Optimization completed with MAD: {mad:.2f} and Optimized PPFD: {optimized_ppfd:.2f}")

    return optimized_intensities, mad, optimized_ppfd

def run_simulation(floor_width, floor_length, target_ppfd, perimeter_reflectivity, verbose=False):
    average_total_lumens = (MIN_LUMENS + MAX_LUMENS) / 2.0

    if NUM_LAYERS == 5:
        lumen_distribution_percentages = [0.0226, 0.0904, 0.1357, 0.0724, 0.6787]
        initial_intensities_first_five = [
            (average_total_lumens * pct) / LUMENS_TO_PPFD_CONVERSION
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
            initial_intensities.append((average_total_lumens * pct) / LUMENS_TO_PPFD_CONVERSION)
        for _ in range(6):
            initial_intensities.append((average_total_lumens * equal_share) / LUMENS_TO_PPFD_CONVERSION)
        initial_intensities_by_layer = initial_intensities
    else:
        initial_intensities_by_layer = [3000.0 / LUMENS_TO_PPFD_CONVERSION] * NUM_LAYERS

    measurement_points = generate_measurement_points(floor_width, floor_length, NUM_MEASUREMENT_POINTS)

    if verbose:
        print(f"Running simulation with floor width {floor_width}m, floor length {floor_length}m,")
        print(f"target PPFD {target_ppfd} µmol/m²/s, and perimeter reflectivity {perimeter_reflectivity}")

    optimized_intensities, mad, optimized_ppfd = optimize_lighting(
        floor_width, floor_length, HEIGHT_FROM_FLOOR,
        initial_intensities_by_layer, target_ppfd, DEFAULT_PERIMETER_REFLECTIVITY, measurement_points, verbose=verbose
    )

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
        "floor_length": floor_length,
        "target_ppfd": target_ppfd,
        "floor_height": HEIGHT_FROM_FLOOR,
        "perimeter_reflectivity": perimeter_reflectivity
    }

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
    except Exception as e:
        print(f"Simulation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
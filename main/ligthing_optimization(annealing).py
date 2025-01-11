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
from scipy.optimize import differential_evolution, dual_annealing, NonlinearConstraint
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
MIN_LUMENS = 1500.0
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

TOLERANCE = 5  # Define a tolerance for constraint satisfaction

def constraint_layer_5_highest(intensities):
    # Check if Layer 5 is the highest within a tolerance
    penalty = 0
    for i in range(len(intensities) - 1):
        if intensities[i] > intensities[4] - TOLERANCE:
            penalty += intensities[i] - (intensities[4] - TOLERANCE)
    return penalty

def constraint_layer_4_less_than_5(intensities):
    # Returns a positive value if Layer 4 is NOT less than Layer 5, 0 otherwise
    if intensities[3] >= intensities[4]:
        return intensities[3] - intensities[4]
    else:
        return 0

def constraint_layer_2_greater_than_3(intensities):
    # Returns a positive value if Layer 2 is NOT greater than Layer 3, 0 otherwise
    if intensities[1] <= intensities[2]:
        return intensities[2] - intensities[1]
    else:
        return 0

def constraint_layer_1_less_than_2_4_5(intensities):
    # Returns a positive value if Layer 1 is NOT less than Layers 2, 4, and 5, 0 otherwise
    penalty = 0
    if intensities[0] >= intensities[1]:
        penalty += intensities[0] - intensities[1]
    if intensities[0] >= intensities[3]:
        penalty += intensities[0] - intensities[3]
    if intensities[0] >= intensities[4]:
        penalty += intensities[0] - intensities[4]
    return penalty

# --- Heuristic for Initial Guess ---
def heuristic_initial_guess(floor_width, floor_length, target_ppfd, num_layers, active_layers, perimeter_reflectivity):
    intensities = [0.0] * num_layers
    if active_layers > 0:
        min_intensity = (MIN_LUMENS / LUMENS_TO_PPFD_CONVERSION) + 1

        # Base intensities considering reflectivity and layer impact
        reflectivity_adjustment = 1 - perimeter_reflectivity

        base_intensities = [
            max(0.7 * target_ppfd / active_layers * reflectivity_adjustment, min_intensity),
            max(1.2 * target_ppfd / active_layers * reflectivity_adjustment, min_intensity),
            max(0.9 * target_ppfd / active_layers * reflectivity_adjustment, min_intensity),
            max(1.3 * target_ppfd / active_layers * reflectivity_adjustment, min_intensity),
            max(1.5 * target_ppfd / active_layers * reflectivity_adjustment, min_intensity)
        ]

        for i in range(min(active_layers, num_layers)):
            intensities[i] = base_intensities[i]

    return intensities

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

# --- Generate Light Sources ---
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

# --- Calculate Spatial Penalty ---
def calculate_spatial_penalty(measurement_points, total_ppfd, floor_width, floor_length):
    print(f"calculate_spatial_penalty called with: measurement_points={measurement_points}, total_ppfd={total_ppfd}, floor_width={floor_width}, floor_length={floor_length}")
    """
    Calculates a spatial penalty based on PPFD differences between neighboring points.

    Args:
        measurement_points: List of (x, y) tuples representing measurement points.
        total_ppfd: Array of total PPFD values at each measurement point.

    Returns:
        The spatial penalty value.
    """
    penalty = 0
    num_points = len(measurement_points)
    ppfd_map = {point: ppfd for point, ppfd in zip(measurement_points, total_ppfd)}

    for i in range(num_points):
        for j in range(i + 1, num_points):
            point1 = measurement_points[i]
            point2 = measurement_points[j]

            # Calculate distance between points
            distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

            # Define a threshold for neighboring points (e.g., 10% of the average distance)
            neighbor_threshold = 0.1 * math.sqrt(floor_width**2 + floor_length**2)

            if distance > 0 and distance <= neighbor_threshold:
                ppfd_diff = abs(ppfd_map[point1] - ppfd_map[point2])

                # Edge-aware penalty: Increase penalty if points are near the edge
                edge_factor = 1.0
                edge_threshold = min(floor_width, floor_length) * 0.1  # 10% from the edge
                if (point1[0] < edge_threshold or point1[0] > floor_width - edge_threshold or
                    point1[1] < edge_threshold or point1[1] > floor_length - edge_threshold or
                    point2[0] < edge_threshold or point2[0] > floor_width - edge_threshold or
                    point2[1] < edge_threshold or point2[1] > floor_length - edge_threshold):
                    edge_factor = 1.5  # Increase penalty near edges

                # Normalize penalty by dividing by the average PPFD
                penalty += (ppfd_diff / distance) * edge_factor / np.mean(total_ppfd)
    print(f"Calculated spatial_penalty: {penalty}")
    return penalty

# --- Objective Function for Optimization ---
def objective_function(light_intensities_by_layer, floor_width, floor_length, height_from_floor,
                       target_ppfd, perimeter_reflectivity, measurement_points, active_layers):

    # --- Enforce bounds on light_intensities_by_layer ---
    clipped_intensities = np.clip(light_intensities_by_layer, [b[0] for b in bounds], [b[1] for b in bounds])

    # Ensure inactive layers are set to 0
    for i in range(active_layers, NUM_LAYERS):
        clipped_intensities[i] = 0.0

    # Use the clipped intensities for calculations
    _, mad, cv, spatial_penalty, mean_intensity = calculate_intensity_metrics(
        floor_width, floor_length, height_from_floor, clipped_intensities,
        perimeter_reflectivity, measurement_points
    )
   
    # Add a penalty for constraint violations
    constraint_penalty = (
        constraint_layer_5_highest(clipped_intensities) +
        constraint_layer_4_less_than_5(clipped_intensities) +
        constraint_layer_2_greater_than_3(clipped_intensities) +
        constraint_layer_1_less_than_2_4_5(clipped_intensities)
    )

    # Adaptive Penalty: Increase penalty over time
    # You can use the current iteration number from the optimizer to implement this
    # (e.g., by passing the iteration number as an argument or using a global variable)
    # For simplicity, we'll use a static multiplier here, but you should make it adaptive
    constraint_penalty_multiplier = 100

    # Adjust weights as needed
    ppfd_penalty = (mean_intensity - target_ppfd) ** 2
    objective_value = cv * 0.5 + ppfd_penalty * 0.2 + spatial_penalty * 0.1 + constraint_penalty * constraint_penalty_multiplier  # Increased weight

    print(f"Objective - Reached PPFD: {mean_intensity:.2f}, CV: {cv:.4f}, PPFD Penalty: {ppfd_penalty:.2f}, Spatial Penalty: {spatial_penalty:.2f}, Constraint Penalty: {constraint_penalty:.2f}")
    return objective_value

# --- Calculate Intensity, Variance, CV, and Spatial Penalty ---
def calculate_intensity_metrics(floor_width, floor_length, height_from_floor, light_intensities_by_layer,
                                perimeter_reflectivity, measurement_points):
    print(f"calculate_intensity_metrics called with: floor_width={floor_width}, floor_length={floor_length}, height_from_floor={height_from_floor}, light_intensities_by_layer={light_intensities_by_layer}, perimeter_reflectivity={perimeter_reflectivity}, measurement_points={measurement_points}")  # Add this line
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
    # Calculate Coefficient of Variation (CV)
    std_dev = np.std(total_ppfd)
    cv = (std_dev / mean_intensity) if mean_intensity > 0 else 0
    # Calculate Spatial Penalty
    spatial_penalty = calculate_spatial_penalty(measurement_points, total_ppfd, floor_width, floor_length)
    print(f"total_ppfd: {total_ppfd}, mad: {mad}, cv: {cv}, spatial_penalty: {spatial_penalty}, mean_intensity: {mean_intensity}")
    return total_ppfd, mad, cv, spatial_penalty, mean_intensity

# --- Optimization Process ---
def optimize_lighting(floor_width, floor_length, height_from_floor,
                      initial_intensities_by_layer, target_ppfd,
                      perimeter_reflectivity, measurement_points, verbose=False, workers=1):
    # Determine the number of active layers based on floor dimensions
    active_layers = determine_active_layers(floor_width, floor_length)

    # Set bounds: For active layers use normal bounds, inactive layers fixed to 0
    global bounds
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

    # First run with differential_evolution to get close to the optimal solution
    # Use 'workers' to control parallelism in differential_evolution
    result = differential_evolution(
        objective_function,
        bounds=bounds,
        args=(
            floor_width,
            floor_length,
            height_from_floor,
            target_ppfd,
            perimeter_reflectivity,
            measurement_points,
            active_layers
        ),
        maxiter=3000,  # Increased further
        popsize=40,    # Increased further
        tol=0.01,
        recombination=0.7,
        mutation=(0.5, 1),  # Experiment with mutation rate
        updating='deferred',
        workers=workers,  # Use workers to control parallelism
        # constraints=constraints, # Constraints handled in objective function now
        x0=initial_intensities_by_layer,
        polish=False
    )

    if not result.success:
        print(f"Warning: Initial optimization did not fully converge: {result.message}")

    # Refine the solution with dual_annealing
    initial_solution = result.x
    result = dual_annealing(
        objective_function,
        bounds=bounds,
        args=(
            floor_width,
            floor_length,
            height_from_floor,
            target_ppfd,
            perimeter_reflectivity,
            measurement_points,
            active_layers
        ),
        maxiter=2000,  # Increased further
        initial_temp=5230,  # Default value, you can experiment with this
        visit=2.0,  # Default value, you can experiment with this
        accept=-5.0,  # Default value, you can experiment with this
        x0=initial_solution,
    )

    if not result.success:
        error_msg = f"Optimization did not converge: {result.message}"
        print(error_msg, file=sys.stderr)
        raise Exception(error_msg)

    optimized_intensities_by_layer = result.x
    print(f"optimized_intensities_by_layer: {optimized_intensities_by_layer}")  # Add this line
    _, mad, cv, spatial_penalty, optimized_ppfd = calculate_intensity_metrics(
        floor_width, floor_length, height_from_floor, optimized_intensities_by_layer,
        perimeter_reflectivity, measurement_points
    )
    print(f"Calculated metrics: mad={mad}, cv={cv}, spatial_penalty={spatial_penalty}, optimized_ppfd={optimized_ppfd}")  # Add this line
    if verbose:
        print(f"Optimization completed with MAD: {mad:.2f}, CV: {cv:.4f}, Spatial Penalty: {spatial_penalty:.2f} and Optimized PPFD: {optimized_ppfd:.2f}")
    return optimized_intensities_by_layer, mad, optimized_ppfd

# --- Run Simulation ---
def run_simulation(floor_width, floor_length, target_ppfd, perimeter_reflectivity, verbose=False):
    active_layers = determine_active_layers(floor_width, floor_length)
    # Use the improved heuristic to generate initial intensities
    initial_intensities_by_layer = heuristic_initial_guess(
        floor_width, floor_length, target_ppfd, NUM_LAYERS, active_layers, perimeter_reflectivity
    )
    measurement_points = generate_measurement_points(floor_width, floor_length, NUM_MEASUREMENT_POINTS)

    if verbose:
        print(f"Running simulation with floor width {floor_width}ft, floor length {floor_length}ft,")
        print(f"target PPFD {target_ppfd} µmol/m²/s, and perimeter reflectivity {perimeter_reflectivity}")
        print(f"Initial intensities (PPFD) by layer: {initial_intensities_by_layer}")

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
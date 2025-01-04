#!/usr/bin/env python3
"""
Lighting Optimization Script with Perimeter Reflectivity and 5 Layers

This script optimizes lighting intensities across 5 layers to achieve
a uniform Photosynthetic Photon Flux Density (PPFD) distribution on the floor,
considering both direct light and reflected light from perimeter walls based on their reflectivity.

Usage:
    python lighting_optimization.py [--floor_width WIDTH] [--floor_length LENGTH]
                                    [--target_ppfd PPFD] [--perimeter_reflectivity REFLECTIVITY]
                                    [--verbose]

Example:
    python lighting_optimization.py --floor_width 12.0 --floor_length 15.0
                                    --target_ppfd 1200 --perimeter_reflectivity 0.3 --verbose

Author: Your Name
Date: 2025-01-04
"""

import math
import numpy as np
from scipy.optimize import minimize
import argparse
import sys

# --- Constants and Parameters ---
NUM_LAYERS = 5  # Reduced from 6 to 5 layers
NUM_MEASUREMENT_POINTS = 800  # Increased from 225 for finer grid
DEFAULT_FLOOR_WIDTH = 12.0
DEFAULT_FLOOR_LENGTH = 12.0  # Using length now
HEIGHT_FROM_FLOOR = 2.5  # Using height from floor now
BEAM_ANGLE_DEG = 120
DEFAULT_TARGET_PPFD = 1000.0
LUMENS_TO_PPFD_CONVERSION = 7
MAX_LUMENS = 15000.0  # Maximum luminous output constraint
MIN_LUMENS = 2000.0  # Minimum luminous output constraint
DEFAULT_PERIMETER_REFLECTIVITY = 0.9  # Default perimeter reflectivity (30%)

# --- Centered Square Layer Arrangement ---
def generate_light_sources(num_layers, center_x, center_y, spacing_x, spacing_y):
    """
    Generates light source coordinates for 5 concentric square layers.
    The first layer includes the central emitter and 4 surrounding lights.
    Subsequent layers form concentric squares around the center.

    Args:
        num_layers (int): Number of layers to generate.
        center_x (float): X-coordinate of the center.
        center_y (float): Y-coordinate of the center.
        spacing_x (float): Spacing between lights along the X-axis.
        spacing_y (float): Spacing between lights along the Y-axis.

    Returns:
        List[List[Tuple[float, float]]]: A list containing lists of (x, y) tuples for each layer.
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
            # Layers 2 to 5: Concentric squares around the center
            for x in range(-n, n + 1):
                for y in range(-n, n + 1):
                    if abs(x) == n or abs(y) == n:
                        layer_sources.append((center_x + x * spacing_x, center_y + y * spacing_y))
        light_sources.append(layer_sources)
    return light_sources

# --- Lambertian Emission Model ---
def lambertian_emission(intensity, distance, z):
    """
    Calculates radiant intensity at a point from a Lambertian source.

    Args:
        intensity (float): Intensity of the light source.
        distance (float): Distance from the light source to the point.
        z (float): Height from the floor.

    Returns:
        float: Radiant intensity at the point.
    """
    if distance == 0:
        return intensity
    return (intensity * z) / ((distance ** 2 + z ** 2) ** 1.5)

# --- Calculate PPFD at a Point ---
def calculate_ppfd_at_point(point, light_sources, light_intensities_by_layer, height_from_floor):
    """
    Calculates total direct PPFD at a point from all light sources.

    Args:
        point (Tuple[float, float]): (x, y) coordinates of the measurement point.
        light_sources (List[List[Tuple[float, float]]]): Light sources organized by layers.
        light_intensities_by_layer (List[float]): Intensities for each layer.
        height_from_floor (float): Height from the floor to the light sources.

    Returns:
        float: Total direct PPFD at the point.
    """
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
    """
    Generates measurement points within the floor area.

    Args:
        floor_width (float): Width of the floor area.
        floor_length (float): Length of the floor area.
        num_points (int): Total number of measurement points.

    Returns:
        List[Tuple[float, float]]: A list of (x, y) tuples representing measurement points.
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

# --- Calculate Reflected PPFD from Perimeter ---
def calculate_reflected_ppfd(measurement_points, light_sources, light_intensities_by_layer, 
                             height_from_floor, floor_width, floor_length, perimeter_reflectivity):
    """
    Calculates the reflected PPFD at each measurement point from perimeter walls.

    Args:
        measurement_points (List[Tuple[float, float]]): Measurement points on the floor.
        light_sources (List[List[Tuple[float, float]]]): Light sources organized by layers.
        light_intensities_by_layer (List[float]): Intensities for each layer.
        height_from_floor (float): Height from the floor to the light sources.
        floor_width (float): Width of the floor area.
        floor_length (float): Length of the floor area.
        perimeter_reflectivity (float): Reflectivity of the perimeter walls (0 to 1).

    Returns:
        np.ndarray: Reflected PPFD values for each measurement point.
    """
    # Initialize reflected PPFD array
    reflected_ppfd = np.zeros(len(measurement_points))

    # Define walls as boundaries
    walls = {
        'left': {'x': 0, 'y_range': (0, floor_length)},
        'right': {'x': floor_width, 'y_range': (0, floor_length)},
        'bottom': {'y': 0, 'x_range': (0, floor_width)},
        'top': {'y': floor_length, 'x_range': (0, floor_width)}
    }

    # Simplified reflection model:
    # Assume that a fixed fraction of the direct PPFD at edge points contributes to reflected PPFD.
    reflected_fraction = 0.1  # 10% reflection from walls

    # Define a threshold distance from walls to consider light hitting the walls
    edge_threshold = min(floor_width, floor_length) * 0.05  # 5% of the smallest dimension

    total_reflected_light = 0
    for idx, point in enumerate(measurement_points):
        # Check if the point is near the perimeter (within edge_threshold)
        near_left = point[0] <= edge_threshold
        near_right = point[0] >= (floor_width - edge_threshold)
        near_bottom = point[1] <= edge_threshold
        near_top = point[1] >= (floor_length - edge_threshold)
        is_near_wall = near_left or near_right or near_bottom or near_top

        if is_near_wall:
            # Calculate direct PPFD at this point
            direct_ppfd = calculate_ppfd_at_point(point, light_sources, light_intensities_by_layer, height_from_floor)
            # Assume a fraction is reflected
            reflected_light = direct_ppfd * reflected_fraction
            total_reflected_light += reflected_light

    # Total reflected light is scaled by perimeter_reflectivity
    total_reflected_light *= perimeter_reflectivity

    # Distribute the total reflected light uniformly across all measurement points
    if len(measurement_points) > 0:
        reflected_ppfd += total_reflected_light / len(measurement_points)

    return reflected_ppfd

# --- Calculate Intensity and Variance ---
def calculate_intensity(floor_width, floor_length, height_from_floor, light_array_width, light_array_height,
                       light_intensities_by_layer, perimeter_reflectivity, measurement_points):
    """
    Calculates photon distribution and variance given input parameters, including perimeter reflectivity.

    Args:
        floor_width (float): Width of the floor area.
        floor_length (float): Length of the floor area.
        height_from_floor (float): Height from the floor to the light sources.
        light_array_width (float): Width of the light array.
        light_array_height (float): Height of the light array.
        light_intensities_by_layer (List[float]): Intensities for each layer.
        perimeter_reflectivity (float): Reflectivity of the perimeter walls (0 to 1).
        measurement_points (List[Tuple[float, float]]): Measurement points on the floor.

    Returns:
        Tuple[np.ndarray, float, float]: Total PPFD, MAD, and mean PPFD.
    """
    spacing_x = light_array_width / (2 * NUM_LAYERS)  # Adjust spacing based on number of layers
    spacing_y = light_array_height / (2 * NUM_LAYERS)
    light_sources = generate_light_sources(NUM_LAYERS, floor_width / 2, floor_length / 2, spacing_x, spacing_y)

    # Calculate direct PPFD
    direct_ppfd = np.array([
        calculate_ppfd_at_point(point, light_sources, light_intensities_by_layer, height_from_floor)
        for point in measurement_points
    ])

    # Calculate reflected PPFD from perimeter
    reflected_ppfd = calculate_reflected_ppfd(
        measurement_points, light_sources, light_intensities_by_layer, 
        height_from_floor, floor_width, floor_length, perimeter_reflectivity
    )

    # Total PPFD is direct plus reflected
    total_ppfd = direct_ppfd + reflected_ppfd

    mean_intensity = np.mean(total_ppfd)
    mad = np.mean(np.abs(total_ppfd - mean_intensity))

    return total_ppfd, mad, mean_intensity

# --- Objective Function for Optimization ---
def objective_function(light_intensities_by_layer, floor_width, floor_length, height_from_floor,
                      light_array_width, light_array_height, target_ppfd, perimeter_reflectivity, measurement_points):
    """
    The function to be minimized, balancing uniformity and target PPFD.

    Args:
        light_intensities_by_layer (List[float]): Intensities for each layer.
        floor_width (float): Width of the floor area.
        floor_length (float): Length of the floor area.
        height_from_floor (float): Height from the floor to the light sources.
        light_array_width (float): Width of the light array.
        light_array_height (float): Height of the light array.
        target_ppfd (float): Target PPFD value.
        perimeter_reflectivity (float): Reflectivity of the perimeter walls (0 to 1).
        measurement_points (List[Tuple[float, float]]): Measurement points on the floor.

    Returns:
        float: Objective function value.
    """
    intensities, mad, mean_intensity = calculate_intensity(
        floor_width, floor_length, height_from_floor, light_array_width, light_array_height,
        light_intensities_by_layer, perimeter_reflectivity, measurement_points
    )

    # Debugging: print current mad and ppfd_penalty
    print(f"Objective Function - MAD: {mad:.2f}, PPFD Penalty: {(mean_intensity - target_ppfd)**2:.2f}")

    # Penalize deviation from target PPFD more strongly
    ppfd_penalty = (mean_intensity - target_ppfd) ** 2

    # Combine MAD and PPFD penalties with adjusted weights
    return mad * 1.0 + ppfd_penalty * 0.3  # Increased weight on ppfd_penalty

# --- Optimization Process ---
def optimize_lighting(floor_width, floor_length, height_from_floor, light_array_width, light_array_height,
                     initial_intensities_by_layer, target_ppfd, perimeter_reflectivity, measurement_points, verbose=False):
    """
    Optimizes light intensities to minimize MAD, considering the target PPFD and perimeter reflectivity.

    Args:
        floor_width (float): Width of the floor area.
        floor_length (float): Length of the floor area.
        height_from_floor (float): Height from the floor to the light sources.
        light_array_width (float): Width of the light array.
        light_array_height (float): Height of the light array.
        initial_intensities_by_layer (List[float]): Initial intensities for each layer.
        target_ppfd (float): Target PPFD value.
        perimeter_reflectivity (float): Reflectivity of the perimeter walls (0 to 1).
        measurement_points (List[Tuple[float, float]]): Measurement points on the floor.
        verbose (bool): If True, prints detailed optimization logs.

    Returns:
        Tuple[List[float], float, float]: Optimized intensities, MAD, and optimized PPFD.
    """
    bounds = [(MIN_LUMENS / LUMENS_TO_PPFD_CONVERSION, MAX_LUMENS / LUMENS_TO_PPFD_CONVERSION)] * NUM_LAYERS

    if verbose:
        print("Starting optimization...")
        print(f"Initial intensities (PPFD): {initial_intensities_by_layer}")
        print(f"Perimeter Reflectivity: {perimeter_reflectivity}")

    # Using 'SLSQP' method
    result = minimize(
        objective_function,
        initial_intensities_by_layer,
        args=(
            floor_width,
            floor_length,
            height_from_floor,
            light_array_width,
            light_array_height,
            target_ppfd,
            perimeter_reflectivity,
            measurement_points
        ),
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 10000, 'disp': verbose}
    )

    if not result.success:
        print("Optimization did not converge:", result.message, file=sys.stderr)
        sys.exit(1)

    optimized_intensities_by_layer = result.x
    _, mad, optimized_ppfd = calculate_intensity(
        floor_width, floor_length, height_from_floor, light_array_width, light_array_height,
        optimized_intensities_by_layer, perimeter_reflectivity, measurement_points
    )

    if verbose:
        print(f"Optimization completed with MAD: {mad:.2f} and Optimized PPFD: {optimized_ppfd:.2f}")

    return optimized_intensities_by_layer, mad, optimized_ppfd

# --- Run Simulation ---
def run_simulation(floor_width, floor_length, target_ppfd, perimeter_reflectivity, verbose=False):
    """
    Main function to run the optimization.

    Args:
        floor_width (float): Width of the floor area.
        floor_length (float): Length of the floor area.
        target_ppfd (float): Target PPFD value.
        perimeter_reflectivity (float): Reflectivity of the perimeter walls (0 to 1).
        verbose (bool): If True, prints detailed simulation logs.

    Returns:
        dict: Contains optimized lumens by layer, MAD, optimized PPFD, and input parameters.
    """
    light_array_width = floor_width
    light_array_height = floor_length
    initial_intensities_by_layer = [3000.0 / LUMENS_TO_PPFD_CONVERSION] * NUM_LAYERS  # Starting with higher lumens per layer
    measurement_points = generate_measurement_points(floor_width, floor_length, NUM_MEASUREMENT_POINTS)

    if verbose:
        print(f"Running simulation with floor width {floor_width}m, floor length {floor_length}m,")
        print(f"target PPFD {target_ppfd} µmol/m²/s, and perimeter reflectivity {perimeter_reflectivity}")

    optimized_intensities, mad, optimized_ppfd = optimize_lighting(
        floor_width, floor_length, HEIGHT_FROM_FLOOR, light_array_width, light_array_height,
        initial_intensities_by_layer, target_ppfd, perimeter_reflectivity, measurement_points, verbose=verbose
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
        "perimeter_reflectivity": perimeter_reflectivity
    }

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Optimize lighting intensities for uniform PPFD distribution, considering perimeter reflectivity and 5 layers.")
    parser.add_argument('--floor_width', type=float, default=DEFAULT_FLOOR_WIDTH,
                        help='Width of the floor area (meters). Default is 12.0')
    parser.add_argument('--floor_length', type=float, default=DEFAULT_FLOOR_LENGTH,
                        help='Length of the floor area (meters). Default is 12.0')
    parser.add_argument('--target_ppfd', type=float, default=DEFAULT_TARGET_PPFD,
                        help='Target PPFD (µmol/m²/s). Default is 1000.0')
    parser.add_argument('--perimeter_reflectivity', type=float, default=DEFAULT_PERIMETER_REFLECTIVITY,
                        help='Perimeter wall reflectivity (0 to 1). Default is 0.3')
    parser.add_argument('--verbose', action='store_true',
                        help='Increase output verbosity')
    args = parser.parse_args()

    # Parameter Validation
    if args.floor_width <= 0 or args.floor_length <= 0:
        print("Error: Floor dimensions must be positive numbers.", file=sys.stderr)
        sys.exit(1)

    if args.target_ppfd <= 0:
        print("Error: Target PPFD must be a positive number.", file=sys.stderr)
        sys.exit(1)

    if not (0.0 <= args.perimeter_reflectivity <= 1.0):
        print("Error: Perimeter reflectivity must be between 0 and 1.", file=sys.stderr)
        sys.exit(1)

    run_simulation(args.floor_width, args.floor_length, args.target_ppfd,
                  args.perimeter_reflectivity, verbose=args.verbose)

if __name__ == '__main__':
    main()

import numpy as np
from scipy.optimize import minimize
import math

# --- Constants ---
SPACING_FT = 1.667           # Spacing between LEDs in feet
PPFD_CONVERSION_FACTOR = 20  # Assume a general conversion factor for our LEDs
FT_TO_M = 0.3048             # Conversion from feet to meters

# --- LED Arrangement (Diamond-61 Pattern) ---
def build_diamond_61_pattern():
    raw_pattern = [
        [(0, 0)],
        [(-1, 0), (1, 0), (0, -1), (0, 1)],
        [(-1, -1), (1, -1), (-1, 1), (1, 1),
         (-2, 0), (2, 0), (0, -2), (0, 2)],
        [(-2, -1), (2, -1), (-2, 1), (2, 1),
         (-1, -2), (1, -2), (-1, 2), (1, 2),
         (-3, 0), (3, 0), (0, -3), (0, 3)],
        [(-2, -2), (2, -2), (-2, 2), (2, 2),
         (-3, -1), (3, -1), (-3, 1), (3, 1),
         (-1, -3), (1, -3), (-1, 3), (1, 3),
         (-4, 0), (4, 0), (0, -4), (0, 4)],
        [(-3, -2), (3, -2), (-3, 2), (3, 2),
         (-2, -3), (2, -3), (-2, 3), (2, 3),
         (-4, -1), (4, -1), (-4, 1), (4, 1),
         (-1, -4), (1, -4), (-1, 4), (1, 4),
         (-5, 0), (5, 0), (0, -5), (0, 5)]
    ]
    # Combine central element and first ring for layer 1
    layer1 = raw_pattern[0] + raw_pattern[1]
    pattern = [layer1] + raw_pattern[2:]
    return pattern

# --- Convert LED Pattern to Meters ---
def convert_pattern_to_meters(pattern):
    return [
        [(x * SPACING_FT * FT_TO_M, y * SPACING_FT * FT_TO_M) for (x, y) in layer]
        for layer in pattern
    ]

# --- Generate Measurement Points ---
def generate_measurement_points(floor_width_ft, floor_length_ft, resolution_ft=0.5):
    x_coords = np.arange(0, floor_width_ft, resolution_ft) * FT_TO_M
    y_coords = np.arange(0, floor_length_ft, resolution_ft) * FT_TO_M
    points = np.array([(x, y, 0) for x in x_coords for y in y_coords])
    return points

# --- Calculate Distance Between LED and Measurement Point ---
def distance(led_pos, point, light_height_m):
    dx = led_pos[0] - point[0]
    dy = led_pos[1] - point[1]
    dz = light_height_m - point[2]
    return np.sqrt(dx*dx + dy*dy + dz*dz)

# --- Calculate Intensity at a Measurement Point from a Single LED ---
def intensity_at_point(led_pos, point, light_height_m, efficiency, power):
    d = distance(led_pos, point, light_height_m)
    if d == 0:
        return 0  # Avoid division by zero
    luminous_flux = power * efficiency  # Convert power to luminous flux using efficiency
    return luminous_flux / (4 * np.pi * d * d)  # Inverse square law

# --- Calculate PPFD at a Measurement Point from All LEDs ---
def ppfd_at_point(point, led_pattern_m, layer_intensities, light_height_m, efficiency, base_power):
    total_intensity = 0.0
    for layer_idx, layer in enumerate(led_pattern_m):
        layer_power = base_power * layer_intensities[layer_idx] # Each layer has a base power
        for led in layer:
            total_intensity += intensity_at_point(led, point, light_height_m, efficiency, layer_power)
    return total_intensity / PPFD_CONVERSION_FACTOR  # Convert intensity to PPFD

# --- Calculate Average PPFD and MAD ---
def calculate_metrics(measurement_points, led_pattern_m, layer_intensities, light_height_m, efficiency, base_power):
    ppfd_values = []
    for pt in measurement_points:
        ppfd_values.append(ppfd_at_point(pt, led_pattern_m, layer_intensities, light_height_m, efficiency, base_power))
    ppfd_values = np.array(ppfd_values)
    ppfd_avg = np.mean(ppfd_values)
    mad = np.mean(np.abs(ppfd_values - ppfd_avg))  # Mean Absolute Deviation
    return ppfd_avg, mad

# --- Objective Function for Optimization ---
def objective(layer_intensities, measurement_points, led_pattern_m, light_height_m, efficiency, base_power, target_ppfd):
    ppfd_avg, mad = calculate_metrics(measurement_points, led_pattern_m, layer_intensities, light_height_m, efficiency, base_power)
    
    # Penalty for not reaching target PPFD (if needed)
    penalty = 0
    if ppfd_avg < target_ppfd:
        penalty = (target_ppfd - ppfd_avg) * 1e6  # Large penalty to enforce target
    
    return mad + penalty  # Minimize MAD, and add penalty if below target PPFD

# --- Optimization Function ---
def optimize_intensities(floor_width_ft, floor_length_ft, light_height_ft, efficiency, base_power, target_ppfd):
    light_height_m = light_height_ft * FT_TO_M
    pattern = build_diamond_61_pattern()
    led_pattern_m = convert_pattern_to_meters(pattern)

    num_layers = len(led_pattern_m)
    measurement_points = generate_measurement_points(floor_width_ft, floor_length_ft)

    # Initial guess for optimization (uniform distribution)
    initial_guess = np.full(num_layers, target_ppfd)

    # Optimization (no bounds, we'll clip negative values later)
    result = minimize(
        objective,
        initial_guess,
        args=(measurement_points, led_pattern_m, light_height_m, efficiency, base_power, target_ppfd),
        method='SLSQP',
        options={'ftol': 1e-6, 'maxiter': 1000}  # Increased iterations
    )

    # Clip negative intensities (if any) to zero and round to 2 decimal places
    optimized_multipliers = np.clip(result.x, 0, None)
    optimized_multipliers = np.round(optimized_multipliers * 100) / 100

    ppfd_avg, mad = calculate_metrics(measurement_points, led_pattern_m, optimized_multipliers, light_height_m, efficiency, base_power)

    # Convert layer multipliers to lumens for each layer
    optimized_lumens = optimized_multipliers * base_power * efficiency

    return optimized_lumens, mad, ppfd_avg, result

# --- Main Execution ---
if __name__ == "__main__":
    floor_width_ft = 10
    floor_length_ft = 10
    light_height_ft = 5
    efficiency = 100    # Example: 100 lumens per watt
    base_power = 1      # Base power per LED element (in watts)
    target_ppfd = 200   # Target PPFD

    optimized_lumens, mad, ppfd_avg, optimization_result = optimize_intensities(
        floor_width_ft, floor_length_ft, light_height_ft, efficiency, base_power, target_ppfd
    )

    # Output the results
    print("Optimized Layer Intensities (lumens):", optimized_lumens)
    print("Mean Absolute Deviation (MAD):", mad)
    print("Average PPFD:", ppfd_avg)
    print("Optimization Success:", optimization_result.success)
    print("Objective Function Value (MAD + penalties):", optimization_result.fun)
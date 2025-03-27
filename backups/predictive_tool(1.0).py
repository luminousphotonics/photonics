#!/usr/bin/env python3
import numpy as np
from scipy.interpolate import SmoothBivariateSpline
import warnings
import math # Added for build_cob_positions if needed elsewhere

# ---------------------------
# NEW Known successful configurations (Flux Params per Layer) - Validated Data (June 2024)
known_configs = {
    10: np.array([ # W=6.096m (20ft), H=0.9144m, PPFD=1248.11
        8000, 9000, 11000, 14500, 13500, 8000, 5500, 13500, 20000, 20000
    ], dtype=np.float64),
    11: np.array([ # W=6.7056m (22ft), H=0.9144m, PPFD=1250.48
        9500, 9000, 7000, 12000, 16000, 14000, 3500, 6500, 14000, 20000, 20000
    ], dtype=np.float64),
    12: np.array([ # W=7.3152m (24ft), H=0.9144m, PPFD=1248.60
        9500, 13000, 11000, 3000, 15000, 17000, 8000, 4000, 10000, 12000, 20000, 20000
    ], dtype=np.float64),
    13: np.array([ # W=7.9248m (26ft), H=0.9144m, PPFD=1259.86
        9500, 13000, 11000, 3000, 13000, 17000, 12000, 6000, 7000, 10000, 11000, 20000, 20000
    ], dtype=np.float64),
    14: np.array([ # W=8.5344m (28ft), H=0.9144m, PPFD=1259.84
        8000, 14000, 14000, 4000, 7000, 15000, 14000, 13000, 7000, 8000, 7000, 11000, 20000, 20000
    ], dtype=np.float64),
    15: np.array([ # W=9.144m (30ft), H=0.9144m, PPFD=1250.59
        8000, 12000, 15000, 6000, 9000, 5000, 13000, 17500, 9500, 8000, 7000, 7000, 11500, 20000, 20000
    ], dtype=np.float64),
    16: np.array([ # W=9.7536m (32ft), H=0.9144m, PPFD=1252.00
        6000, 8000, 11000, 13000, 15000, 4000, 2000, 16500, 15500, 11000, 7000, 7000, 6000, 12000, 20000, 20000
    ], dtype=np.float64),
    17: np.array([ # W=9.10.3632 (34ft), H=0.9144m, PPFD=1249.59
    4612.7476, 10601.7316, 12225.2797, 11233.8901, 9378.0608, 8406.3522, 9253.4791, 10783.1226, 11575.3645, 11181.6309, 10140.7523, 9035.7677, 8449.7164, 8965.6376, 11166.5705, 15635.5543, 20000.0000,     
    ], dtype=np.float64),
    18: np.array([ # W=9.10.9728 (36ft), H=0.9144m, PPFD=1249.59
    4479.0225, 10412.5155, 12146.2339, 11298.3421, 9487.0047, 8330.0463, 8847.3014, 10230.1356, 11321.7999, 11317.6453, 10516.8130, 9434.1654, 8584.5645, 8478.1138, 9451.9112, 11625.0138, 15102.5620, 20000.0000,
    ], dtype=np.float64),
    19: np.array([ # W=9.11.5824 (38ft), H=0.9144m, PPFD=1249.59
    4457.8784, 10211.6949, 12042.3028, 11376.5219, 9641.1723, 8263.0738, 8379.0505, 9554.9527, 10827.1349, 11300.8096, 10877.1854, 9966.6900, 8988.0460, 8359.9758, 8486.8064, 9589.4539, 11762.4015, 15097.6294, 20000.0000,
    ], dtype=np.float64),
    # Removed old N=17, N=18 data
}
# Target Average PPFD
TARGET_PPFD = 1250.0

# Desired number of layers for prediction
# --- Target N=17 (Reduced from 20 for less extrapolation) ---
NUM_LAYERS_TARGET = 22
# --- Target N=17 ---

EXTRAPOLATION_BOOST_FACTOR = 1.033 # Boost for predicted flux when extrapolating


# Spline parameters
SPLINE_DEGREE = 3
SMOOTHING_FACTOR_MODE = "multiplier"
SMOOTHING_MULTIPLIER = 1.5 # Start with this, may need tuning

# PPFD/Lumen Ratio Fitting parameters
RATIO_FIT_DEGREE = 2 # Seems reasonable for 7 data points (N=10 to N=16)

# Clamping Parameters
CLAMP_OUTER_LAYER = True
OUTER_LAYER_MAX_FLUX = 20000.0 # Keep clamp, new data also has high outer flux

# --- Parameters for potential future use ---
# Empirical PPFD correction - currently none based on new data
EMPIRICAL_PPFD_CORRECTION = {
    #18: 1.03225 # Correction factor based on 1210.95 achieved vs 1250 target
    # Keep previous entries if you want, but they aren't used once added to known_configs
    # 17: 1.0251 # No longer needed as N=17 is in known_configs
}

# --- Refinement is OFF for initial prediction ---
APPLY_PPFD_REFINEMENT = False
NUM_REFINEMENT_ITERATIONS = 0
REFINEMENT_LEARNING_RATE = 0.40 # Placeholder
MULTIPLIER_MIN = 0.7 # Placeholder
MULTIPLIER_MAX = 1.3 # Placeholder
# --- Parameters ---

# ---------------------------

# --- Helper function for COB geometry (Included for context, not directly used in prediction logic) ---
def build_cob_positions(FIXED_NUM_LAYERS, W, L, H):
    n = FIXED_NUM_LAYERS - 1
    positions = []
    # Center COB
    positions.append((0, 0, H, 0)) # (x, y, z, layer_index)

    # Layers 1 to n
    for i in range(1, n + 1):
        # Add COBs along axes
        positions.append((i, 0, H, i))
        positions.append((-i, 0, H, i))
        positions.append((0, i, H, i))
        positions.append((0, -i, H, i))
        # Add COBs in quadrants
        for x in range(1, i):
            y = i - x
            positions.append((x, y, H, i))
            positions.append((-x, y, H, i))
            positions.append((x, -y, H, i))
            positions.append((-x, -y, H, i))

    # The provided formula a(n) = 2*n*(n+1)+1 describes the *number* of COBs up to layer n.
    # The generation above seems different, creating square-like layers.
    # Let's assume the Python code generating COB positions *is* the ground truth for layer definition.

    # Apply scaling and rotation if needed (Example similar to provided)
    theta = math.radians(0) # Example: No rotation
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    centerX = W / 2
    centerY = L / 2
    # Scaling might need adjustment based on how 'layers' relate to physical spread
    scale_x = W / (2 * n) if n > 0 else W
    scale_y = L / (2 * n) if n > 0 else L

    transformed = []
    for (xx, yy, hh, layer) in positions:
        # Apply rotation if theta != 0
        rx = xx * cos_t - yy * sin_t
        ry = xx * sin_t + yy * cos_t
        # Apply scaling and centering
        px = centerX + rx * scale_x
        py = centerY + ry * scale_y
        # Apply height adjustment if needed
        pz = hh * 1.0 # Example: Use H directly
        transformed.append((px, py, pz, layer))

    # Sort by layer, then optionally by position within layer for consistency
    transformed.sort(key=lambda p: (p[3], p[0], p[1]))

    return np.array(transformed, dtype=np.float64)
# --- End Helper function ---


def prepare_spline_data(known_configs):
    n_list = []
    x_norm_list = []
    flux_list = []
    min_n, max_n = float('inf'), float('-inf')

    sorted_keys = sorted(known_configs.keys()) # Process in order
    print("Preparing spline data from configurations:")
    for n in sorted_keys:
        fluxes = known_configs[n]
        if n < 2: continue # Need at least 2 layers
        num_layers_in_config = len(fluxes)
        if num_layers_in_config != n:
             warnings.warn(f"Mismatch for N={n}. Expected {n} fluxes, got {num_layers_in_config}. Using actual count {num_layers_in_config}.", UserWarning)
             n_actual = num_layers_in_config
             if n_actual < 2: continue
        else:
             n_actual = n
        print(f"  Using N={n_actual} data ({len(fluxes)} flux values)")
        min_n = min(min_n, n_actual)
        max_n = max(max_n, n_actual)
        norm_positions = np.linspace(0, 1, n_actual) # Normalized layer index (0=center, 1=outer)
        for i, flux in enumerate(fluxes):
            n_list.append(n_actual)
            x_norm_list.append(norm_positions[i])
            flux_list.append(flux)

    if not n_list:
        raise ValueError("No valid configuration data found in known_configs.")

    num_data_points = len(flux_list)
    print(f"Spline fitting using data from N={min_n} to N={max_n}. Total points: {num_data_points}")
    return np.array(n_list), np.array(x_norm_list), np.array(flux_list), min_n, max_n, num_data_points

def fit_and_predict_ratio(n_target, known_configs, target_ppfd=TARGET_PPFD, poly_degree=RATIO_FIT_DEGREE):
    n_values_sorted = sorted(known_configs.keys())
    total_fluxes = [np.sum(known_configs[n]) for n in n_values_sorted]
    n_array = np.array(n_values_sorted)

    ratios = []
    valid_n = []
    # --- Update with N=18 actual PPFD ---
    actual_ppfds = {
        10: 1248.11,
        11: 1250.48,
        12: 1248.60,
        13: 1259.86,
        14: 1259.84,
        15: 1250.59,
        16: 1252.00,
        17: 1249.59, # From validated N=17 run
        18: 1249.52, # From validated N=18 run (using corrected fluxes)
        19: 1246.90,
    }

    print("\nCalculating PPFD/Lumen ratio and fitting trend vs N:")
    for i, n in enumerate(n_values_sorted):
        # Ensure we are using the correct total flux for the validated configs
        # The total flux stored in known_configs should correspond to the actual_ppfd
        current_total_flux = np.sum(known_configs[n])
        if current_total_flux > 1e-6:
            ppfd_to_use = actual_ppfds.get(n)
            if ppfd_to_use is None:
                 warnings.warn(f"Missing actual PPFD for N={n}. Using target_ppfd.", UserWarning)
                 ppfd_to_use = target_ppfd
            # Use the *actual* total flux that yielded the actual PPFD
            ratio = ppfd_to_use / current_total_flux
            ratios.append(ratio)
            valid_n.append(n)
            print(f"  N={n}: Total Flux={current_total_flux:.1f}, Actual PPFD={ppfd_to_use:.2f}, Ratio={ratio:.6e}")
        else:
            print(f"  N={n}: Total Flux is zero, skipping.")

    if not ratios:
        raise ValueError("Could not calculate PPFD/Lumen ratio from any configuration.")

    num_valid_points = len(valid_n)
    if num_valid_points <= poly_degree:
        original_degree = poly_degree
        poly_degree = max(0, num_valid_points - 1)
        warnings.warn(f"Not enough valid data points ({num_valid_points}) for degree {original_degree}. Reducing degree to {poly_degree}.", UserWarning)

    valid_n_array = np.array(valid_n)
    ratios_array = np.array(ratios)

    predicted_ratio = 0
    if poly_degree >= 0:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)
            coeffs = np.polyfit(valid_n_array, ratios_array, poly_degree)
        poly_func = np.poly1d(coeffs)
        predicted_ratio = poly_func(n_target)
        print(f"\nFitted Ratio Trend (Degree {poly_degree}): Ratio = {poly_func}")
    else:
        warnings.warn("Cannot fit ratio trend (0 points). Using mean ratio.", UserWarning)
        predicted_ratio = np.mean(ratios) if ratios else 0

    if predicted_ratio <= 1e-9:
         warnings.warn(f"Ratio trend predicted non-positive ratio ({predicted_ratio:.4e}) for N={n_target}. Using nearest known N ratio.", UserWarning)
         try:
             nearest_n_idx = np.abs(valid_n_array - n_target).argmin()
             predicted_ratio = ratios_array[nearest_n_idx]
             print(f"Using ratio from nearest N ({valid_n_array[nearest_n_idx]}): {predicted_ratio:.6e}")
         except (IndexError, ValueError):
              print("Error finding nearest N ratio, returning 0.")
              predicted_ratio = 0

    print(f"Predicted PPFD/Lumen Ratio for N={n_target}: {predicted_ratio:.6e}")
    return predicted_ratio

# --- apply_inner_refinement_step, apply_global_refinement_step ---
# (No changes needed in these functions - they are inactive anyway)
def apply_inner_refinement_step(flux_input, ppfd_feedback, target_ppfd, target_total_flux, outer_clamp_value, iteration, learn_rate=REFINEMENT_LEARNING_RATE, mult_min=MULTIPLIER_MIN, mult_max=MULTIPLIER_MAX):
    num_layers = len(flux_input)
    if ppfd_feedback is None or len(ppfd_feedback) != num_layers:
        warnings.warn(f"PPFD feedback invalid for inner refinement iteration {iteration}. Skipping step.", UserWarning)
        return flux_input

    print(f"\n--- Applying INNER PPFD Refinement Iteration {iteration} ---")
    print(f"Using Learning Rate: {learn_rate:.2f}")
    errors = target_ppfd - ppfd_feedback
    relative_errors = np.divide(errors, target_ppfd, out=np.zeros_like(errors), where=abs(target_ppfd)>1e-9)
    multipliers = 1.0 + learn_rate * relative_errors
    multipliers = np.clip(multipliers, mult_min, mult_max)

    print(f"Refinement Multipliers (Iter {iteration}, min={mult_min:.2f}, max={mult_max:.2f}):")
    with np.printoptions(precision=4, suppress=True):
         print(multipliers[:-1]) # Only show inner multipliers

    flux_inner_refined = flux_input[:-1] * multipliers[:-1]
    flux_inner_refined = np.maximum(flux_inner_refined, 0)

    inner_flux_budget = target_total_flux - outer_clamp_value
    if inner_flux_budget < 0:
         warnings.warn(f"Target total flux ({target_total_flux:.2f}) is less than outer clamp ({outer_clamp_value:.2f}) in iter {iteration}. Setting inner flux to zero.", UserWarning)
         inner_flux_budget = 0
         flux_inner_refined[:] = 0 # Set all inner fluxes to 0

    refined_inner_total = np.sum(flux_inner_refined)
    print(f"Refined Inner Flux Total (Iter {iteration}, before rescale): {refined_inner_total:.2f}")

    rescale_inner_factor = 0
    if refined_inner_total > 1e-6:
         rescale_inner_factor = inner_flux_budget / refined_inner_total
    elif inner_flux_budget > 1e-6 : # Handle case where budget exists but refined sum is zero
         warnings.warn(f"Refined inner flux sum is zero in iteration {iteration}, but budget exists. Cannot rescale.", UserWarning)
         # Keep flux_inner_refined as zeros
    # else: both are zero, factor remains 0

    print(f"Inner rescale factor (Iter {iteration}): {rescale_inner_factor:.6f}")
    final_fluxes_inner = flux_inner_refined * rescale_inner_factor
    final_fluxes_inner = np.maximum(final_fluxes_inner, 0)

    final_fluxes = np.zeros_like(flux_input)
    final_fluxes[:-1] = final_fluxes_inner
    final_fluxes[-1] = outer_clamp_value # Ensure clamp is reapplied

    # Final check on total flux
    actual_total_flux_iter = np.sum(final_fluxes)
    if not np.isclose(actual_total_flux_iter, target_total_flux, rtol=1e-3):
         warnings.warn(f"Total flux ({actual_total_flux_iter:.2f}) after inner refinement iter {iteration} differs from target ({target_total_flux:.2f}). Target budget: {inner_flux_budget + outer_clamp_value:.2f}", UserWarning)

    return final_fluxes

def apply_global_refinement_step(flux_input, ppfd_feedback, target_ppfd, target_total_flux, iteration, learn_rate=REFINEMENT_LEARNING_RATE, mult_min=MULTIPLIER_MIN, mult_max=MULTIPLIER_MAX):
    if ppfd_feedback is None or len(ppfd_feedback) != len(flux_input):
        warnings.warn(f"PPFD feedback invalid for global refinement iteration {iteration}. Skipping step.", UserWarning)
        return flux_input

    print(f"\n--- Applying GLOBAL PPFD Refinement Iteration {iteration} ---")
    print(f"Using Learning Rate: {learn_rate:.2f}")
    errors = target_ppfd - ppfd_feedback
    relative_errors = np.divide(errors, target_ppfd, out=np.zeros_like(errors), where=abs(target_ppfd)>1e-9)
    multipliers = 1.0 + learn_rate * relative_errors
    multipliers = np.clip(multipliers, mult_min, mult_max)

    print(f"Refinement Multipliers (Iter {iteration}, min={mult_min:.2f}, max={mult_max:.2f}):")
    with np.printoptions(precision=4, suppress=True):
         print(multipliers)

    flux_refined = flux_input * multipliers
    flux_refined = np.maximum(flux_refined, 0) # Ensure non-negative

    refined_total_flux = np.sum(flux_refined)
    print(f"Refined Flux Total (Iter {iteration}, before final rescale): {refined_total_flux:.2f}")

    final_fluxes_iter = np.zeros_like(flux_input) # Default to zeros if scaling fails
    if refined_total_flux > 1e-6:
         final_rescale_factor = target_total_flux / refined_total_flux
         print(f"Final rescale factor (Iter {iteration}): {final_rescale_factor:.6f}")
         final_fluxes_iter = flux_refined * final_rescale_factor
    elif target_total_flux > 1e-6:
         warnings.warn(f"Refined flux total near zero in iteration {iteration}, but target is non-zero. Result will be zero.", UserWarning)
         # final_fluxes_iter remains zeros
    # else: both target and refined sum are near zero, zeros is fine.

    final_fluxes_iter = np.maximum(final_fluxes_iter, 0) # Ensure non-negative again after scaling

    # Final check
    actual_total_flux_iter = np.sum(final_fluxes_iter)
    if not np.isclose(actual_total_flux_iter, target_total_flux, rtol=1e-3):
         warnings.warn(f"Total flux ({actual_total_flux_iter:.2f}) after global refinement iter {iteration} differs significantly from target ({target_total_flux:.2f}).", UserWarning)

    return final_fluxes_iter

# --- Main Prediction Function ---
# Renamed from generate_flux_assignments_v23
def predict_flux_assignments(num_layers_target, known_configs, target_ppfd=TARGET_PPFD, k=SPLINE_DEGREE, smoothing_mode=SMOOTHING_FACTOR_MODE, smoothing_mult=SMOOTHING_MULTIPLIER, clamp_outer=CLAMP_OUTER_LAYER, outer_max=OUTER_LAYER_MAX_FLUX, apply_refinement=APPLY_PPFD_REFINEMENT, ppfd_feedback_list=None, num_refine_iter=NUM_REFINEMENT_ITERATIONS, learn_rate=0.0, mult_min=0.0, mult_max=0.0, extrapolation_boost=EXTRAPOLATION_BOOST_FACTOR): # Added boost parameter
    """
    Predicts flux assignments using SmoothBivariateSpline.
    Includes a proactive boost factor for extrapolation.
    Refinement step is disabled.
    """
    if num_layers_target < 2:
        raise ValueError("Number of layers must be at least 2.")

    # --- Step 1: Prepare data and fit spline ---
    n_coords, x_norm_coords, flux_values, min_n_data, max_n_data, num_data_points = prepare_spline_data(known_configs)
    # ... (spline smoothing factor calculation - unchanged) ...
    s_value = None
    if smoothing_mode == "num_points":
        s_value = float(num_data_points)
        print(f"Using explicit spline smoothing factor s = {s_value:.1f}")
    elif smoothing_mode == "multiplier":
        s_value = float(num_data_points) * smoothing_mult
        print(f"Using smoothing factor s = {s_value:.1f} ({smoothing_mult} * num_points)")
    elif isinstance(smoothing_mode, (int, float)):
        s_value = float(smoothing_mode)
        print(f"Using explicit spline smoothing factor s = {s_value:.1f}")
    else:
        print("Using default spline smoothing factor s (estimated by FITPACK)")

    unique_n = len(np.unique(n_coords))
    unique_x = len(np.unique(x_norm_coords))
    current_k = k
    if unique_n <= k or unique_x <= k:
         new_k = min(unique_n - 1, unique_x - 1, current_k)
         new_k = max(1, new_k)
         warnings.warn(f"Insufficient unique coordinates for degree {current_k}. Reducing degree to kx=ky={new_k}.", UserWarning)
         current_k = new_k

    try:
        with warnings.catch_warnings():
             warnings.filterwarnings("ignore", message="The required storage space", category=UserWarning)
             warnings.filterwarnings("ignore", message="The number of knots required", category=UserWarning)
             warnings.filterwarnings("ignore", message="Warning: s=", category=UserWarning)
             warnings.filterwarnings("ignore", message=".*ier=.*", category=UserWarning)
             spline = SmoothBivariateSpline(n_coords, x_norm_coords, flux_values, kx=current_k, ky=current_k, s=s_value)
    except Exception as e:
         print(f"\nError fitting spline: {e}")
         raise

    # --- Step 2: Evaluate spline, predict ratio, calculate target flux ---
    x_target_norm = np.linspace(0, 1, num_layers_target)
    n_target_array = np.full_like(x_target_norm, num_layers_target)
    interpolated_fluxes = spline(n_target_array, x_target_norm, grid=False)

    is_extrapolating = (num_layers_target > max_n_data)
    if is_extrapolating:
        warnings.warn(f"Target N={num_layers_target} is outside the range of known data (N={min_n_data} to N={max_n_data}). Extrapolating flux profile.", UserWarning)

    interpolated_fluxes = np.maximum(interpolated_fluxes, 0)
    spline_shape_total_flux = np.sum(interpolated_fluxes)
    print(f"\nSpline evaluation for N={num_layers_target} yielded raw profile shape (Total Flux: {spline_shape_total_flux:.2f})")

    if spline_shape_total_flux <= 1e-6:
         warnings.warn("Spline evaluation resulted in near-zero total flux. Cannot scale.", UserWarning)
         return interpolated_fluxes

    predicted_ppfd_per_lumen = fit_and_predict_ratio(num_layers_target, known_configs, target_ppfd)
    if predicted_ppfd_per_lumen <= 1e-9:
         warnings.warn("Predicted PPFD/Lumen ratio is non-positive. Cannot scale.", UserWarning)
         return interpolated_fluxes

    # Calculate initial target total flux
    target_total_flux = target_ppfd / predicted_ppfd_per_lumen
    print(f"Initial Target Total Flux (based on ratio fit): {target_total_flux:.2f} lumens")

    # --- APPLY PROACTIVE EXTRAPOLATION BOOST ---
    if is_extrapolating:
        print(f"Applying proactive extrapolation boost factor: {extrapolation_boost:.4f}")
        target_total_flux *= extrapolation_boost
        print(f"Boosted Target Total Flux for N={num_layers_target}: {target_total_flux:.2f} lumens")
    else:
        print(f"Target Total Flux (interpolation or within data range): {target_total_flux:.2f} lumens")
    # --- End Boost Application ---


    # --- Step 3: Apply Initial Scaling / Clamping (using potentially boosted flux) ---
    flux_current = np.zeros_like(interpolated_fluxes)
    outer_clamp_value_to_use = 0.0

    if clamp_outer:
        # Use the ORIGINAL outer_max value defined at the top
        print(f"Applying outer layer clamp: Layer {num_layers_target - 1} max flux = {outer_max:.1f} lumens")

        if outer_max >= target_total_flux: # Compare ORIGINAL clamp to BOOSTED target
             warnings.warn(f"Outer layer clamp value ({outer_max:.1f}) >= boosted target total flux ({target_total_flux:.1f}). Setting outer layer to target total.", UserWarning)
             flux_current[-1] = target_total_flux
             outer_clamp_value_to_use = target_total_flux
        else:
             outer_clamp_value_to_use = outer_max # Use the defined maximum
             flux_current[-1] = outer_clamp_value_to_use
             inner_flux_budget = target_total_flux - outer_clamp_value_to_use

             if inner_flux_budget < 0:
                  warnings.warn("Negative inner flux budget after clamping. Setting inner layers to zero.", UserWarning)
                  inner_flux_budget = 0

             inner_shape_sum = np.sum(interpolated_fluxes[:-1])
             if inner_shape_sum > 1e-6:
                  S_inner = inner_flux_budget / inner_shape_sum
                  print(f"Scaling inner layers (0 to {num_layers_target - 2}) by factor: {S_inner:.6f} to meet budget {inner_flux_budget:.2f}")
                  flux_current[:-1] = S_inner * interpolated_fluxes[:-1]
             elif inner_flux_budget > 1e-6:
                  warnings.warn("Inner layer shape sum is zero, but budget exists. Cannot scale inner layers.", UserWarning)
             # else: both budget and shape sum are zero

    else: # No clamping
        print("Applying global scaling factor to match target total flux.")
        global_scale_factor = target_total_flux / spline_shape_total_flux
        print(f"Global scaling factor: {global_scale_factor:.6f}")
        flux_current = interpolated_fluxes * global_scale_factor

    flux_current = np.maximum(flux_current, 0)
    initial_total_flux = np.sum(flux_current)
    print(f"\nFinal Flux Profile Total (Before Refinement): {initial_total_flux:.2f}")
    if not np.isclose(initial_total_flux, target_total_flux, rtol=1e-3):
         warnings.warn(f"Final total flux {initial_total_flux:.2f} differs slightly from target {target_total_flux:.2f}.", UserWarning)

    # --- Step 4: Refinement (Skipped) ---
    print("Skipping PPFD refinement step(s) as configured.")

    return flux_current


def main():
    print(f"--- Generating prediction for {NUM_LAYERS_TARGET} layers using SmoothBivariateSpline ---")
    print(f"--- Based on validated data (N=10 to N={max(known_configs.keys())}) ---")
    print(f"--- Applying proactive boost factor {EXTRAPOLATION_BOOST_FACTOR:.4f} if extrapolating ---")
    try:
        # Ensure refinement parameters are passed but inactive
        refinement_enabled = APPLY_PPFD_REFINEMENT
        num_iterations_to_run = NUM_REFINEMENT_ITERATIONS

        # --- Make sure N=18 validated data is in known_configs and actual_ppfds ---
        # (Double-check the values entered above)
        max_n_in_data = max(known_configs.keys())

        flux_assignments = predict_flux_assignments(
            NUM_LAYERS_TARGET,
            known_configs,
            target_ppfd=TARGET_PPFD,
            # Pass other parameters
            k=SPLINE_DEGREE,
            smoothing_mode=SMOOTHING_FACTOR_MODE,
            smoothing_mult=SMOOTHING_MULTIPLIER,
            clamp_outer=CLAMP_OUTER_LAYER,
            outer_max=OUTER_LAYER_MAX_FLUX, # Use the original clamp value
            apply_refinement=refinement_enabled,
            ppfd_feedback_list=None,
            num_refine_iter=num_iterations_to_run,
            extrapolation_boost=EXTRAPOLATION_BOOST_FACTOR # Pass boost factor
        )

        print(f"\nPredicted luminous flux assignments (N={NUM_LAYERS_TARGET} prediction with proactive boost):")
        for i, flux in enumerate(flux_assignments):
            layer_name = f"Layer {i}"
            if i == 0: layer_name = "Center COB (Layer 0)"
            elif i == NUM_LAYERS_TARGET - 1: layer_name = f"Outer Layer ({i})"
            print(f"    {flux:.4f},     # {layer_name}")

        print(f"\nTotal Predicted Flux: {np.sum(flux_assignments):.2f}")
        print(f"This flux profile is scaled to target an average PPFD of {TARGET_PPFD:.1f} µmol/m²/s.")

        # Report relevant settings used
        is_extrapolating_run = (NUM_LAYERS_TARGET > max_n_in_data)
        if is_extrapolating_run:
             print(f"Proactive extrapolation boost factor {EXTRAPOLATION_BOOST_FACTOR:.4f} was applied.")
        else:
             print("Prediction was within data range (interpolation), no boost applied.")

        if CLAMP_OUTER_LAYER:
             print(f"Outer layer clamp was applied (Max: {OUTER_LAYER_MAX_FLUX:.1f}).") # Report original clamp value
        else:
             print("Outer layer clamp was *not* applied.")

        print("PPFD refinement was disabled for this prediction.")

    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import numpy as np
from scipy.interpolate import SmoothBivariateSpline
import warnings

# ---------------------------
# Known successful configurations (Flux Params per Layer)
known_configs = {
    10: np.array([
        4036.684, 2018.342, 8074.0665, 5045.855, 6055.0856,
        12110.052, 15137.565, 2018.342, 20183.42, 24220.104
    ], dtype=np.float64),
    11: np.array([
        3343.1405, 4298.3235, 4298.9822, 2865.549, 11462.2685,
        17193.294, 7641.464, 2387.9575, 9074.2385, 22406.9105,
        22924.392
    ], dtype=np.float64),
    12: np.array([
        3393.159, 3393.159, 3393.8276, 2908.422, 9694.8,
        17450.532, 11633.688, 2423.685, 9210.003, 9694.74,
        19389.48, 23267.376
    ], dtype=np.float64),
    13: np.array([
        3431.5085, 3431.5085, 6373.4516, 2941.293, 6863.0557,
        12745.603, 12745.603, 13235.8185, 9314.0945, 3921.724,
        3921.724, 23530.344, 23530.344
    ], dtype=np.float64),
    14: np.array([
        3399.7962, 3399.7962, 5344.9613, 5344.2873, 6799.6487,
        7771.0271, 11656.5406, 15053.6553, 15053.6553, 3885.5135,
        3885.5135, 9713.7838, 19427.5677, 23313.0812
    ], dtype=np.float64),
    15: np.array([
        1932.1547, 2415.1934, 4344.4082, 8212.8061, 7728.6775,
        7728.6188, 6762.5415, 12559.0156, 16423.3299, 9660.7733,
        3864.3093, 5796.4640, 7728.6188, 23185.8563, 23185.8563
    ], dtype=np.float64),
    16: np.array([ # Added Successful N=16 Manual Data
        3801.6215, 3162.5110, 4132.8450, 7000.3495, 9000.9811,
        9000.4963, 7500.8053, 7000.4393, 13000.3296, 16500.3501,
        11000.8911, 4571.2174, 5127.4023, 9924.7413, 16685.5291,
        25000.0000
    ], dtype=np.float64),
    17: np.array([ # Added Successful N=17 Refined Data (V20)
        4332.1143, 3657.6542, 4017.4842, 5156.7357, 6708.1373,
        8319.1898, 9815.3845, 11114.2396, 12022.3055, 12091.5716,
        10714.1765, 8237.7581, 6442.0864, 6687.0190, 10137.5776,
        16513.6125, 25000.0000
    ], dtype=np.float64),
    # --- Added Successful N=18 Manually Tuned Data ---
    18: np.array([
        4372.8801, 3801.7895, 5000.3816, 6500.3919, 8000.5562,
        7691.6100, 8000.2891, 8000.7902, 10000.7478, 13000.3536,
        12700.1186, 9909.0286, 7743.6182, 6354.5384, 6986.5800,
        10529.9883, 16699.2440, 25000.0000
    ], dtype=np.float64),
    # --- Added Successful N=18 Manually Tuned Data ---
}
# Target Average PPFD
TARGET_PPFD = 1250.0

# Desired number of layers for prediction
# --- Target N=19 ---
NUM_LAYERS_TARGET = 20
# --- Target N=19 ---

# Spline parameters
SPLINE_DEGREE = 3
SMOOTHING_FACTOR_MODE = "multiplier"
SMOOTHING_MULTIPLIER = 1.5

# PPFD/Lumen Ratio Fitting parameters
RATIO_FIT_DEGREE = 2

# Clamping Parameters
CLAMP_OUTER_LAYER = True
OUTER_LAYER_MAX_FLUX = 25000.0

# --- V23 Parameters ---
# No empirical correction needed for first N=19 attempt
EMPIRICAL_PPFD_CORRECTION = {}

# --- Turn off refinement for initial N=19 prediction ---
APPLY_PPFD_REFINEMENT = False
NUM_REFINEMENT_ITERATIONS = 0
REFINEMENT_LEARNING_RATE = 0.40
MULTIPLIER_MIN = 0.7
MULTIPLIER_MAX = 1.3
# --- V23 Parameters ---

# ---------------------------

# --- Functions prepare_spline_data, fit_and_predict_ratio, apply_inner_refinement_step, apply_global_refinement_step ---
# (No changes needed in these functions)
def prepare_spline_data(known_configs):
    # ... (same code) ...
    n_list = []
    x_norm_list = []
    flux_list = []
    min_n, max_n = float('inf'), float('-inf')

    for n, fluxes in known_configs.items():
        if n < 2: continue
        num_layers_in_config = len(fluxes)
        if num_layers_in_config != n:
             warnings.warn(f"Mismatch for N={n}. Expected {n} fluxes, got {num_layers_in_config}. Using actual count {num_layers_in_config}.", UserWarning)
             n_actual = num_layers_in_config
             if n_actual < 2: continue
        else:
             n_actual = n

        min_n = min(min_n, n_actual)
        max_n = max(max_n, n_actual)
        norm_positions = np.linspace(0, 1, n_actual)
        for i, flux in enumerate(fluxes):
            n_list.append(n_actual)
            x_norm_list.append(norm_positions[i])
            flux_list.append(flux)

    if not n_list:
        raise ValueError("No valid configuration data found.")

    num_data_points = len(flux_list)
    print(f"Spline fitting using data from N={min_n} to N={max_n}. Total points: {num_data_points}") # Now includes N=18
    return np.array(n_list), np.array(x_norm_list), np.array(flux_list), min_n, max_n, num_data_points

def fit_and_predict_ratio(n_target, known_configs, target_ppfd=TARGET_PPFD, poly_degree=RATIO_FIT_DEGREE):
    # ... (same code, now uses N=16/17/18 actual PPFD) ...
    n_values_sorted = sorted(known_configs.keys())
    total_fluxes = [np.sum(known_configs[n]) for n in n_values_sorted]
    n_array = np.array(n_values_sorted)

    ratios = []
    valid_n = []
    actual_ppfds = {
        16: 1248.63,
        17: 1246.87,
        18: 1247.32, # Added N=18 actual PPFD
    }

    print("\nCalculating PPFD/Lumen ratio and fitting trend vs N:")
    for i, n in enumerate(n_values_sorted):
        if total_fluxes[i] > 1e-6:
            ppfd_to_use = actual_ppfds.get(n, target_ppfd)
            ratio = ppfd_to_use / total_fluxes[i]
            ratios.append(ratio)
            valid_n.append(n)
            print(f"  N={n}: Total Flux={total_fluxes[i]:.1f}, PPFD={ppfd_to_use:.2f}, Ratio={ratio:.6e}")
        else:
            print(f"  N={n}: Total Flux is zero, skipping.")

    if not ratios:
        raise ValueError("Could not calculate PPFD/Lumen ratio from any configuration.")

    if len(valid_n) <= poly_degree:
        original_degree = poly_degree
        poly_degree = len(valid_n) - 1
        warnings.warn(f"Not enough valid data points ({len(valid_n)}) to fit ratio trend with degree {original_degree}. Reducing degree to {poly_degree}.", UserWarning)
        if poly_degree < 0:
             warnings.warn("Cannot fit ratio trend (0 points). Using mean ratio.", UserWarning)
             return np.mean(ratios) if ratios else 0

    valid_n_array = np.array(valid_n)
    ratios_array = np.array(ratios)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        coeffs = np.polyfit(valid_n_array, ratios_array, poly_degree)

    poly_func = np.poly1d(coeffs)
    predicted_ratio = poly_func(n_target)

    if predicted_ratio <= 1e-9:
         warnings.warn(f"Ratio trend fit predicted non-positive ratio ({predicted_ratio:.4e}) for N={n_target}. Using ratio from nearest known N instead.", UserWarning)
         try:
             nearest_n_idx = np.abs(valid_n_array - n_target).argmin()
             predicted_ratio = ratios_array[nearest_n_idx]
             print(f"Using ratio from nearest N ({valid_n_array[nearest_n_idx]}): {predicted_ratio:.6e}")
         except IndexError:
              print("Error finding nearest N ratio, returning 0.")
              predicted_ratio = 0

    print(f"\nFitted Ratio Trend (Degree {poly_degree}): Ratio = {poly_func}")
    print(f"Predicted PPFD/Lumen Ratio for N={n_target}: {predicted_ratio:.6e}")
    return predicted_ratio

def apply_inner_refinement_step(flux_input, ppfd_feedback, target_ppfd, target_total_flux, outer_clamp_value, iteration, learn_rate=REFINEMENT_LEARNING_RATE, mult_min=MULTIPLIER_MIN, mult_max=MULTIPLIER_MAX):
    # ... (same code) ...
    num_layers = len(flux_input)
    if ppfd_feedback is None or len(ppfd_feedback) != num_layers:
        warnings.warn(f"PPFD feedback invalid for iteration {iteration}. Skipping refinement step.", UserWarning)
        return flux_input

    print(f"\n--- Applying INNER PPFD Refinement Iteration {iteration} ---")
    print(f"Using Learning Rate: {learn_rate:.2f}")
    errors = target_ppfd - ppfd_feedback
    relative_errors = np.divide(errors, target_ppfd, out=np.zeros_like(errors), where=abs(target_ppfd)>1e-9)
    multipliers = 1.0 + learn_rate * relative_errors
    multipliers = np.clip(multipliers, mult_min, mult_max)

    print(f"Refinement Multipliers (Iter {iteration}, min={mult_min:.2f}, max={mult_max:.2f}):")
    with np.printoptions(precision=4, suppress=True):
         print(multipliers)

    flux_inner_refined = flux_input[:-1] * multipliers[:-1]
    flux_inner_refined = np.maximum(flux_inner_refined, 0)

    inner_flux_budget = target_total_flux - outer_clamp_value
    if inner_flux_budget < 0:
         warnings.warn(f"Target total flux ({target_total_flux:.2f}) is less than outer clamp ({outer_clamp_value:.2f}) in iter {iteration}. Setting inner flux to zero.", UserWarning)
         inner_flux_budget = 0
         flux_inner_refined[:] = 0

    refined_inner_total = np.sum(flux_inner_refined)
    print(f"Refined Inner Flux Total (Iter {iteration}, before rescale): {refined_inner_total:.2f}")

    if refined_inner_total <= 1e-6:
         warnings.warn(f"Refined inner flux total near zero in iteration {iteration}. Check multipliers/inputs.", UserWarning)
         rescale_inner_factor = 0
    else:
         rescale_inner_factor = inner_flux_budget / refined_inner_total

    print(f"Inner rescale factor (Iter {iteration}): {rescale_inner_factor:.6f}")
    final_fluxes_inner = flux_inner_refined * rescale_inner_factor
    final_fluxes_inner = np.maximum(final_fluxes_inner, 0)

    final_fluxes = np.zeros_like(flux_input)
    final_fluxes[:-1] = final_fluxes_inner
    final_fluxes[-1] = outer_clamp_value

    actual_total_flux_iter = np.sum(final_fluxes)
    if not np.isclose(actual_total_flux_iter, target_total_flux, rtol=1e-3):
         warnings.warn(f"Total flux ({actual_total_flux_iter:.2f}) after inner refinement iter {iteration} differs slightly from target ({target_total_flux:.2f}).", UserWarning)

    return final_fluxes

def apply_global_refinement_step(flux_input, ppfd_feedback, target_ppfd, target_total_flux, iteration, learn_rate=REFINEMENT_LEARNING_RATE, mult_min=MULTIPLIER_MIN, mult_max=MULTIPLIER_MAX):
    # ... (same code) ...
    if ppfd_feedback is None or len(ppfd_feedback) != len(flux_input):
        warnings.warn(f"PPFD feedback invalid for iteration {iteration}. Skipping refinement step.", UserWarning)
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
    flux_refined = np.maximum(flux_refined, 0)

    refined_total_flux = np.sum(flux_refined)
    print(f"Refined Flux Total (Iter {iteration}, before final rescale): {refined_total_flux:.2f}")

    if refined_total_flux <= 1e-6:
         warnings.warn(f"Refined flux total near zero in iteration {iteration}. Returning unscaled.", UserWarning)
         return flux_refined
    else:
         final_rescale_factor = target_total_flux / refined_total_flux
         print(f"Final rescale factor (Iter {iteration}): {final_rescale_factor:.6f}")
         final_fluxes_iter = flux_refined * final_rescale_factor

    final_fluxes_iter = np.maximum(final_fluxes_iter, 0)

    actual_total_flux_iter = np.sum(final_fluxes_iter)
    if not np.isclose(actual_total_flux_iter, target_total_flux, rtol=1e-3):
         warnings.warn(f"Total flux ({actual_total_flux_iter:.2f}) after global refinement iter {iteration} differs significantly from target ({target_total_flux:.2f}).", UserWarning)

    return final_fluxes_iter

# Renamed function for clarity
def generate_flux_assignments_v23(num_layers_target, known_configs, target_ppfd=TARGET_PPFD, k=SPLINE_DEGREE, smoothing_mode=SMOOTHING_FACTOR_MODE, smoothing_mult=SMOOTHING_MULTIPLIER, clamp_outer=CLAMP_OUTER_LAYER, outer_max=OUTER_LAYER_MAX_FLUX, apply_refinement=APPLY_PPFD_REFINEMENT, ppfd_feedback_list=None, num_refine_iter=NUM_REFINEMENT_ITERATIONS, learn_rate=REFINEMENT_LEARNING_RATE, mult_min=MULTIPLIER_MIN, mult_max=MULTIPLIER_MAX, ppfd_correction_map=EMPIRICAL_PPFD_CORRECTION):
    """
    V23: Predicts N=19 using updated dataset (incl. N=18). No refinement initially.
    """
    if num_layers_target < 2:
        raise ValueError("Number of layers must be at least 2.")

    # --- Steps 1 & 2: Get Spline Shape and Target Total Flux ---
    # ... (logic as in V22) ...
    n_coords, x_norm_coords, flux_values, min_n_data, max_n_data, num_data_points = prepare_spline_data(known_configs)
    s_value = None
    if smoothing_mode == "num_points":
        s_value = float(num_data_points)
        print(f"Using explicit spline smoothing factor s = {s_value:.1f}")
    elif smoothing_mode == "multiplier":
        s_value = float(num_data_points) * smoothing_mult
        print(f"Using explicit spline smoothing factor s = {s_value:.1f} ({smoothing_mult} * num_points)")
    elif isinstance(smoothing_mode, (int, float)):
        s_value = float(smoothing_mode)
        print(f"Using explicit spline smoothing factor s = {s_value:.1f}")
    else:
        print("Using default spline smoothing factor s (estimated by FITPACK)")

    unique_n = len(np.unique(n_coords))
    unique_x = len(np.unique(x_norm_coords))
    if unique_n <= k or unique_x <= k:
         k = min(unique_n - 1, unique_x - 1, k)
         k = max(1, k)
         warnings.warn(f"Insufficient unique coordinates for spline degree {SPLINE_DEGREE}. Reducing degree to {k}.", UserWarning)

    try:
        with warnings.catch_warnings():
             warnings.filterwarnings("ignore", message="The required storage space", category=UserWarning)
             warnings.filterwarnings("ignore", message="The number of knots required", category=UserWarning)
             warnings.filterwarnings("ignore", message="Warning: s=", category=UserWarning)
             warnings.filterwarnings("ignore", message="Warning: ier=.*", category=UserWarning)
             spline = SmoothBivariateSpline(n_coords, x_norm_coords, flux_values, kx=k, ky=k, s=s_value)
    except Exception as e:
         print(f"\nError fitting spline: {e}")
         raise

    x_target_norm = np.linspace(0, 1, num_layers_target)
    n_target_array = np.full_like(x_target_norm, num_layers_target)
    interpolated_fluxes = spline(n_target_array, x_target_norm, grid=False)

    if num_layers_target < min_n_data or num_layers_target > max_n_data:
        warnings.warn(f"Extrapolating flux profile for N={num_layers_target}. Results may be less reliable.", UserWarning)

    interpolated_fluxes = np.maximum(interpolated_fluxes, 0)
    spline_shape_total_flux = np.sum(interpolated_fluxes)
    print(f"\nSpline evaluation for N={num_layers_target} yielded raw profile shape (Total Flux: {spline_shape_total_flux:.2f})")

    if spline_shape_total_flux <= 1e-6:
         warnings.warn("Spline evaluation resulted in near-zero total flux. Cannot scale reliably.", UserWarning)
         return interpolated_fluxes

    predicted_ppfd_per_lumen = fit_and_predict_ratio(num_layers_target, known_configs, target_ppfd)
    if predicted_ppfd_per_lumen <= 1e-9:
         warnings.warn("Predicted PPFD/Lumen ratio is non-positive. Cannot scale.", UserWarning)
         return interpolated_fluxes

    target_total_flux_initial = target_ppfd / predicted_ppfd_per_lumen
    correction_factor = ppfd_correction_map.get(num_layers_target, 1.0)
    target_total_flux = target_total_flux_initial * correction_factor
    if abs(correction_factor - 1.0) > 1e-4:
         print(f"Applying empirical PPFD correction factor for N={num_layers_target}: {correction_factor:.4f}")
    print(f"Final Target Total Flux required for {target_ppfd:.1f} PPFD: {target_total_flux:.2f} lumens")

    # --- Step 3: Apply Initial Scaling / Clamping ---
    flux_current = np.zeros_like(interpolated_fluxes)
    outer_clamp_value_to_use = 0.0

    if clamp_outer:
        # ... (clamping logic) ...
        print(f"Applying outer layer clamp: Layer {num_layers_target - 1} = {outer_max:.1f} lumens")
        if outer_max >= target_total_flux:
             warnings.warn("Outer layer clamp value exceeds target total flux. Setting inner layers to near zero.", UserWarning)
             flux_current[-1] = target_total_flux
             outer_clamp_value_to_use = target_total_flux
        else:
             flux_current[-1] = outer_max
             outer_clamp_value_to_use = outer_max
             inner_flux_budget = target_total_flux - outer_clamp_value_to_use
             inner_shape_sum = np.sum(interpolated_fluxes[:-1])
             if inner_shape_sum <= 1e-6:
                  warnings.warn("Sum of inner layer shapes from spline is near zero. Cannot scale inner layers.", UserWarning)
             else:
                  S_inner = inner_flux_budget / inner_shape_sum
                  print(f"Scaling inner layers (0 to {num_layers_target - 2}) by factor: {S_inner:.6f}")
                  flux_current[:-1] = S_inner * interpolated_fluxes[:-1]
    else:
        # ... (global scaling logic) ...
        print("Applying global scaling factor to match target total flux.")
        global_scale_factor = target_total_flux / spline_shape_total_flux
        print(f"Global scaling factor: {global_scale_factor:.6f}")
        flux_current = interpolated_fluxes * global_scale_factor
        outer_clamp_value_to_use = flux_current[-1]

    flux_current = np.maximum(flux_current, 0)
    print(f"Base Flux Profile Total (Before Refinement): {np.sum(flux_current):.2f}")

    # --- Step 4: Apply Refinement (Skipped) ---
    if apply_refinement: # Checks if refinement is intended AT ALL
         if ppfd_feedback_list is None or len(ppfd_feedback_list) < num_refine_iter or num_refine_iter == 0:
             print("Skipping PPFD refinement step(s) due to config or missing data.")
             return flux_current
         else:
             # This part shouldn't be reached if num_refine_iter is 0
             warnings.warn("Refinement logic error: Should not run loop if num_refine_iter is 0.")
             return flux_current # Fallback
    else:
         print("Skipping PPFD refinement step(s) as configured.")
         return flux_current


def main():
    print(f"--- Generating prediction for {NUM_LAYERS_TARGET} layers using SmoothBivariateSpline (V23 - N=19 Base Prediction) ---") # Title V23
    try:
        # Refinement is off, no feedback needed
        refinement_enabled = APPLY_PPFD_REFINEMENT # Should be False
        num_iterations_to_run = NUM_REFINEMENT_ITERATIONS # Should be 0

        flux_assignments = generate_flux_assignments_v23( # Changed function name
            NUM_LAYERS_TARGET,
            known_configs,
            target_ppfd=TARGET_PPFD,
            # Pass refinement params even though disabled
            ppfd_feedback_list=None,
            num_refine_iter=num_iterations_to_run,
            apply_refinement=refinement_enabled,
            learn_rate=REFINEMENT_LEARNING_RATE,
            clamp_outer=CLAMP_OUTER_LAYER,
            outer_max=OUTER_LAYER_MAX_FLUX
        )

        print("\nPredicted luminous flux assignments (V23 - Base N=19):")
        for i, flux in enumerate(flux_assignments):
            layer_name = "Center COB" if i == 0 else f"Layer {i}"
            print(f"    {flux:.4f},     # {layer_name}")

        print(f"\nTotal Predicted Flux: {np.sum(flux_assignments):.2f}")
        print(f"This flux profile is scaled to target an average PPFD of {TARGET_PPFD:.1f} µmol/m²/s.")
        current_correction = EMPIRICAL_PPFD_CORRECTION.get(NUM_LAYERS_TARGET, 1.0)
        if abs(current_correction - 1.0) > 1e-4 :
             print(f"Empirical PPFD correction factor {current_correction:.4f} applied.")
        if CLAMP_OUTER_LAYER:
             print(f"Outer layer clamp applied.")
        if refinement_enabled and num_iterations_to_run > 0:
             # This should not print
             print(f"{num_iterations_to_run} PPFD refinement iteration(s) applied.")
        else:
             print("PPFD refinement was disabled for this prediction.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
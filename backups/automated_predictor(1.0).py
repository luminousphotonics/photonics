#!/usr/bin/env python3
import numpy as np
from scipy.interpolate import SmoothBivariateSpline
import warnings
import time
import math
from numba import njit, prange
from functools import lru_cache
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import csv # Keep for potential final output if needed
from collections import deque
from scipy.optimize import curve_fit

# ==============================================================================
# Configuration & Targets
# ==============================================================================

# --- Prediction Targets ---
NUM_LAYERS_TARGET = 18      # The N we want to optimize for
TARGET_PPFD = 1250.0        # Target average PPFD (µmol/m²/s)
TARGET_DOU = 94.0           # Target Distribution Uniformity (%)
PPFD_TOLERANCE = 0.01       # Allowed relative tolerance for average PPFD (e.g., 0.01 = +/- 1%)
MAX_ITERATIONS = 10         # Maximum number of refinement iterations

# --- Simulation Geometry (Dynamic Sizing) ---
BASE_N = 10                 # Number of layers corresponding to BASE_W/L
BASE_W = 6.096            # 36 ft in meters (Assume this was the intended size for N=10)
BASE_L = 6.096            # 36 ft in meters
SIZE_INCREMENT_PER_N = 0.6096 # 2 ft in meters (Increment per side for each layer > BASE_N)
# SIM_W, SIM_L will be calculated dynamically based on NUM_LAYERS_TARGET
SIM_H = 0.9144              # 3 ft in meters (Ceiling height)

# --- Strip LED Configuration ---
ADD_STRIPS = True                  # Master toggle for adding LED strips
NUM_MINI_COBS_PER_SEGMENT = 5      # Number of mini-COBs between each pair of main COBs on a strip layer
INITIAL_STRIP_TO_COB_RATIO = 0.40  # Initial guess: Strip layer flux = X * COB layer flux

# --- Predictive Model Parameters ---
# !!! IMPORTANT !!!
# The numerical flux data below is from simulations WITH THE REFLECTION BUG.
# It MUST BE REPLACED with new data generated AFTER the bug fix.
# Placeholder strip data (zeros) is added.
# The structure is now N: (cob_flux_array, strip_flux_array)
known_configs = {
    # N: (cob_fluxes, strip_fluxes - PLACEHOLDER ZEROS)
    10: (np.array([4036.684, 2018.342, 8074.0665, 5045.855, 6055.0856, 12110.052, 15137.565, 2018.342, 20183.42, 24220.104], dtype=np.float64),
         np.zeros(10, dtype=np.float64)), # Placeholder
    11: (np.array([3343.1405, 4298.3235, 4298.9822, 2865.549, 11462.2685, 17193.294, 7641.464, 2387.9575, 9074.2385, 22406.9105, 22924.392], dtype=np.float64),
         np.zeros(11, dtype=np.float64)), # Placeholder
    12: (np.array([3393.159, 3393.159, 3393.8276, 2908.422, 9694.8, 17450.532, 11633.688, 2423.685, 9210.003, 9694.74, 19389.48, 23267.376], dtype=np.float64),
         np.zeros(12, dtype=np.float64)), # Placeholder
    13: (np.array([3431.5085, 3431.5085, 6373.4516, 2941.293, 6863.0557, 12745.603, 12745.603, 13235.8185, 9314.0945, 3921.724, 3921.724, 23530.344, 23530.344], dtype=np.float64),
         np.zeros(13, dtype=np.float64)), # Placeholder
    14: (np.array([3399.7962, 3399.7962, 5344.9613, 5344.2873, 6799.6487, 7771.0271, 11656.5406, 15053.6553, 15053.6553, 3885.5135, 3885.5135, 9713.7838, 19427.5677, 23313.0812], dtype=np.float64),
         np.zeros(14, dtype=np.float64)), # Placeholder
    15: (np.array([1932.1547, 2415.1934, 4344.4082, 8212.8061, 7728.6775, 7728.6188, 6762.5415, 12559.0156, 16423.3299, 9660.7733, 3864.3093, 5796.4640, 7728.6188, 23185.8563, 23185.8563], dtype=np.float64),
         np.zeros(15, dtype=np.float64)), # Placeholder
    16: (np.array([3801.6215, 3162.5110, 4132.8450, 7000.3495, 9000.9811, 9000.4963, 7500.8053, 7000.4393, 13000.3296, 16500.3501, 11000.8911, 4571.2174, 5127.4023, 9924.7413, 16685.5291, 25000.0000], dtype=np.float64),
         np.zeros(16, dtype=np.float64)), # Placeholder
    17: (np.array([4332.1143, 3657.6542, 4017.4842, 5156.7357, 6708.1373, 8319.1898, 9815.3845, 11114.2396, 12022.3055, 12091.5716, 10714.1765, 8237.7581, 6442.0864, 6687.0190, 10137.5776, 16513.6125, 25000.0000], dtype=np.float64),
         np.zeros(17, dtype=np.float64)), # Placeholder
    18: (np.array([4372.8801, 3801.7895, 5000.3816, 6500.3919, 8000.5562, 7691.6100, 8000.2891, 8000.7902, 10000.7478, 13000.3536, 12700.1186, 9909.0286, 7743.6182, 6354.5384, 6986.5800, 10529.9883, 16699.2440, 25000.0000], dtype=np.float64),
         np.zeros(18, dtype=np.float64)), # Placeholder
}
SPLINE_DEGREE = 3
SMOOTHING_FACTOR_MODE = "multiplier"
SMOOTHING_MULTIPLIER = 1.5
RATIO_FIT_DEGREE = 2 # Now refers to fit of total flux ratio
CLAMP_OUTER_LAYER = True # Clamps the OUTERMOST MAIN COB layer
OUTER_LAYER_MAX_FLUX = 25000.0
EMPIRICAL_PPFD_CORRECTION = {} # This will be updated dynamically

# --- Refinement Parameters ---
REFINEMENT_LEARNING_RATE = 0.30
MULTIPLIER_MIN = 0.85
MULTIPLIER_MAX = 1.15

# --- Simulation Parameters ---
REFL_WALL = 0.8
REFL_CEIL = 0.8
REFL_FLOOR = 0.1 # *** Bug was likely related to how this was used ***
LUMINOUS_EFFICACY = 182.0  # lumens/W
SPD_FILE = "spd_data.csv"
MAX_RADIOSITY_BOUNCES = 10 # *** Increased needed after fixing reflections? Keep 10 for now. ***
RADIOSITY_CONVERGENCE_THRESHOLD = 1e-3 # *** May need adjustment after fix ***
WALL_SUBDIVS_X = 10
WALL_SUBDIVS_Y = 5
CEIL_SUBDIVS_X = 10
CEIL_SUBDIVS_Y = 10
FLOOR_GRID_RES = 0.08  # m
MC_SAMPLES = 16 # Monte Carlo samples for reflection calc

COB_ANGLE_DATA = np.array([
    [0, 1.00], [10, 0.98], [20, 0.95], [30, 0.88], [40, 0.78],
    [50, 0.65], [60, 0.50], [70, 0.30], [80, 0.10], [90, 0.00],
], dtype=np.float64)
COB_angles_deg = COB_ANGLE_DATA[:, 0]
COB_shape = COB_ANGLE_DATA[:, 1]

# --- Optional Plotting ---
SHOW_FINAL_HEATMAP = True
ANNOTATION_STEP = 10

# ==============================================================================
# Simulation Core Functions (Minor changes for clarity/robustness)
# ==============================================================================

# Constants (place near other constants if preferred)
H_PLANCK = 6.626e-34  # J*s
C_LIGHT = 3.0e8      # m/s
N_AVOGADRO = 6.022e23 # mol^-1

def compute_conversion_factor(spd_file):
    """
    Calculates the conversion factor from total radiant Watts to PAR photon flux (µmol/s).
    Returns: Factor in µmol/J (or µmol/s per W).
    """
    try:
        # Try comma delimiter first, fallback to space
        try:
            spd = np.loadtxt(spd_file, delimiter=',', skiprows=1)
        except ValueError:
             spd = np.loadtxt(spd_file, delimiter=' ', skiprows=1) # Fallback to space
        except Exception as inner_e:
             raise FileNotFoundError(f"Could not load SPD file '{spd_file}' with comma or space delimiter. Error: {inner_e}")

        if spd.shape[1] < 2:
            raise ValueError(f"SPD file '{spd_file}' must have at least 2 columns (Wavelength, Intensity).")

        wl_nm = spd[:, 0]
        intensity = spd[:, 1] # Assume intensity is proportional to W/nm

    except FileNotFoundError as e:
        print(f"[Error] SPD file '{spd_file}' not found or failed to load: {e}. Using default 4.6 µmol/J.")
        return 4.6
    except Exception as e:
        print(f"[Error] Error processing SPD data from '{spd_file}': {e}. Using default 4.6 µmol/J.")
        return 4.6

    # --- Calculate Total Radiant Energy Integral (proportional) ---
    # Ensure wavelengths are increasing for reliable integration
    sort_indices = np.argsort(wl_nm)
    wl_nm_sorted = wl_nm[sort_indices]
    intensity_sorted = intensity[sort_indices]

    # Check for duplicate wavelengths, average intensity if found
    unique_wl, unique_indices = np.unique(wl_nm_sorted, return_index=True)
    if len(unique_wl) < len(wl_nm_sorted):
        warnings.warn(f"Duplicate wavelengths found in {spd_file}. Averaging intensities.", UserWarning)
        avg_intensity = np.zeros_like(unique_wl)
        for i, u_wl in enumerate(unique_wl):
            mask = (wl_nm_sorted == u_wl)
            avg_intensity[i] = np.mean(intensity_sorted[mask])
        wl_nm_processed = unique_wl
        intensity_processed = avg_intensity
    else:
        wl_nm_processed = wl_nm_sorted
        intensity_processed = intensity_sorted

    # Ensure sufficient data points for integration
    if len(wl_nm_processed) < 2:
         print(f"[Error] Not enough data points ({len(wl_nm_processed)}) in SPD file '{spd_file}' after processing for integration. Using default 4.6 µmol/J.")
         return 4.6

    # Total integral (proportional to total Watts)
    try:
        total_energy_integral = np.trapz(intensity_processed, wl_nm_processed)
    except Exception as e:
        print(f"[Error] Failed during total energy integration for {spd_file}: {e}. Using default 4.6 µmol/J.")
        return 4.6

    if total_energy_integral <= 1e-9:
        print(f"[Warning] Total energy integral for SPD '{spd_file}' is near zero. Check SPD data. Using default 4.6 µmol/J.")
        return 4.6 # Prevent division by zero

    # --- Calculate PAR Photon Flux Integral ---
    mask_par = (wl_nm_processed >= 400) & (wl_nm_processed <= 700)
    wl_par_nm = wl_nm_processed[mask_par]
    intensity_par = intensity_processed[mask_par]

    if len(wl_par_nm) < 2:
        print(f"[Warning] No data points within PAR range (400-700nm) found in {spd_file}. PAR conversion factor will be 0. Using fallback 4.6 µmol/J.")
        # Returning 0 would be physically correct if no PAR, but fallback might be safer for the predictor
        return 4.6

    # Energy per photon (Joules)
    wl_par_m = wl_par_nm * 1e-9
    energy_per_photon = H_PLANCK * C_LIGHT / wl_par_m # J/photon

    # Photon flux density (photons/s per nm per Watt_total - relative)
    # intensity is proportional to W/nm
    # energy_per_photon is J/photon
    # (W/nm) / (J/photon) = (J/s/nm) / (J/photon) = photons/s/nm
    photon_flux_density = intensity_par / energy_per_photon # Relative photons/s/nm

    # Integrate photon flux density over PAR range (relative photons/s)
    try:
        par_photon_flux_integral = np.trapz(photon_flux_density, wl_par_nm)
    except Exception as e:
        print(f"[Error] Failed during PAR photon flux integration for {spd_file}: {e}. Using default 4.6 µmol/J.")
        return 4.6


    # --- Calculate Conversion Factor (µmol/J) ---
    # (relative photons/s) / (relative J/s) = photons/J
    photons_per_joule = par_photon_flux_integral / total_energy_integral

    # Convert photons/J to µmol/J
    umol_per_joule = photons_per_joule / N_AVOGADRO * 1e6

    print(f"[INFO] SPD: Calculated conversion factor = {umol_per_joule:.5f} µmol/J for '{spd_file}'")

    # Sanity check
    if not (0.1 < umol_per_joule < 10.0):
         warnings.warn(f"Calculated conversion factor ({umol_per_joule:.3f} µmol/J) is outside the typical range (0.1-10.0). Check SPD file '{spd_file}'. Using calculated value anyway.", UserWarning)
    # Optional: Force fallback if extreme value?
    # if not (0.1 < umol_per_joule < 10.0):
    #     print("[Warning] Calculated factor out of range, using fallback 4.6 µmol/J.")
    #     return 4.6

    return umol_per_joule

# Ensure this global calculation uses the reverted function
CONVERSION_FACTOR = compute_conversion_factor(SPD_FILE)

def integrate_shape_for_flux(angles_deg, shape):
    # ... (same as before) ...
    rad_angles = np.radians(angles_deg)
    G = 0.0
    for i in range(len(rad_angles) - 1):
        th0, th1 = rad_angles[i], rad_angles[i+1]
        s0, s1 = shape[i], shape[i+1]
        s_mean = 0.5*(s0 + s1)
        dtheta = (th1 - th0)
        th_mid = 0.5*(th0 + th1)
        sin_mid = math.sin(th_mid)
        G_seg = s_mean * 2.0*math.pi * sin_mid * dtheta
        G += G_seg
    return G

SHAPE_INTEGRAL = integrate_shape_for_flux(COB_angles_deg, COB_shape)
if SHAPE_INTEGRAL <= 1e-6:
    warnings.warn("COB Shape Integral is near zero. Intensity calculation might fail.")
    SHAPE_INTEGRAL = 1.0 # Avoid division by zero

@njit
def luminous_intensity(angle_deg, total_lumens):
    # ... (same as before) ...
    if angle_deg <= COB_angles_deg[0]:
        rel = COB_shape[0]
    elif angle_deg >= COB_angles_deg[-1]:
        rel = COB_shape[-1]
    else:
        # Use np.interp equivalent for Numba
        rel = np.interp(angle_deg, COB_angles_deg, COB_shape)
    # Ensure SHAPE_INTEGRAL is not zero
    intensity = (total_lumens * rel) / SHAPE_INTEGRAL if SHAPE_INTEGRAL > 1e-9 else 0.0
    return intensity

def get_ordered_layer_coords(layer_index, W, L, H, num_total_layers, transform_params):
    """
    Generates the transformed coordinates for a specific MAIN COB layer's perimeter IN ORDER.
    Helper function for build_light_source_positions.
    Returns list of (px, py, pz) tuples for the main COBs of that layer.
    """
    if layer_index == 0: # Center COB
        center_x = transform_params['center_x']
        center_y = transform_params['center_y']
        pz = H * 0.98
        return [(center_x, center_y, pz)]
    if layer_index >= num_total_layers:
        return []

    i = layer_index
    ordered_abstract_coords = []

    # Generate abstract coordinates in clockwise order (example)
    # Top-Right to Top-Left (Quadrant 1 changing to 2)
    for x in range(i, 0, -1): ordered_abstract_coords.append((x, i - x))
    # Top-Left to Bottom-Left (Quadrant 2 changing to 3)
    for x in range(0, -i, -1): ordered_abstract_coords.append((x, i + x))
    # Bottom-Left to Bottom-Right (Quadrant 3 changing to 4)
    for x in range(-i, 0, 1): ordered_abstract_coords.append((x, -i - x))
    # Bottom-Right to Top-Right (Quadrant 4 changing to 1)
    for x in range(0, i + 1, 1): # Include the starting point (i, 0) again? No, let caller handle wrap. End at x=i-1.
         if x == i: break # Stop before repeating the start point
         ordered_abstract_coords.append((x, -i + x))
    # Add the starting point explicitly if needed by linspace later? Let's try without first.
    # The loop above generates 4*i points. Correct. Start point is (i,0) transformed.
    # Let's generate explicitly (i,0), (i-1, 1)...(1, i-1), (0,i), (-1, i-1)...(-i,0)...(0,-i)...(i-1, -1)
    ordered_abstract_coords_new = []
    # Q1 -> Q2
    for x in range(i, -i-1, -1): # x from i down to -i
        y = i - abs(x)
        if x <= 0: y = -y # Make y negative in Q3, Q4
        if abs(x) + abs(y) == i:
             ordered_abstract_coords_new.append((x,y))
             if x > -i and x <= 0: # Add the (-x, y) point for the other side Q3->Q4? No, diamond logic is simpler.
                pass # Diamond logic below handles it

    # Simpler way: generate all perimeter points, then sort? Too complex.
    # Let's stick to the segment generation logic from the visualization example.

    ordered_abstract_coords = []
    # Top-Right to Top-Left (Quadrant 1 changing to 2)
    for x in range(i, 0, -1): ordered_abstract_coords.append((float(x), float(i - x)))
    # Top-Left to Bottom-Left (Quadrant 2 changing to 3)
    for x in range(0, -i, -1): ordered_abstract_coords.append((float(x), float(i + x)))
    # Bottom-Left to Bottom-Right (Quadrant 3 changing to 4)
    for x in range(-i, 0, 1): ordered_abstract_coords.append((float(x), float(-i - x)))
    # Bottom-Right to Top-Right (Quadrant 4 changing to 1)
    for x in range(0, i, 1): ordered_abstract_coords.append((float(x), float(-i + x)))


    # Apply the transformation
    center_x = transform_params['center_x']
    center_y = transform_params['center_y']
    scale_x = transform_params['scale_x']
    scale_y = transform_params['scale_y']
    cos_t = transform_params['cos_t']
    sin_t = transform_params['sin_t']
    pz = H * 0.98

    ordered_transformed_coords = []
    for (ax, ay) in ordered_abstract_coords:
        rx = ax * cos_t - ay * sin_t
        ry = ax * sin_t + ay * cos_t
        px = center_x + rx * scale_x
        py = center_y + ry * scale_y
        ordered_transformed_coords.append((px, py, pz))

    return ordered_transformed_coords

def build_light_source_positions(W, L, H, num_total_layers, add_strips=True, num_mini_cobs=3):
    """
    Builds positions for main COBs and mini-COBs representing LED strips.

    Returns:
        np.array: Array of shape (num_sources, 5)
                  Columns: [px, py, pz, layer_index, is_strip_source (0 or 1)]
    """
    print(f"[Geo] Building light sources for N={num_total_layers} (Strips: {'Yes' if add_strips else 'No'})")
    n = num_total_layers - 1 # Max layer index
    light_sources = []

    # --- Calculate Transformation Parameters ---
    theta = math.radians(45)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    center_x, center_y = W / 2.0, L / 2.0
    scale_x = W / (n * math.sqrt(2)) if n > 0 else W / 2.0
    scale_y = L / (n * math.sqrt(2)) if n > 0 else L / 2.0
    pz = H * 0.98 # Consistent Z height

    transform_params = {
        'center_x': center_x, 'center_y': center_y,
        'scale_x': scale_x, 'scale_y': scale_y,
        'cos_t': cos_t, 'sin_t': sin_t,
        'H': H # Pass H for pz calculation consistency
    }

    # --- Generate Main COB Positions ---
    main_cob_coords_by_layer = {} # Store transformed coords per layer
    # Layer 0 (Center)
    light_sources.append([center_x, center_y, pz, 0, 0]) # [x, y, z, layer, is_strip]
    main_cob_coords_by_layer[0] = [(center_x, center_y, pz)]

    # Layers 1 to n
    for i in range(1, n + 1):
        ordered_layer_i_coords = get_ordered_layer_coords(i, W, L, H, num_total_layers, transform_params)
        main_cob_coords_by_layer[i] = ordered_layer_i_coords
        for (px, py, pz_coord) in ordered_layer_i_coords:
             light_sources.append([px, py, pz_coord, i, 0]) # Main COB

    num_main_cobs = len(light_sources)
    print(f"[Geo] Generated {num_main_cobs} main COB positions.")

    # --- Generate Strip Mini-COB Positions ---
    if add_strips and num_total_layers > 1 and num_mini_cobs > 0:
        num_strip_sources_added = 0
        for i in range(1, n + 1): # Strips connect main COBs of layers 1 to n
            main_cobs_layer_i = main_cob_coords_by_layer.get(i, [])
            num_points = len(main_cobs_layer_i)
            if num_points < 2: continue # Need at least 2 points to form a segment

            for k in range(num_points):
                p1 = np.array(main_cobs_layer_i[k])       # Start point of segment
                p2 = np.array(main_cobs_layer_i[(k + 1) % num_points]) # End point (wrap around)

                # Generate points *between* p1 and p2 using linspace
                # We want num_mini_cobs points, so segments = num_mini_cobs + 1
                # Linspace includes endpoints, so generate num_mini_cobs + 2 points and slice
                strip_points = np.linspace(p1, p2, num_mini_cobs + 2)

                # Add the intermediate points (exclude p1 and p2)
                for j in range(1, num_mini_cobs + 1):
                    px, py, pz_coord = strip_points[j]
                    light_sources.append([px, py, pz_coord, i, 1]) # Strip source, layer i
                    num_strip_sources_added += 1

        print(f"[Geo] Added {num_strip_sources_added} strip mini-COB positions.")

    final_sources = np.array(light_sources, dtype=np.float64)

    # Verification
    expected_cobs = 2*num_total_layers**2 - 2*num_total_layers + 1 if num_total_layers > 0 else 0
    if num_main_cobs != expected_cobs:
         warnings.warn(f"Unexpected number of MAIN COBs generated for N={num_total_layers}. Expected {expected_cobs}, got {num_main_cobs}.", UserWarning)
    expected_strips = 0
    if add_strips and num_mini_cobs > 0:
        for i in range(1, n+1):
            expected_strips += (4*i) * num_mini_cobs # 4*i segments per layer * num_mini_cobs per segment
    num_strip_sources_found = np.sum(final_sources[:, 4] == 1)
    if add_strips and num_strip_sources_found != expected_strips:
         warnings.warn(f"Unexpected number of STRIP sources generated for N={num_total_layers}. Expected {expected_strips}, got {num_strip_sources_found}.", UserWarning)

    print(f"[Geo] Total light sources generated: {len(final_sources)}")
    return final_sources


def pack_luminous_flux(cob_flux_params_per_layer, strip_flux_params_per_layer, light_source_positions):
    """Assigns flux values to each light source based on its type and layer."""
    led_intensities = []
    num_cob_layers = len(cob_flux_params_per_layer)
    num_strip_layers = len(strip_flux_params_per_layer) # Should match num_cob_layers

    # Ensure strip params has entry for layer 0, even if unused (value should be 0)
    if num_strip_layers < num_cob_layers:
        strip_flux_params_per_layer = np.pad(strip_flux_params_per_layer, (0, num_cob_layers - num_strip_layers), 'constant')
        warnings.warn(f"Padding strip_flux_params to match cob_flux_params length ({num_cob_layers}).", UserWarning)

    total_assigned_cob_flux = 0.0
    total_assigned_strip_flux = 0.0

    for src in light_source_positions:
        layer = int(src[3])
        is_strip = int(src[4])
        intensity = 0.0

        if 0 <= layer < num_cob_layers:
            if is_strip == 0: # Main COB
                intensity = cob_flux_params_per_layer[layer]
                total_assigned_cob_flux += intensity
            elif is_strip == 1: # Strip mini-COB
                if layer > 0 : # Strips defined for layers 1+
                     intensity = strip_flux_params_per_layer[layer]
                     total_assigned_strip_flux += intensity
                else:
                     # Should not happen if geometry is correct, but safety check
                     intensity = 0.0 # Layer 0 has no strips
            else:
                warnings.warn(f"Light source has invalid is_strip flag {is_strip}. Assigning 0 flux.", UserWarning)
                intensity = 0.0
        else:
            warnings.warn(f"Light source has invalid layer index {layer}. Assigning 0 flux.", UserWarning)
            intensity = 0.0

        # Distribute strip layer flux among mini-COBs of that layer
        # *** CORRECTION: The pack function should assign the PER-LAYER flux / num_sources_in_layer ***
        # This needs rethinking. The input flux arrays are PER LAYER. The packing needs to know
        # how many sources *of each type* are in that layer to divide the flux.

    # --- REVISED Flux Packing Logic ---
    led_intensities = np.zeros(len(light_source_positions), dtype=np.float64)
    counts = {} # (layer, is_strip) -> count

    for i, src in enumerate(light_source_positions):
        layer = int(src[3])
        is_strip = int(src[4])
        key = (layer, is_strip)
        counts[key] = counts.get(key, 0) + 1

    total_assigned_cob_flux = 0.0
    total_assigned_strip_flux = 0.0

    for i, src in enumerate(light_source_positions):
        layer = int(src[3])
        is_strip = int(src[4])
        key = (layer, is_strip)
        num_sources_in_group = counts.get(key, 1) # Avoid division by zero if count failed
        if num_sources_in_group == 0: num_sources_in_group = 1 # Safety

        intensity_per_source = 0.0
        if 0 <= layer < num_cob_layers:
            if is_strip == 0: # Main COB
                layer_flux = cob_flux_params_per_layer[layer]
                intensity_per_source = layer_flux / num_sources_in_group
                total_assigned_cob_flux += intensity_per_source
            elif is_strip == 1: # Strip mini-COB
                 if layer > 0 and layer < num_strip_layers:
                     layer_flux = strip_flux_params_per_layer[layer]
                     intensity_per_source = layer_flux / num_sources_in_group
                     total_assigned_strip_flux += intensity_per_source
                 # else intensity remains 0
            # else intensity remains 0
        # else intensity remains 0

        led_intensities[i] = intensity_per_source

    print(f"[Flux] Total packed COB flux: {total_assigned_cob_flux:.2f} (Target sum: {np.sum(cob_flux_params_per_layer):.2f})")
    print(f"[Flux] Total packed Strip flux: {total_assigned_strip_flux:.2f} (Target sum: {np.sum(strip_flux_params_per_layer[1:]):.2f})") # Sum strips from layer 1

    return led_intensities


# LRU Cache might need adjustment if W, L, H change, but geometry should be fixed per run_automated_prediction call
@lru_cache(maxsize=4)
def cached_build_floor_grid(W: float, L: float, grid_res: float):
    # ... (same as before) ...
    num_x = int(round(W / grid_res)) + 1
    num_y = int(round(L / grid_res)) + 1
    xs = np.linspace(0, W, num_x)
    ys = np.linspace(0, L, num_y)
    X, Y = np.meshgrid(xs, ys)
    return X, Y

def build_floor_grid(W, L):
    return cached_build_floor_grid(W, L, FLOOR_GRID_RES)

@lru_cache(maxsize=4)
def cached_build_patches(W: float, L: float, H: float,
                         wall_x: int, wall_y: int, ceil_x: int, ceil_y: int,
                         refl_f: float, refl_c: float, refl_w: float):
    # ... (same as before, check if bug originated here - unlikely based on description) ...
    patch_centers = []
    patch_areas = []
    patch_normals = []
    patch_refl = []

    # Floor (single patch) - Reflectivity REFL_FLOOR was likely the issue source
    patch_centers.append((W/2, L/2, 0.0))
    patch_areas.append(W * L)
    patch_normals.append((0.0, 0.0, 1.0)) # Normal points UP
    patch_refl.append(refl_f) # USE THE CORRECT REFLECTIVITY

    # Ceiling
    xs_ceiling = np.linspace(0, W, ceil_x + 1)
    ys_ceiling = np.linspace(0, L, ceil_y + 1)
    for i in range(ceil_x):
        for j in range(ceil_y):
            cx = (xs_ceiling[i] + xs_ceiling[i+1]) / 2
            cy = (ys_ceiling[j] + ys_ceiling[j+1]) / 2
            area = (xs_ceiling[i+1] - xs_ceiling[i]) * (ys_ceiling[j+1] - ys_ceiling[j])
            patch_centers.append((cx, cy, H )) # Ceiling at H
            patch_areas.append(area)
            patch_normals.append((0.0, 0.0, -1.0)) # Normal points DOWN
            patch_refl.append(refl_c)

    # Walls (4 walls)
    xs_wall = np.linspace(0, W, wall_x + 1)
    zs_wall = np.linspace(0, H, wall_y + 1)
    # Wall Y=0
    for i in range(wall_x):
        for j in range(wall_y):
            cx = (xs_wall[i] + xs_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (xs_wall[i+1]-xs_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((cx, 0.0, cz))
            patch_areas.append(area)
            patch_normals.append((0.0, 1.0, 0.0)) # Normal points INTO room (positive Y)
            patch_refl.append(refl_w)
    # Wall Y=L
    for i in range(wall_x):
        for j in range(wall_y):
            cx = (xs_wall[i] + xs_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (xs_wall[i+1]-xs_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((cx, L, cz))
            patch_areas.append(area)
            patch_normals.append((0.0, -1.0, 0.0)) # Normal points INTO room (negative Y)
            patch_refl.append(refl_w)

    ys_wall = np.linspace(0, L, wall_x + 1) # Use wall_x for consistency? Or allow separate subdivisions? Using wall_x
    # Wall X=0
    for i in range(wall_x): # Assuming same subdivision count for length-wise walls
        for j in range(wall_y):
            cy = (ys_wall[i] + ys_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (ys_wall[i+1]-ys_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((0.0, cy, cz))
            patch_areas.append(area)
            patch_normals.append((1.0, 0.0, 0.0)) # Normal points INTO room (positive X)
            patch_refl.append(refl_w)
    # Wall X=W
    for i in range(wall_x):
        for j in range(wall_y):
            cy = (ys_wall[i] + ys_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (ys_wall[i+1]-ys_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((W, cy, cz))
            patch_areas.append(area)
            patch_normals.append((-1.0, 0.0, 0.0)) # Normal points INTO room (negative X)
            patch_refl.append(refl_w)

    print(f"[INFO] Built {len(patch_centers)} patches for radiosity.")
    return (np.array(patch_centers, dtype=np.float64),
            np.array(patch_areas, dtype=np.float64),
            np.array(patch_normals, dtype=np.float64),
            np.array(patch_refl, dtype=np.float64))

def build_patches(W, L, H):
    # Pass the configured reflectivities correctly
    return cached_build_patches(W, L, H,
                                WALL_SUBDIVS_X, WALL_SUBDIVS_Y, CEIL_SUBDIVS_X, CEIL_SUBDIVS_Y,
                                REFL_FLOOR, REFL_CEIL, REFL_WALL)

def prepare_geometry(W, L, H, num_total_layers, add_strips, num_mini_cobs):
    """Prepares all static geometry needed for the simulation."""
    light_source_positions = build_light_source_positions(W, L, H, num_total_layers, add_strips, num_mini_cobs)
    X, Y = build_floor_grid(W, L)
    patches = build_patches(W, L, H) # Uses dynamically calculated W, L
    return (light_source_positions, X, Y, patches)

# --- Numba JIT Compiled Calculation Functions ---
# compute_direct_floor, compute_patch_direct, iterative_radiosity_loop
# These functions take light source positions and lumens per source. They don't
# inherently care if it's a main COB or a strip mini-COB, as long as the
# position and lumen value are provided. No changes needed here.
# *** However, the BUG was likely in how reflections were calculated or applied.
# Review iterative_radiosity_loop and compute_row_reflection_mc carefully. ***

@njit(parallel=True)
def compute_direct_floor(light_source_positions, lumens_per_source, X, Y):
    # Takes generic light sources - NO CHANGE NEEDED
    # ... (same as before) ...
    min_dist2 = (FLOOR_GRID_RES / 2.0)**2
    rows, cols = X.shape
    out = np.zeros((rows, cols), dtype=np.float64) # Explicit dtype

    for r in prange(rows): # Use prange for parallel loop
        for c in range(cols):
            fx, fy = X[r, c], Y[r, c]
            val = 0.0
            for k in range(light_source_positions.shape[0]): # Use combined sources
                lx, ly, lz = light_source_positions[k, 0], light_source_positions[k, 1], light_source_positions[k, 2]
                lumens_k = lumens_per_source[k] # Use packed lumens
                if lumens_k <= 0: continue # Skip zero-flux sources

                dx, dy, dz = fx - lx, fy - ly, 0.0 - lz # Floor point Z=0
                d2 = dx*dx + dy*dy + dz*dz
                if d2 < min_dist2: d2 = min_dist2 # Avoid singularity
                dist = math.sqrt(d2)

                # Cosine of angle from LED axis (downward Z) to floor point
                cos_th_led = -dz / dist # dz is negative, so cos_th is positive
                if cos_th_led <= 1e-6: continue # Point is behind or exactly at 90 deg from LED normal

                clipped_cos_th = max(-1.0, min(cos_th_led, 1.0)) # Manual clip for scalar
                angle_deg = math.degrees(math.acos(clipped_cos_th)) # Angle from vertical downward axis
                I_theta = luminous_intensity(angle_deg, lumens_k)

                # Cosine of angle for floor normal (upward Z) - same as cos_th_led
                cos_in_floor = cos_th_led

                E_local = (I_theta / d2) * cos_in_floor # Illuminance = Intensity / dist^2 * cos(incidence_angle)
                val += E_local
            out[r, c] = val
    return out

@njit
def compute_patch_direct(light_source_positions, lumens_per_source, patch_centers, patch_normals, patch_areas):
    # Takes generic light sources - NO CHANGE NEEDED
    # ... (same as before) ...
    min_dist2 = (FLOOR_GRID_RES / 2.0)**2
    Np = patch_centers.shape[0]
    out = np.zeros(Np, dtype=np.float64)

    for ip in range(Np):
        pc = patch_centers[ip]
        n = patch_normals[ip]
        norm_n = 1.0 # Assume unit normal

        accum = 0.0
        for k in range(light_source_positions.shape[0]): # Loop through combined sources
            lx, ly, lz = light_source_positions[k, 0], light_source_positions[k, 1], light_source_positions[k, 2]
            lumens_k = lumens_per_source[k] # Use packed lumens
            if lumens_k <= 0: continue

            # Vector L -> P
            dx, dy, dz = pc[0] - lx, pc[1] - ly, pc[2] - lz
            d2 = dx*dx + dy*dy + dz*dz
            if d2 < min_dist2: d2 = min_dist2
            dist = math.sqrt(d2)

            # Angle relative to LED Z-axis (downward)
            cos_th_led = -dz / dist # Positive if patch is below LED Z-plane

            if cos_th_led <= 1e-6: continue # Light doesn't go towards patch

            clipped_cos_th = max(-1.0, min(cos_th_led, 1.0))
            angle_deg = math.degrees(math.acos(clipped_cos_th))
            I_theta = luminous_intensity(angle_deg, lumens_k)

            dot_patch_correct = (-dx)*n[0] + (-dy)*n[1] + (-dz)*n[2]
            cos_in_patch = dot_patch_correct / dist

            if cos_in_patch <= 1e-6: continue

            E_local = (I_theta / d2) * cos_in_patch
            accum += E_local

        out[ip] = accum

    return out

# --- Check Radiosity and Reflection Calculation ---
# The previous bug description ("0 intensity from reflections") strongly suggests
# the issue might be in iterative_radiosity_loop (how B or E_indirect is calculated/used)
# or compute_row_reflection_mc (how flux is transferred from patches to floor).

@njit
def iterative_radiosity_loop(patch_centers, patch_normals, patch_direct_E, patch_areas, patch_refl,
                             max_bounces, convergence_threshold):
    # Let's double-check the core radiosity equation implementation
    # B_i = rho_i * E_total_i = rho_i * (E_direct_i + E_indirect_i)
    # E_indirect_i = sum_{j!=i} ( B_j * FormFactor_{j->i} )
    # FormFactor_{j->i} approx (cos_j * cos_i * Area_j) / (pi * dist_ij^2)  <- This is the term used
    Np = patch_direct_E.shape[0]
    patch_B = patch_refl * patch_direct_E # B_0 = rho * E_direct
    patch_B_new = np.zeros_like(patch_B)
    patch_E_indirect_this_bounce = np.zeros_like(patch_direct_E) # Accumulator for E_indirect in *one* bounce
    epsilon = 1e-9

    for bounce in range(max_bounces):
        patch_E_indirect_this_bounce.fill(0.0) # Reset for this bounce calculation

        for j in range(Np): # Source patch j
            # *** Check 1: Is B from the PREVIOUS bounce used correctly? Yes, patch_B holds B(n-1) ***
            # *** Check 2: Is reflectance check okay? Yes. ***
            if patch_refl[j] <= 1e-6 or patch_B[j] <= 1e-6: continue

            pj = patch_centers[j]
            nj = patch_normals[j]
            area_j = patch_areas[j] # Needed for form factor

            for i in range(Np): # Receiver patch i
                if i == j: continue

                pi = patch_centers[i]
                ni = patch_normals[i]

                vij = pi - pj
                dist2 = vij[0]*vij[0] + vij[1]*vij[1] + vij[2]*vij[2]
                if dist2 < 1e-15: continue
                dist = math.sqrt(dist2)

                cos_j = np.dot(nj, vij) / dist
                cos_i = np.dot(ni, -vij) / dist

                if cos_j <= 1e-6 or cos_i <= 1e-6: continue

                # *** Check 3: Is the indirect illuminance calculation correct? ***
                # E_indirect_on_i_from_j = B_j * FormFactor_{j->i}
                # E_indirect_on_i_from_j = patch_B[j] * (cos_j * cos_i * area_j) / (math.pi * dist2)
                # This looks dimensionally correct (B is lm/m^2 or W/m^2, FF is dimensionless).
                E_indirect_on_i_from_j = patch_B[j] * cos_j * cos_i / (math.pi * dist2) * area_j
                E_indirect_on_i_from_j = max(0.0, E_indirect_on_i_from_j)

                patch_E_indirect_this_bounce[i] += E_indirect_on_i_from_j

        # *** Check 4: Is B updated correctly for the NEXT bounce? ***
        # B_new(n) = rho * (E_direct + E_indirect(n))
        # E_indirect(n) was just calculated in patch_E_indirect_this_bounce
        max_rel_change = 0.0
        for i in range(Np):
            total_E_i = patch_direct_E[i] + patch_E_indirect_this_bounce[i] # E_total for this step
            patch_B_new[i] = patch_refl[i] * total_E_i # B for *next* step

            # Check convergence based on B
            change = abs(patch_B_new[i] - patch_B[i])
            denom = abs(patch_B[i]) + epsilon
            rel_change = change / denom
            if rel_change > max_rel_change: max_rel_change = rel_change

        # Update patch_B for the next iteration
        patch_B[:] = patch_B_new[:]

        # Print statement removed previously - ok
        if max_rel_change < convergence_threshold:
            print("  Radiosity converged.")
            break
    else:
        # Non-convergence warning removed previously - ok
        pass

    # *** Check 5: Is the final returned value correct? ***
    # We need final TOTAL illuminance E = E_direct + E_indirect(all bounces)
    # The final patch_B contains rho * E_total. So E_total = patch_B / rho
    final_E = np.zeros_like(patch_direct_E)
    for i in range(Np):
         if patch_refl[i] > 1e-6:
             final_E[i] = patch_B[i] / patch_refl[i] # E = B / rho
         else:
             final_E[i] = patch_direct_E[i] # If no reflection, E_total = E_direct
         final_E[i] = max(0.0, final_E[i]) # Ensure non-negative

    # This logic seems sound. If the bug was here, it might have been a subtle numerical issue
    # or perhaps related to incorrect patch properties (normals, areas, reflectance).
    # The use of `patch_refl` seems correct now.
    return final_E

@njit
def compute_row_reflection_mc(r, X, Y, patch_centers, patch_normals, patch_areas, patch_total_E, patch_refl, mc_samples):
    # Calculates indirect illuminance on floor points from patch radiosity B = rho * E
    row_vals = np.zeros(X.shape[1], dtype=np.float64)
    patch_B = patch_refl * patch_total_E # Use the final E_total from radiosity

    for c in range(X.shape[1]):
        fx, fy, fz = X[r, c], Y[r, c], 0.0 # Floor point
        val = 0.0

        for p in range(patch_centers.shape[0]):
            # *** Check 1: Correctly skip floor (p=0) and non-emitting patches? ***
            # Skip floor patch (index 0 assumed). Is this always true? Yes, based on build_patches.
            # Skip patches with zero reflectance or zero final E (hence zero B). Yes.
            if p == 0 or patch_refl[p] <= 1e-6 or patch_B[p] <= 1e-6:
                continue

            radiosity_p = patch_B[p]
            pc = patch_centers[p]
            n = patch_normals[p]
            area_p = patch_areas[p]

            # Tangent generation for sampling - seems okay, maybe minor edge cases? Unlikely source of 0 intensity.

            if abs(n[2]) < 0.999: tangent1 = np.array([-n[1], n[0], 0.0])
            else: tangent1 = np.array([1.0, 0.0, 0.0])
            norm_t1 = np.linalg.norm(tangent1)
            if norm_t1 > 1e-9: tangent1 /= norm_t1
            else: tangent1 = np.array([1.0, 0.0, 0.0]) if abs(n[2]) > 0.999 else np.array([-n[1], n[0], 0.0])
            tangent2 = np.cross(n, tangent1)
            norm_t2 = np.linalg.norm(tangent2)
            if norm_t2 > 1e-9: tangent2 /= norm_t2
            else: tangent2 = np.cross(n, np.array([0.0, 1.0, 0.0])); norm_t2=np.linalg.norm(tangent2); tangent2 = tangent2/norm_t2 if norm_t2 > 1e-9 else np.cross(n, np.array([0.0, 0.0, 1.0])); norm_t2=np.linalg.norm(tangent2); tangent2=tangent2/norm_t2 if norm_t2 > 1e-9 else np.array([0., 1., 0.])


            half_side = math.sqrt(area_p) / 2.0
            sample_sum_ff = 0.0
            for _ in range(mc_samples):
                offset1 = np.random.uniform(-half_side, half_side)
                offset2 = np.random.uniform(-half_side, half_side)
                sample_point = pc + offset1*tangent1 + offset2*tangent2

                v_pf = np.array([fx - sample_point[0], fy - sample_point[1], fz - sample_point[2]])
                dist2 = v_pf[0]**2 + v_pf[1]**2 + v_pf[2]**2
                if dist2 < 1e-15: continue
                dist = math.sqrt(dist2)

                cos_p = np.dot(n, v_pf) / dist      # Angle at patch p normal
                nf = np.array([0.0, 0.0, 1.0])      # Floor normal
                cos_f = np.dot(nf, -v_pf) / dist    # Angle at floor point normal

                # *** Check 2: Visibility check? Yes. ***
                if cos_p <= 1e-6 or cos_f <= 1e-6: continue

                # *** Check 3: Form factor term calculation? Yes, seems standard. ***
                geom_term = (cos_p * cos_f) / (math.pi * dist2)
                sample_sum_ff += max(0.0, geom_term)

            avg_geom_term = sample_sum_ff / mc_samples

            # *** Check 4: Contribution calculation? E_f += B_p * avg_geom_term * Area_p ***
            # This looks correct. Illuminance at floor = Sum over patches [ Radiosity_patch * AvgFormFactor_patch->floor_point * Area_patch ]
            val += radiosity_p * avg_geom_term * area_p

        row_vals[c] = val
    return row_vals

def compute_reflection_on_floor(X, Y, patch_centers, patch_normals, patch_areas, patch_total_E, patch_refl,
                                mc_samples=MC_SAMPLES):
    # Parallel execution logic - NO CHANGE NEEDED
    # ... (same as before) ...
    rows, cols = X.shape
    print(f"[INFO] Computing indirect floor illuminance via MC ({mc_samples} samples/patch-point pair)...")
    start_time = time.time()
    # Make sure patch_total_E input here is the output of iterative_radiosity_loop
    results = Parallel(n_jobs=-1)(delayed(compute_row_reflection_mc)(
        r, X, Y, patch_centers, patch_normals, patch_areas, patch_total_E, patch_refl, mc_samples
    ) for r in range(rows))
    end_time = time.time()
    print(f"[INFO] MC reflection calculation finished in {end_time - start_time:.2f}s.")

    out = np.zeros((rows, cols), dtype=np.float64)
    for r, row_vals in enumerate(results):
        out[r, :] = row_vals
    # *** If this function returned zeros before, it means either:
    #     a) patch_total_E was zero for all reflecting patches (unlikely if direct light exists)
    #     b) patch_refl was zero for all patches except floor (possible bug source if REFL_WALL/CEIL were effectively 0)
    #     c) Bug inside compute_row_reflection_mc causing val to remain 0 (cos checks, geom_term calc?)
    # Assuming the primary bug fix was ensuring REFL_FLOOR, REFL_WALL, REFL_CEIL were correctly used in build_patches AND that patch_refl is used correctly in radiosity/MC, this should now work.
    return out

# --- Main Simulation Function ---
def simulate_lighting(cob_flux_params_per_layer, strip_flux_params_per_layer, geometry_data):
    """
    Runs the full lighting simulation for given COB and Strip flux parameters.

    Args:
        cob_flux_params_per_layer (np.array): Flux values for each main COB layer (N elements).
        strip_flux_params_per_layer (np.array): Flux values for each strip layer (N elements, index 0 unused).
        geometry_data (tuple): Pre-calculated geometry: (light_source_positions, X, Y, patches).
                                Patches = (p_centers, p_areas, p_normals, p_refl)

    Returns:
        tuple: (floor_ppfd, X, Y, light_source_positions)
    """
    light_source_positions, X, Y, (p_centers, p_areas, p_normals, p_refl) = geometry_data
    num_layers = len(cob_flux_params_per_layer)
    print(f"\n--- Running Simulation for N={num_layers} ---")

    # 1. Assign flux to individual light sources (COBs and Strips)
    lumens_per_source = pack_luminous_flux(cob_flux_params_per_layer, strip_flux_params_per_layer, light_source_positions)
    total_sim_flux = np.sum(lumens_per_source)
    print(f"[Sim] Total Luminous Flux in Simulation: {total_sim_flux:.2f} lm")
    if abs(total_sim_flux) < 1e-6:
         warnings.warn("[Sim] Total simulation flux is near zero. Results will be zero.", UserWarning)
         return (np.zeros_like(X), X, Y, light_source_positions) # Return zero PPFD

    # 2. Compute Direct Illuminance on Floor and Patches
    print("[Sim] Calculating direct illuminance...")
    start_direct = time.time()
    # Pass combined positions and per-source lumens
    floor_lux_direct = compute_direct_floor(light_source_positions, lumens_per_source, X, Y)
    patch_E_direct = compute_patch_direct(light_source_positions, lumens_per_source, p_centers, p_normals, p_areas)
    print(f"[Sim] Direct calculation time: {time.time() - start_direct:.2f}s")
    print(f"[Sim] Max Direct Floor Lux: {np.max(floor_lux_direct):.2f}")
    print(f"[Sim] Max Direct Patch E: {np.max(patch_E_direct):.2f}")


    # 3. Compute Radiosity (Total Illuminance on Patches)
    # This step should now include reflections correctly if the bug was fixed upstream
    print("[Sim] Calculating radiosity...")
    start_radio = time.time()
    patch_E_total = iterative_radiosity_loop(p_centers, p_normals, patch_E_direct, p_areas, p_refl,
                                            MAX_RADIOSITY_BOUNCES, RADIOSITY_CONVERGENCE_THRESHOLD)
    print(f"[Sim] Radiosity calculation time: {time.time() - start_radio:.2f}s")
    print(f"[Sim] Max Total Patch E: {np.max(patch_E_total):.2f}") # Should be > Max Direct if reflections work

    # 4. Compute Indirect Illuminance on Floor from Patches
    # This step should also yield non-zero results now
    print("[Sim] Calculating indirect illuminance on floor...")
    floor_lux_indirect = compute_reflection_on_floor(X, Y, p_centers, p_normals, p_areas, patch_E_total, p_refl, MC_SAMPLES)
    print(f"[Sim] Max Indirect Floor Lux: {np.max(floor_lux_indirect):.2f}") # Expect non-zero

    # 5. Calculate Total Illuminance (Lux) and Convert to PPFD
    total_luminous_lux = floor_lux_direct + floor_lux_indirect

    total_radiant_Wm2 = total_luminous_lux / LUMINOUS_EFFICACY
    floor_ppfd = total_radiant_Wm2 * CONVERSION_FACTOR

    print(f"[Sim] Simulation Complete. Max PPFD: {np.max(floor_ppfd):.2f} µmol/m²/s")
    return floor_ppfd, X, Y, light_source_positions


# --- Simulation Results Analysis ---
def calculate_metrics(floor_ppfd):
    # ... (same as before) ...
    if floor_ppfd is None or floor_ppfd.size == 0:
        return 0.0, 0.0
    mean_ppfd = np.mean(floor_ppfd)
    if mean_ppfd < 1e-6: return 0.0, 0.0
    rmse = np.sqrt(np.mean((floor_ppfd - mean_ppfd)**2))
    dou = 100 * (1 - rmse / mean_ppfd)
    return mean_ppfd, dou

def calculate_per_layer_ppfd(floor_ppfd, X, Y, light_source_positions, W, L, num_layers):
    """
    Calculates the average PPFD for points corresponding to each COB layer ring.
    Uses the main COB positions to define the rings.
    """
    # Filter to get only main COB positions for defining radii/rings
    main_cob_positions = light_source_positions[light_source_positions[:, 4] == 0]

    if floor_ppfd is None or floor_ppfd.size == 0 or main_cob_positions.size == 0:
        warnings.warn("[Feedback] No PPFD data or main COB positions for layer feedback.", UserWarning)
        # Return overall average for all layers as fallback
        overall_avg = np.mean(floor_ppfd) if floor_ppfd is not None and floor_ppfd.size > 0 else 0
        return np.full(num_layers, overall_avg, dtype=np.float64)

    layer_data = {i: [] for i in range(num_layers)}
    cob_layers = main_cob_positions[:, 3].astype(int)
    center_x, center_y = W / 2.0, L / 2.0

    # Determine characteristic radius for each layer (max distance of MAIN COBs in layer)
    layer_radii = np.zeros(num_layers, dtype=np.float64)
    non_zero_radii = []
    for i in range(num_layers):
        layer_cob_indices = np.where(cob_layers == i)[0]
        if len(layer_cob_indices) > 0:
            layer_cob_coords = main_cob_positions[layer_cob_indices, :2]
            distances = np.sqrt((layer_cob_coords[:, 0] - center_x)**2 +
                                (layer_cob_coords[:, 1] - center_y)**2)
            if distances.size > 0:
                layer_radii[i] = np.max(distances)
                if layer_radii[i] > 1e-6: non_zero_radii.append(layer_radii[i])
            else: layer_radii[i] = layer_radii[i-1] if i > 0 else 0.0
        elif i > 0: layer_radii[i] = layer_radii[i-1]
    layer_radii = np.sort(layer_radii)

    # Define ring boundaries
    ring_boundaries = np.zeros(num_layers, dtype=np.float64)
    if num_layers == 1: ring_boundaries[0] = max(W, L) * 1.5
    else:
        first_ring_radius = layer_radii[1] if len(layer_radii) > 1 and layer_radii[1] > 1e-5 else (min(non_zero_radii) if non_zero_radii else max(W, L) / 2.0)
        ring_boundaries[0] = first_ring_radius / 2.0
        for i in range(1, num_layers - 1):
            r_i = layer_radii[i]
            r_next = layer_radii[i+1]
            if r_next > r_i + 1e-6: ring_boundaries[i] = (r_i + r_next) / 2.0
            else: ring_boundaries[i] = ring_boundaries[i-1]
        ring_boundaries[num_layers - 1] = max(W, L) * 1.5
    for i in range(1, num_layers):
        if ring_boundaries[i] < ring_boundaries[i-1] + 1e-6 : ring_boundaries[i] = ring_boundaries[i-1] + 1e-6
    ring_boundaries[0] = max(1e-6, ring_boundaries[0])

    print(f"[Feedback] Layer Radii (Main COBs): {np.array2string(layer_radii, precision=3)}")
    print(f"[Feedback] Ring Boundaries: {np.array2string(ring_boundaries, precision=3)}")

    # Assign points to layers
    rows, cols = floor_ppfd.shape
    points_assigned = [0] * num_layers
    for r in range(rows):
        for c in range(cols):
            dist_to_center = math.sqrt((X[r,c] - center_x)**2 + (Y[r,c] - center_y)**2)
            assigned_layer = -1
            if dist_to_center <= ring_boundaries[0] + 1e-9: assigned_layer = 0
            else:
                for i in range(1, num_layers):
                    if dist_to_center > ring_boundaries[i-1] + 1e-9 and dist_to_center <= ring_boundaries[i] + 1e-9:
                        assigned_layer = i; break
            # if assigned_layer == -1 and dist_to_center > ring_boundaries[num_layers-1] + 1e-9: # Covered by last boundary extent
            #      assigned_layer = num_layers - 1
            if assigned_layer == -1: # If somehow outside last large boundary, assign to outermost
                if dist_to_center > ring_boundaries[-1]: assigned_layer = num_layers -1

            if assigned_layer != -1:
                layer_data[assigned_layer].append(floor_ppfd[r, c])
                points_assigned[assigned_layer] += 1

    # Calculate average PPFD per layer
    avg_ppfd_per_layer = np.zeros(num_layers, dtype=np.float64)
    overall_avg_ppfd = np.mean(floor_ppfd) if floor_ppfd.size > 0 else 0
    valid_layers = 0
    for i in range(num_layers):
        if layer_data[i]:
            avg_ppfd_per_layer[i] = np.mean(layer_data[i])
            valid_layers += 1
        else:
            avg_ppfd_per_layer[i] = overall_avg_ppfd
            warnings.warn(f"Layer {i} had no PPFD points assigned. Using overall average PPFD ({overall_avg_ppfd:.2f}) as feedback.", UserWarning)

    if valid_layers < num_layers: print(f"[Feedback] Warning: Only {valid_layers}/{num_layers} layers received PPFD points.")
    # else: print(f"[Feedback] All {num_layers} layers received PPFD points.") # Reduce verbose

    return avg_ppfd_per_layer

# --- Optional Plotting Function ---
def plot_heatmap(floor_ppfd, X, Y, light_source_positions, title="Floor PPFD Heatmap", annotation_step=5):
    # ... (Adapt to show main COBs and maybe representative strip markers) ...
    fig, ax = plt.subplots(figsize=(10, 8))
    if floor_ppfd is None or floor_ppfd.size == 0:
        ax.set_title(f"{title} (No Data)")
        return

    extent = [X.min(), X.max(), Y.min(), Y.max()]
    im = ax.imshow(floor_ppfd, cmap='hot', interpolation='nearest', origin='lower', extent=extent, aspect='equal')
    fig.colorbar(im, ax=ax, label="PPFD (µmol/m²/s)")

    # Annotate PPFD values
    rows, cols = floor_ppfd.shape
    step = max(1, annotation_step) # Ensure step is at least 1
    for r in range(0, rows, step):
        for c in range(0, cols, step):
            try:
                ax.text(X[r, c], Y[r, c], f"{floor_ppfd[r, c]:.0f}",
                        ha="center", va="center", color="white", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.5))
            except IndexError: continue

    # Plot MAIN COB positions
    main_cobs = light_source_positions[light_source_positions[:, 4] == 0]
    if len(main_cobs) > 0:
        layers = main_cobs[:, 3].astype(int)
        unique_layers = sorted(np.unique(layers))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_layers)))
        for i, layer in enumerate(unique_layers):
            idx = np.where(layers == layer)[0]
            ax.scatter(main_cobs[idx, 0], main_cobs[idx, 1], marker='o',
                       color=colors[i % len(colors)], edgecolors='black', s=30, label=f"COB Layer {layer}", zorder=5)

    # Optional: Plot representative strip markers (e.g., small dots)
    strip_sources = light_source_positions[light_source_positions[:, 4] == 1]
    if len(strip_sources) > 0:
         ax.scatter(strip_sources[:, 0], strip_sources[:, 1], marker='.',
                    color='red', s=5, label="Strip Sources", zorder=4, alpha=0.6)


    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(fontsize='small', loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
    plt.show(block=False)
    plt.pause(0.1)

# ==============================================================================
# Predictive Model & Refinement Functions (Adapted for Dual Flux)
# ==============================================================================

def prepare_spline_data(known_configs):
    """Prepares data for spline fitting, using ONLY COB fluxes for now."""
    n_list, x_norm_list, flux_list = [], [], []
    min_n, max_n = float('inf'), float('-inf')
    has_valid_data = False

    # --- IMPORTANT WARNING ---
    print("\n" + "="*30 + " WARNING " + "="*30)
    print(" Using placeholder 'known_configs' data derived from simulations")
    print(" with a known reflection bug. Predictions based on this data")
    print(" will be inaccurate until valid data is generated and used.")
    print(" The script structure is updated, but numerical results depend")
    print(" on regenerating 'known_configs' with corrected simulations.")
    print("="*70 + "\n")
    # --- END WARNING ---


    sorted_keys = sorted(known_configs.keys())
    print("[Spline] Preparing data from N:", sorted_keys)
    for n in sorted_keys:
        if n not in known_configs: continue
        cob_fluxes, _ = known_configs[n] # Only use COB fluxes for spline shape

        num_layers_in_config = len(cob_fluxes)
        if num_layers_in_config == 0: continue

        n_actual = num_layers_in_config # Use actual length found
        if n_actual < 2: continue # Need at least 2 layers for spline

        min_n, max_n = min(min_n, n_actual), max(max_n, n_actual)
        norm_positions = np.linspace(0, 1, n_actual)
        for i, flux in enumerate(cob_fluxes):
            n_list.append(n_actual)
            x_norm_list.append(norm_positions[i])
            flux_list.append(flux)
            has_valid_data = True

    if not has_valid_data:
        raise ValueError("[Spline] No valid COB configuration data found for spline fitting.")

    num_data_points = len(flux_list)
    print(f"[Spline] Fitting COB shape using data from N={min_n} to N={max_n}. Total points: {num_data_points}")
    return np.array(n_list), np.array(x_norm_list), np.array(flux_list), min_n, max_n, num_data_points

def fit_and_predict_ratio(n_target, known_configs, actual_ppfds, target_ppfd, plot_fit=False):
    """
    Calculates PPFD/(Total Lumen) ratio using combined COB + Strip flux from known_configs.
    Fits trend models and predicts ratio for n_target.
    """
    n_values_sorted = sorted(known_configs.keys())
    total_fluxes = []
    valid_n_for_ratio = []

    print("\n[Ratio] Calculating PPFD/Total Lumen ratio (COB+Strip) and fitting trend vs N:")
    for n in n_values_sorted:
        if n not in known_configs: continue
        cob_fluxes, strip_fluxes = known_configs[n]
        # Ensure strip_fluxes is array, handle potential None if config incomplete
        if strip_fluxes is None: strip_fluxes = np.zeros_like(cob_fluxes)

        # Check lengths match roughly, adjust strip if needed (e.g. if loaded from old format)
        if len(cob_fluxes) != len(strip_fluxes):
             strip_fluxes = np.pad(strip_fluxes, (0, len(cob_fluxes) - len(strip_fluxes)), 'constant') if len(cob_fluxes) > len(strip_fluxes) else strip_fluxes[:len(cob_fluxes)]

        current_total_flux = np.sum(cob_fluxes) + np.sum(strip_fluxes[1:]) # Sum strips from layer 1

        if current_total_flux > 1e-6:
            # Use actual measured PPFD if available, otherwise use the target as placeholder
            # (Using target might skew the ratio if target wasn't actually achieved)
            # This needs actual PPFD results paired with the known_configs runs.
            # Using placeholder 'actual_ppfds_known' passed in for now.
            ppfd_to_use = actual_ppfds.get(n, target_ppfd) # Default to target if no actual result
            ratio = ppfd_to_use / current_total_flux
            total_fluxes.append(current_total_flux)
            valid_n_for_ratio.append(n)
            print(f"  N={n}: Total Flux={current_total_flux:.1f} (COB={np.sum(cob_fluxes):.1f}, Strip={np.sum(strip_fluxes[1:]):.1f}), "
                  f"PPFD={ppfd_to_use:.2f} (Source: {'Actual' if n in actual_ppfds else 'Target'}), Ratio={ratio:.6e}")
        else:
            print(f"  N={n}: Total Flux (COB+Strip) is zero, skipping ratio calculation.")

    if not valid_n_for_ratio:
        warnings.warn("[Ratio] Could not calculate PPFD/Total Lumen ratio from any configuration. Using fallback estimate.", UserWarning)
        # Fallback: Use a reasonable guess based on efficacy/conversion, maybe 0.005?
        return 0.005

    valid_n_array = np.array(valid_n_for_ratio)
    ratios_array = np.array([actual_ppfds.get(n, target_ppfd) / flux for n, flux in zip(valid_n_for_ratio, total_fluxes)])

    # --- Fit Models (Linear, Quadratic, Exponential) ---
    # ... (Polynomial and Exponential fitting code remains the same as before) ...
    # Quadratic Fit (Degree 2)
    coeffs_quad = None
    poly_func_quad = None
    predicted_ratio_quad = None
    if len(valid_n_array) > 2:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)
            coeffs_quad = np.polyfit(valid_n_array, ratios_array, 2)
        poly_func_quad = np.poly1d(coeffs_quad)
        predicted_ratio_quad = poly_func_quad(n_target)
        print(f"\n[Ratio] Quadratic Fit (Degree 2): Ratio = {poly_func_quad}")
        print(f"[Ratio] Quadratic Prediction for N={n_target}: {predicted_ratio_quad:.6e}")

    # Linear Fit (Degree 1)
    coeffs_lin = None
    poly_func_lin = None
    predicted_ratio_lin = None
    if len(valid_n_array) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)
            coeffs_lin = np.polyfit(valid_n_array, ratios_array, 1)
        poly_func_lin = np.poly1d(coeffs_lin)
        predicted_ratio_lin = poly_func_lin(n_target)
        print(f"\n[Ratio] Linear Fit (Degree 1): Ratio = {poly_func_lin}")
        print(f"[Ratio] Linear Prediction for N={n_target}: {predicted_ratio_lin:.6e}")

    # Exponential Decay Model
    def exp_decay_func(n_vals, a, b, c):
        b_constrained = max(1e-9, b)
        return a * np.exp(-b_constrained * n_vals) + c

    coeffs_exp = None
    predicted_ratio_exp = None
    try:
        initial_guess = [np.max(ratios_array) - np.min(ratios_array), 0.1, np.min(ratios_array)]
        coeffs_exp, _ = curve_fit(exp_decay_func, valid_n_array, ratios_array, p0=initial_guess, maxfev=5000)
        predicted_ratio_exp = exp_decay_func(n_target, *coeffs_exp)
        print(f"\n[Ratio] Exponential Fit (a*exp(-b*N)+c): a={coeffs_exp[0]:.4e}, b={coeffs_exp[1]:.4e}, c={coeffs_exp[2]:.4e}")
        print(f"[Ratio] Exponential Prediction for N={n_target}: {predicted_ratio_exp:.6e}")
    except Exception as e:
        print(f"[Ratio] Could not fit exponential model: {e}")
        predicted_ratio_exp = None # Fallback handled below

    # Select Model (Exp > Quad > Lin > Mean)
    final_predicted_ratio = predicted_ratio_exp if predicted_ratio_exp is not None else \
                           (predicted_ratio_quad if predicted_ratio_quad is not None else \
                           (predicted_ratio_lin if predicted_ratio_lin is not None else np.mean(ratios_array)))

    print(f"\n[Ratio] Using prediction from best available model: {final_predicted_ratio:.6e}")

    # Plotting (optional, same as before)
    if plot_fit:
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_n_array, ratios_array, label='Data Points (Total Flux Ratio)', color='red', zorder=5)
        n_plot = np.linspace(min(valid_n_array), n_target + 1 , 100)
        if poly_func_quad: plt.plot(n_plot, poly_func_quad(n_plot), label=f'Quad Fit (Pred={predicted_ratio_quad:.4e})', ls='--')
        if poly_func_lin: plt.plot(n_plot, poly_func_lin(n_plot), label=f'Lin Fit (Pred={predicted_ratio_lin:.4e})', ls=':')
        if coeffs_exp is not None: plt.plot(n_plot, exp_decay_func(n_plot, *coeffs_exp), label=f'Exp Fit (Pred={predicted_ratio_exp:.4e})', ls='-.')
        plt.scatter([n_target], [final_predicted_ratio], color='blue', marker='x', s=100, zorder=6, label=f'Selected Pred. N={n_target}')
        plt.xlabel("Number of Layers (N)")
        plt.ylabel("PPFD / Total Luminous Flux Ratio")
        plt.title("PPFD/Total Lumen Ratio vs. N and Fitted Models")
        plt.legend(); plt.grid(True); plt.show(block=False); plt.pause(0.1)

    # Sanity check
    if final_predicted_ratio <= 1e-9:
         warnings.warn(f"[Ratio] Final predicted ratio ({final_predicted_ratio:.4e}) is non-positive for N={n_target}. Using ratio from nearest known N.", UserWarning)
         try:
             nearest_n_idx = np.abs(valid_n_array - n_target).argmin()
             final_predicted_ratio = ratios_array[nearest_n_idx]
             print(f"  Using ratio from nearest N ({valid_n_array[nearest_n_idx]}): {final_predicted_ratio:.6e}")
         except Exception:
              fallback_ratio = 0.005
              print(f"  Error finding nearest N ratio, using fallback: {fallback_ratio:.6e}")
              final_predicted_ratio = fallback_ratio

    return final_predicted_ratio


def generate_initial_flux_prediction(num_layers_target, known_configs, actual_ppfds,
                                     target_ppfd=TARGET_PPFD, k=SPLINE_DEGREE,
                                     smoothing_mode=SMOOTHING_FACTOR_MODE, smoothing_mult=SMOOTHING_MULTIPLIER,
                                     clamp_outer=CLAMP_OUTER_LAYER, outer_max=OUTER_LAYER_MAX_FLUX,
                                     initial_ppfd_correction_factor=1.0,
                                     add_strips=True, initial_strip_ratio=INITIAL_STRIP_TO_COB_RATIO):
    """
    Generates initial COB and Strip flux predictions.
    Fits spline to COB data, estimates strip flux based on ratio, scales both to meet target total flux.
    """
    print(f"\n--- Generating Initial Prediction for N={num_layers_target} ---")
    if num_layers_target < 1: raise ValueError("Number of layers must be at least 1.")
    if num_layers_target == 1 and add_strips:
        print("[Predict] N=1, disabling strips as they start at Layer 1.")
        add_strips = False

    # --- Step 1: Fit Spline to Known COB Data ---
    n_coords, x_norm_coords, flux_values, min_n_data, max_n_data, num_data_points = prepare_spline_data(known_configs)

    s_value = None
    if smoothing_mode == "num_points": s_value = float(num_data_points)
    elif smoothing_mode == "multiplier": s_value = float(num_data_points) * smoothing_mult
    elif isinstance(smoothing_mode, (int, float)): s_value = float(smoothing_mode)

    if s_value is not None: print(f"[Spline] Using smoothing factor s = {s_value:.2f}")
    else: print("[Spline] Using default smoothing factor s (estimated by FITPACK)")

    # Adjust spline degree k if needed
    k_spline = k
    unique_n = len(np.unique(n_coords))
    unique_x = len(np.unique(x_norm_coords))
    if unique_n <= k or unique_x <= k:
         k_spline = max(1, min(unique_n - 1, unique_x - 1, k))
         warnings.warn(f"[Spline] Insufficient unique coordinates for degree {k}. Reducing to {k_spline}.", UserWarning)

    # Fit Spline
    try:
        with warnings.catch_warnings():
             warnings.filterwarnings("ignore", message=".*storage space.*", category=UserWarning)
             warnings.filterwarnings("ignore", message=".*knots required.*", category=UserWarning)
             warnings.filterwarnings("ignore", message=".*Warning: s=.*", category=UserWarning)
             warnings.filterwarnings("ignore", message=".*Warning: ier=.*", category=UserWarning)
             spline = SmoothBivariateSpline(n_coords, x_norm_coords, flux_values, kx=k_spline, ky=k_spline, s=s_value)
        print(f"[Spline] Successfully fitted COB shape spline (degree k={k_spline}).")
    except Exception as e: print(f"\n[Spline] Error fitting spline: {e}"); raise

    # --- Step 2: Evaluate Spline for Target N (COB Shape) ---
    if num_layers_target == 1:
        cob_flux_shape = np.array([1.0]) # Single layer, shape is just 1
    else:
        x_target_norm = np.linspace(0, 1, num_layers_target)
        n_target_array = np.full_like(x_target_norm, num_layers_target)
        cob_flux_shape = spline(n_target_array, x_target_norm, grid=False)
        cob_flux_shape = np.maximum(cob_flux_shape, 0) # Ensure non-negative
        if num_layers_target < min_n_data or num_layers_target > max_n_data:
            warnings.warn(f"[Spline] Extrapolating COB flux profile for N={num_layers_target} (data range {min_n_data}-{max_n_data}).", UserWarning)

    # --- Step 3: Define Initial Strip Flux Shape ---
    strip_flux_shape = np.zeros(num_layers_target, dtype=np.float64)
    if add_strips and num_layers_target > 1:
        # Simple initial guess: Strip flux is a ratio of COB flux for layers > 0
        strip_flux_shape[1:] = cob_flux_shape[1:] * initial_strip_ratio
        strip_flux_shape = np.maximum(strip_flux_shape, 0) # Ensure non-negative
        print(f"[Predict] Initial strip shape based on ratio: {initial_strip_ratio:.2f}")

    # --- Step 4: Predict Target Total Flux ---
    predicted_ppfd_per_lumen = fit_and_predict_ratio(num_layers_target, known_configs, actual_ppfds, target_ppfd)
    if predicted_ppfd_per_lumen <= 1e-9:
         warnings.warn("[Predict] Predicted PPFD/Total Lumen ratio is non-positive. Using arbitrary target flux.", UserWarning)
         target_total_flux = 200000.0 # Arbitrary large flux
    else:
        target_total_flux_uncorrected = target_ppfd / predicted_ppfd_per_lumen
        target_total_flux = target_total_flux_uncorrected * initial_ppfd_correction_factor
        print(f"[Predict] Target Total Flux (Uncorrected): {target_total_flux_uncorrected:.2f} lm")
        if abs(initial_ppfd_correction_factor - 1.0) > 1e-4:
            print(f"[Predict] Applying Initial Correction Factor: {initial_ppfd_correction_factor:.4f}")
        print(f"[Predict] Final Initial Target Total Flux: {target_total_flux:.2f} lm")

    # --- Step 5: Scale Shapes to Target Total Flux & Apply Clamping (to outer COB) ---
    shape_total_flux = np.sum(cob_flux_shape) + np.sum(strip_flux_shape[1:]) # Sum strips from layer 1
    print(f"[Predict] Combined Shape Total Flux (COB + Strip): {shape_total_flux:.2f}")

    if shape_total_flux <= 1e-6:
         warnings.warn("[Predict] Combined shape resulted in near-zero total flux. Cannot scale reliably. Returning zero fluxes.", UserWarning)
         return np.zeros_like(cob_flux_shape), np.zeros_like(strip_flux_shape), 0.0

    flux_initial_cob = np.zeros_like(cob_flux_shape)
    flux_initial_strip = np.zeros_like(strip_flux_shape)
    outer_cob_clamp_value_actual = 0.0

    if clamp_outer and num_layers_target > 0:
        outer_cob_layer_idx = num_layers_target - 1
        print(f"[Predict] Applying outer MAIN COB layer clamp: Layer {outer_cob_layer_idx} = {outer_max:.1f} lm")

        if outer_max >= target_total_flux:
             warnings.warn("[Predict] Outer COB clamp value >= target total flux. Assigning budget to outer COB, others zero.", UserWarning)
             flux_initial_cob[outer_cob_layer_idx] = target_total_flux
             outer_cob_clamp_value_actual = target_total_flux
             # Inner COBs and all strips remain zero
        else:
             # Assign clamp value to outer COB layer
             flux_initial_cob[outer_cob_layer_idx] = outer_max
             outer_cob_clamp_value_actual = outer_max

             # Remaining budget for inner COBs and all strips
             inner_flux_budget = target_total_flux - outer_cob_clamp_value_actual
             if inner_flux_budget < 0: inner_flux_budget = 0

             # Get sum of the shapes for inner COBs and all relevant strips
             inner_cob_shape_sum = np.sum(cob_flux_shape[:outer_cob_layer_idx])
             inner_strip_shape_sum = np.sum(strip_flux_shape[1:]) # All strips contribute to inner budget
             inner_shape_total_sum = inner_cob_shape_sum + inner_strip_shape_sum

             if inner_shape_total_sum <= 1e-6:
                  warnings.warn("[Predict] Sum of inner COB/Strip shapes is near zero. Cannot scale inner sources.", UserWarning)
                  # Inner sources remain zero
             else:
                  # Scale inner COBs and strips based on their shape proportion and remaining budget
                  S_inner = inner_flux_budget / inner_shape_total_sum
                  print(f"[Predict] Scaling inner COBs (0-{outer_cob_layer_idx-1}) and Strips (1-{num_layers_target-1}) by factor: {S_inner:.6f}")
                  flux_initial_cob[:outer_cob_layer_idx] = S_inner * cob_flux_shape[:outer_cob_layer_idx]
                  flux_initial_strip[1:] = S_inner * strip_flux_shape[1:] # Scale strips from layer 1

    else:
        # No clamping - Apply global scaling to COBs and Strips
        print("[Predict] Applying global scaling factor (no outer COB clamp).")
        global_scale_factor = target_total_flux / shape_total_flux
        print(f"[Predict] Global scaling factor: {global_scale_factor:.6f}")
        flux_initial_cob = cob_flux_shape * global_scale_factor
        flux_initial_strip = strip_flux_shape * global_scale_factor # Includes scaling strip[0] to 0
        if num_layers_target > 0:
            outer_cob_clamp_value_actual = flux_initial_cob[-1] # Store the resulting outer COB value

    # Ensure non-negativity again after scaling
    flux_initial_cob = np.maximum(flux_initial_cob, 0)
    flux_initial_strip = np.maximum(flux_initial_strip, 0)
    final_total_flux = np.sum(flux_initial_cob) + np.sum(flux_initial_strip[1:])

    if not np.isclose(final_total_flux, target_total_flux, rtol=1e-3):
         warnings.warn(f"[Predict] Initial predicted total flux ({final_total_flux:.2f}) differs slightly from target ({target_total_flux:.2f}).", UserWarning)

    print(f"[Predict] Initial COB Flux Profile Total: {np.sum(flux_initial_cob):.2f}")
    print(f"[Predict] Initial Strip Flux Profile Total (Layers 1+): {np.sum(flux_initial_strip[1:]):.2f}")

    # Return both flux arrays and the target total flux used
    return flux_initial_cob, flux_initial_strip, target_total_flux, outer_cob_clamp_value_actual


def apply_refinement_step(cob_flux_input, strip_flux_input, ppfd_feedback_per_layer,
                           target_ppfd, target_total_flux, outer_cob_clamp_value, iteration,
                           learn_rate=REFINEMENT_LEARNING_RATE, mult_min=MULTIPLIER_MIN, mult_max=MULTIPLIER_MAX,
                           add_strips=True):
    """
    Refines inner COB fluxes and strip fluxes based on per-layer PPFD feedback,
    maintaining outer COB clamp and total flux budget.
    """
    num_layers = len(cob_flux_input)
    if ppfd_feedback_per_layer is None or len(ppfd_feedback_per_layer) != num_layers:
        warnings.warn(f"[Refine] PPFD feedback invalid for Iteration {iteration}. Skipping refinement.", UserWarning)
        return cob_flux_input, strip_flux_input # Return unchanged inputs

    print(f"\n--- Applying PPFD Refinement Iteration {iteration} ---")
    print(f"[Refine] Target Total Flux for this step: {target_total_flux:.2f} lm")
    if num_layers > 0:
        print(f"[Refine] Outer Main COB Layer {num_layers-1} clamped to: {outer_cob_clamp_value:.2f} lm")
    print(f"[Refine] Using Learning Rate: {learn_rate:.2f}")

    # Calculate multipliers based on relative errors (same as before)
    errors = target_ppfd - ppfd_feedback_per_layer
    relative_errors = np.divide(errors, target_ppfd, out=np.zeros_like(errors), where=abs(target_ppfd)>1e-9)
    multipliers = 1.0 + learn_rate * relative_errors
    multipliers = np.clip(multipliers, mult_min, mult_max)

    print(f"[Refine] PPFD Feedback (Avg per Layer Ring):\n{np.array2string(ppfd_feedback_per_layer, precision=2, suppress_small=True)}")
    print(f"[Refine] Refinement Multipliers (min={mult_min:.2f}, max={mult_max:.2f}):\n{np.array2string(multipliers, precision=4, suppress_small=True)}")

    # --- Apply multipliers to inner COBs and corresponding Strips ---
    outer_cob_layer_idx = num_layers - 1
    flux_cob_refined = np.copy(cob_flux_input)
    flux_strip_refined = np.copy(strip_flux_input)

    # Refine inner COBs (Layers 0 to N-2)
    if outer_cob_layer_idx > 0:
        flux_cob_refined[:outer_cob_layer_idx] = cob_flux_input[:outer_cob_layer_idx] * multipliers[:outer_cob_layer_idx]

    # Refine Strips (Layers 1 to N-1, using multipliers from corresponding layer index)
    if add_strips and num_layers > 1:
        flux_strip_refined[1:] = strip_flux_input[1:] * multipliers[1:]

    # Ensure non-negativity
    flux_cob_refined = np.maximum(flux_cob_refined, 0)
    flux_strip_refined = np.maximum(flux_strip_refined, 0)

    # --- Rescale inner COBs and all Strips to meet the budget ---
    # Budget = Target Total Flux - Outer COB Clamp Value
    inner_flux_budget = target_total_flux - outer_cob_clamp_value
    if inner_flux_budget < 0:
         warnings.warn(f"[Refine] Target total flux ({target_total_flux:.2f}) < outer COB clamp ({outer_cob_clamp_value:.2f}) in iter {iteration}. Setting inner budget to zero.", UserWarning)
         inner_flux_budget = 0

    # Calculate the current total flux of the sources to be rescaled
    refined_inner_cob_total = np.sum(flux_cob_refined[:outer_cob_layer_idx]) if outer_cob_layer_idx > 0 else 0.0
    refined_strip_total = np.sum(flux_strip_refined[1:]) if add_strips and num_layers > 1 else 0.0
    total_flux_to_rescale = refined_inner_cob_total + refined_strip_total

    print(f"[Refine] Refined Inner COB Flux Total (Before Rescale): {refined_inner_cob_total:.2f}")
    if add_strips: print(f"[Refine] Refined Strip Flux Total (Before Rescale): {refined_strip_total:.2f}")
    print(f"[Refine] Inner+Strip Flux Budget: {inner_flux_budget:.2f}")

    # Calculate rescaling factor
    if total_flux_to_rescale <= 1e-6:
         warnings.warn(f"[Refine] Refined inner/strip flux total near zero in iter {iteration}. Cannot rescale.", UserWarning)
         # Keep inner/strip fluxes as they are (likely near zero)
         final_fluxes_cob = np.copy(flux_cob_refined)
         final_fluxes_strip = np.copy(flux_strip_refined)
         # Still need to set outer COB clamp
         if num_layers > 0: final_fluxes_cob[outer_cob_layer_idx] = outer_cob_clamp_value
    else:
         rescale_factor = inner_flux_budget / total_flux_to_rescale
         print(f"[Refine] Inner COB / Strip rescale factor: {rescale_factor:.6f}")

         # Apply rescale factor
         final_fluxes_cob = np.copy(flux_cob_refined)
         final_fluxes_strip = np.copy(flux_strip_refined)

         if outer_cob_layer_idx > 0:
             final_fluxes_cob[:outer_cob_layer_idx] = flux_cob_refined[:outer_cob_layer_idx] * rescale_factor
         if add_strips and num_layers > 1:
             final_fluxes_strip[1:] = flux_strip_refined[1:] * rescale_factor

         # Ensure non-negativity after rescaling
         final_fluxes_cob = np.maximum(final_fluxes_cob, 0)
         final_fluxes_strip = np.maximum(final_fluxes_strip, 0)

         # Set the clamped outer COB layer value
         if num_layers > 0:
             final_fluxes_cob[outer_cob_layer_idx] = outer_cob_clamp_value

    # Final check on total flux
    actual_total_flux_iter = np.sum(final_fluxes_cob) + np.sum(final_fluxes_strip[1:])
    if not np.isclose(actual_total_flux_iter, target_total_flux, rtol=1e-3):
         warnings.warn(f"[Refine] Total flux ({actual_total_flux_iter:.2f}) after refinement iter {iteration} differs slightly from target ({target_total_flux:.2f}).", UserWarning)
    print(f"[Refine] Refined COB Flux Total (Iter {iteration}): {np.sum(final_fluxes_cob):.2f}")
    if add_strips: print(f"[Refine] Refined Strip Flux Total (Iter {iteration}): {np.sum(final_fluxes_strip[1:]):.2f}")


    return final_fluxes_cob, final_fluxes_strip

# ==============================================================================
# Main Automation Workflow
# ==============================================================================

def calculate_room_dimensions(num_layers_target, base_n, base_w, base_l, increment):
    """Calculates room dimensions based on target number of layers."""
    if num_layers_target <= base_n:
        return base_w, base_l
    else:
        diff_n = num_layers_target - base_n
        new_w = base_w + diff_n * increment
        new_l = base_l + diff_n * increment
        return new_w, new_l

def run_automated_prediction(num_layers_target, target_ppfd, target_dou,
                             max_iterations, ppfd_tolerance,
                             base_n, base_w, base_l, size_increment, sim_h, # Dynamic size params
                             add_strips, num_mini_cobs): # Strip params
    """
    Automates the Predict -> Simulate -> Refine cycle with dynamic sizing and optional strips.
    Returns:
        tuple: (success_flag, final_cob_fluxes, final_strip_fluxes, final_ppfd, final_dou, iterations_run)
    """
    print(f"===== Starting Automated Prediction for N={num_layers_target} =====")
    print(f"Target PPFD: {target_ppfd:.1f} +/- {ppfd_tolerance*100:.1f}%")
    print(f"Target DOU: > {target_dou:.1f}%")
    print(f"Max Iterations: {max_iterations}")
    print(f"Adding LED Strips: {'Yes' if add_strips else 'No'}")
    if add_strips: print(f"Mini-COBs per Strip Segment: {num_mini_cobs}")

    # --- Calculate Dynamic Room Dimensions ---
    sim_w, sim_l = calculate_room_dimensions(num_layers_target, base_n, base_w, base_l, size_increment)
    print(f"\n[Setup] Dynamic Room Size for N={num_layers_target}: W={sim_w:.4f}m, L={sim_l:.4f}m")

    # --- Pre-calculate Geometry (using dynamic W, L) ---
    print("[Setup] Preparing simulation geometry...")
    try:
        # Pass dynamic W, L and strip config
        geometry_data = prepare_geometry(sim_w, sim_l, sim_h, num_layers_target, add_strips, num_mini_cobs)
        light_source_positions, X_grid, Y_grid, _ = geometry_data # Unpack for later use
    except Exception as e:
        print(f"[Error] Failed to prepare geometry: {e}")
        return False, None, None, 0.0, 0.0, 0

    # --- Initialize PPFD Correction History ---
    correction_history = deque(maxlen=3)
    current_ppfd_correction = EMPIRICAL_PPFD_CORRECTION.get(num_layers_target, 1.0)
    correction_history.append(current_ppfd_correction)
    print(f"[Setup] Initial PPFD Correction Factor: {current_ppfd_correction:.5f}")

    # --- Initial Prediction (Iteration 0) ---
    # Use actual PPFDs for known N=16, 17, 18 for ratio fit - NEEDS UPDATE WITH REAL RESULTS POST-FIX
    # Placeholder: Using target PPFD for missing actual results in ratio calc for now
    actual_ppfds_known = { 16: 1248.63, 17: 1246.87, 18: 1247.32 } # !!! OLD DATA - REPLACE !!!
    print("[Warning] Using OLD/placeholder 'actual_ppfds_known' for initial ratio fit. REPLACE with actual results post-bugfix.")


    try:
        initial_smoothed_correction = np.mean(list(correction_history))
        current_cob_fluxes, current_strip_fluxes, target_total_flux, outer_cob_clamp_value = \
            generate_initial_flux_prediction(
                num_layers_target,
                known_configs, # Contains placeholder data!
                actual_ppfds_known, # Contains placeholder data!
                target_ppfd=target_ppfd,
                initial_ppfd_correction_factor=initial_smoothed_correction,
                add_strips=add_strips,
                initial_strip_ratio=INITIAL_STRIP_TO_COB_RATIO,
                clamp_outer=CLAMP_OUTER_LAYER,
                outer_max=OUTER_LAYER_MAX_FLUX
            )
    except Exception as e:
        print(f"[Error] Failed during initial flux prediction: {e}")
        import traceback; traceback.print_exc()
        return False, None, None, 0.0, 0.0, 0

    final_cob_fluxes = current_cob_fluxes
    final_strip_fluxes = current_strip_fluxes
    success = False
    floor_ppfd = None

    # --- Iteration Loop ---
    for i in range(max_iterations + 1):
        print(f"\n======= Iteration {i} / {max_iterations} =======")
        if target_total_flux < 0: print("[Warn] Target total flux negative, capping at 0."); target_total_flux = 0

        print(f"[Fluxes] Current COB Flux Profile (N={num_layers_target}):")
        print(np.array2string(current_cob_fluxes, precision=4, suppress_small=True))
        if add_strips:
            print(f"[Fluxes] Current Strip Flux Profile (N={num_layers_target}, Layer 0 unused):")
            print(np.array2string(current_strip_fluxes, precision=4, suppress_small=True))
        print(f"    Total Flux: {np.sum(current_cob_fluxes) + np.sum(current_strip_fluxes[1:]):.2f} (Target: {target_total_flux:.2f})")

        # --- Simulate ---
        start_sim_time = time.time()
        try:
            current_cob_fluxes = np.maximum(0, current_cob_fluxes)
            current_strip_fluxes = np.maximum(0, current_strip_fluxes)
            floor_ppfd, _, _, _ = simulate_lighting(current_cob_fluxes, current_strip_fluxes, geometry_data)
        except Exception as e:
            print(f"[Error] Simulation failed in Iteration {i}: {e}")
            import traceback; traceback.print_exc()
            return False, current_cob_fluxes, current_strip_fluxes, 0.0, 0.0, i
        end_sim_time = time.time()
        print(f"[Sim] Iteration {i} simulation time: {end_sim_time - start_sim_time:.2f}s")


        # --- Analyze Results ---
        avg_ppfd, dou = calculate_metrics(floor_ppfd)
        print(f"\n[Results] Iteration {i}:")
        print(f"  Average PPFD = {avg_ppfd:.2f} µmol/m²/s")
        print(f"  DOU (RMSE based) = {dou:.2f}%")

        # --- Check Criteria ---
        ppfd_met = abs(avg_ppfd - target_ppfd) <= ppfd_tolerance * target_ppfd if avg_ppfd > 1e-3 else False
        dou_met = dou >= target_dou

        if ppfd_met and dou_met:
            print(f"\n[Success] Criteria met after {i} iterations!")
            success = True
            final_cob_fluxes = current_cob_fluxes
            final_strip_fluxes = current_strip_fluxes
            if SHOW_FINAL_HEATMAP:
                 plot_heatmap(floor_ppfd, X_grid, Y_grid, light_source_positions, title=f"Final PPFD N={num_layers_target} (Iter {i})", annotation_step=ANNOTATION_STEP)
            break
        elif i == max_iterations:
            print(f"\n[Failure] Maximum iterations ({max_iterations}) reached. Criteria not met.")
            final_cob_fluxes = current_cob_fluxes
            final_strip_fluxes = current_strip_fluxes
            if SHOW_FINAL_HEATMAP:
                 plot_heatmap(floor_ppfd, X_grid, Y_grid, light_source_positions, title=f"Final PPFD N={num_layers_target} (Iter {i} - Failed)", annotation_step=ANNOTATION_STEP)
            break

        # --- Refine (if not last iteration and criteria not met) ---
        print("\n[Refine] Criteria not met, proceeding to refinement...")

        # Calculate per-layer PPFD feedback (based on main COB rings)
        ppfd_feedback = calculate_per_layer_ppfd(floor_ppfd, X_grid, Y_grid, light_source_positions, sim_w, sim_l, num_layers_target)
        if ppfd_feedback is None or np.any(np.isnan(ppfd_feedback)):
             print("[Error] Failed to get valid per-layer PPFD feedback. Aborting.")
             final_avg_ppfd, final_dou = calculate_metrics(floor_ppfd)
             return False, current_cob_fluxes, current_strip_fluxes, final_avg_ppfd, final_dou, i

        # --- Update Correction Factor (Smoothed) ---
        if avg_ppfd > 1e-6: # Avoid division by zero
             raw_correction = target_ppfd / avg_ppfd
             # *** Apply Clipping ***
             latest_correction = np.clip(raw_correction, 0.2, 25.0) # Use the WIDE range
             correction_history.append(latest_correction)
             current_ppfd_correction = np.mean(list(correction_history))
             EMPIRICAL_PPFD_CORRECTION[num_layers_target] = latest_correction
             print(f"[Refine] Raw Correction Factor Needed: {raw_correction:.5f}")
             print(f"[Refine] Clipped/Latest Correction Factor: {latest_correction:.5f}")
             print(f"[Refine] Smoothed PPFD Correction Factor (Avg of last {len(correction_history)}): {current_ppfd_correction:.5f}")
        else:
             warnings.warn("[Refine] Average PPFD near zero, cannot update correction factor.", UserWarning)
             # current_ppfd_correction remains the mean of the existing history

        # --- Recalculate the target total flux for the *next* iteration's scaling ---
        # --- using the *updated smoothed* correction factor ---
        predicted_ppfd_per_lumen = fit_and_predict_ratio(num_layers_target, known_configs, actual_ppfds_known, target_ppfd) # Still uses placeholder base data
        if predicted_ppfd_per_lumen > 1e-9:
             target_total_flux_uncorrected = target_ppfd / predicted_ppfd_per_lumen
             # Calculate the target flux needed for the NEXT refinement step/iteration
             next_target_total_flux = target_total_flux_uncorrected * current_ppfd_correction # Apply **UPDATED SMOOTHED** correction
             if next_target_total_flux < 0:
                  warnings.warn(f"[Refine] Calculated next target total flux ({next_target_total_flux:.2f}) is negative. Setting to 0.", UserWarning)
                  next_target_total_flux = 0
        else:
             warnings.warn("[Refine] PPFD/Lumen ratio is non-positive. Cannot recalculate target flux, using previous value.", UserWarning)
             # Fallback: Keep the target flux the same as the current iteration's beginning value
             next_target_total_flux = target_total_flux # Use the value from the start of this iteration


        # --- Apply the refinement step ---
        try:
            # Determine the clamp value to use in the refinement scaling.
            # It should be the configured hardware limit, BUT no larger than the calculated *next* total flux budget.
            base_outer_clamp = OUTER_LAYER_MAX_FLUX if CLAMP_OUTER_LAYER and num_layers_target > 0 else (current_cob_fluxes[-1] if num_layers_target > 0 else 0)
            # The effective clamp value passed to the refinement function
            current_outer_cob_clamp_for_refine = min(base_outer_clamp, next_target_total_flux)

            refined_cob_fluxes, refined_strip_fluxes = apply_refinement_step(
                current_cob_fluxes,
                current_strip_fluxes,
                ppfd_feedback,
                target_ppfd,
                next_target_total_flux,           # <--- PASS THE NEWLY CALCULATED TARGET FLUX
                current_outer_cob_clamp_for_refine, # <--- PASS THE CORRECTLY LIMITED CLAMP VALUE
                i + 1,                             # Iteration number for logging
                add_strips=add_strips
            )
            # Update the fluxes for the next simulation run
            current_cob_fluxes = refined_cob_fluxes
            current_strip_fluxes = refined_strip_fluxes
            # Update the main target_total_flux variable so the *next* loop reports the correct target
            target_total_flux = next_target_total_flux # IMPORTANT: Update the loop's target value

        except Exception as e:
            print(f"[Error] Refinement failed in Iteration {i}: {e}")
            import traceback; traceback.print_exc()
            # Use metrics from current state as final, mark as failed
            final_avg_ppfd, final_dou = calculate_metrics(floor_ppfd)
            return False, current_cob_fluxes, current_strip_fluxes, final_avg_ppfd, final_dou, i

    # --- End of Loop ---
    final_avg_ppfd, final_dou = calculate_metrics(floor_ppfd)

    print("\n===== Automated Prediction Finished =====")
    print(f"Success: {success}")
    print(f"Iterations Run: {i}")
    print(f"Final Average PPFD: {final_avg_ppfd:.2f}")
    print(f"Final DOU: {final_dou:.2f}")
    print("\nFinal COB Flux Assignments:")
    print(np.array2string(final_cob_fluxes, precision=4, suppress_small=True))
    if add_strips:
        print("\nFinal Strip Flux Assignments (Layer 0 unused):")
        print(np.array2string(final_strip_fluxes, precision=4, suppress_small=True))
    print(f"\n    Total Final Flux: {np.sum(final_cob_fluxes) + np.sum(final_strip_fluxes[1:]):.2f}")


    return success, final_cob_fluxes, final_strip_fluxes, final_avg_ppfd, final_dou, i

# ==============================================================================
# Script Execution
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()

    # --- IMPORTANT: Add user confirmation about placeholder data ---
    print("*"*75)
    print("! WARNING: This script currently uses PLACEHOLDER 'known_configs' data. !")
    print("! The simulation results and predictions will be inaccurate until     !")
    print("! this data is replaced with results from the corrected simulation.   !")
    print("*"*75)
    # proceed = input("Do you want to continue with placeholder data? (yes/no): ")
    # if proceed.lower() != 'yes':
    #     print("Exiting.")
    #     exit()
    print("\nProceeding with placeholder data for structural testing...\n")
    # ---

    success, final_cob_fluxes, final_strip_fluxes, final_ppfd, final_dou, iters = run_automated_prediction(
        NUM_LAYERS_TARGET,
        TARGET_PPFD,
        TARGET_DOU,
        MAX_ITERATIONS,
        PPFD_TOLERANCE,
        BASE_N, BASE_W, BASE_L, SIZE_INCREMENT_PER_N, SIM_H, # Dynamic size args
        ADD_STRIPS, NUM_MINI_COBS_PER_SEGMENT # Strip args
    )

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")

    # Optional: Save final successful fluxes
    if success:
        # Example: Save to CSV
        try:
            filename_base = f"successful_flux_N{NUM_LAYERS_TARGET}"
            # Save COB fluxes
            with open(f"{filename_base}_COB.csv", 'w', newline='') as f:
                writer = csv.writer(f); writer.writerow(['Layer', 'Flux']);
                for i, flux in enumerate(final_cob_fluxes): writer.writerow([i, flux])
            print(f"Saved successful COB fluxes to {filename_base}_COB.csv")
            # Save Strip fluxes
            if ADD_STRIPS and final_strip_fluxes is not None:
                 with open(f"{filename_base}_Strip.csv", 'w', newline='') as f:
                     writer = csv.writer(f); writer.writerow(['Layer', 'Flux']);
                     for i, flux in enumerate(final_strip_fluxes): writer.writerow([i, flux]) # Include layer 0 (should be 0)
                 print(f"Saved successful Strip fluxes to {filename_base}_Strip.csv")

            # *** ADD TO KNOWN_CONFIGS (IN MEMORY) - CAUTION WITH PLACEHOLDERS ***
            # Only update if we trust the result (i.e., not run with placeholder base data)
            # known_configs[NUM_LAYERS_TARGET] = (final_cob_fluxes, final_strip_fluxes)
            # print(f"Updated known_configs in memory for N={NUM_LAYERS_TARGET}")

        except Exception as e:
            print(f"Error saving final fluxes to CSV: {e}")

    # Keep plots open
    if SHOW_FINAL_HEATMAP and plt.get_fignums():
        print("\nClose plot window(s) to exit.")
        plt.show(block=True)
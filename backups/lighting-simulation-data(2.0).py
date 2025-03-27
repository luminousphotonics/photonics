# --- START OF REVISED FILE lighting-simulation-data(2.0).py ---

#!/usr/bin/env python3
import csv
import math
import numpy as np
from numba import njit, prange
from functools import lru_cache
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import argparse
import warnings # Import warnings

# Suppress NumbaPerformanceWarning for np.searchsorted boundary check
from numba import NumbaPerformanceWarning

warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)


# ------------------------------------
# 1) Basic Config & Reflectances
# ------------------------------------
REFL_WALL = 0.8
REFL_CEIL = 0.8
REFL_FLOOR = 0.1

LUMINOUS_EFFICACY = 182.0  # lumens/W
# <<< !!! SET ACTUAL PATH to your SPD file !!! >>>
SPD_FILE = "/Users/austinrouse/photonics/backups/spd_data.csv"

MAX_RADIOSITY_BOUNCES = 10
RADIOSITY_CONVERGENCE_THRESHOLD = 1e-3

WALL_SUBDIVS_X = 10 # Subdivisions along W or L for walls
WALL_SUBDIVS_Y = 5  # Subdivisions along H for walls
CEIL_SUBDIVS_X = 10
CEIL_SUBDIVS_Y = 10

FLOOR_GRID_RES = 0.08  # m
FIXED_NUM_LAYERS = 10 # Tied to the structure of build_cob_positions and params length
MC_SAMPLES = 128 # Monte Carlo samples for indirect floor illumination

# --- LED Strip Module Configuration (Datasheet Based) ---
# <<< !!! SET ACTUAL PATH to your Strip IES file !!! >>>
STRIP_IES_FILE = "/Users/austinrouse/photonics/backups/Standard_Horti_G2.ies"
STRIP_MODULE_LENGTH = 0.561 # meters (from 561.0 mm)
# STRIP_MODULE_LUMENS is now defined per layer in main()

# --- Constants for emitter types ---
EMITTER_TYPE_COB = 0
EMITTER_TYPE_STRIP = 1

# --- COB Configuration ---
# <<< !!! SET ACTUAL PATH to your COB IES file !!! >>>
COB_IES_FILE = "/Users/austinrouse/photonics/backups/cob.ies"

# ------------------------------------
# 4.5) Load and Prepare Full IES Data (REVISED)
# ------------------------------------

def parse_ies_file_full(ies_filepath):
    """
    Parses an IESNA:LM-63-1995 file format.
    Extracts full 2D candela data, vertical/horizontal angles,
    and normalization lumens.
    """
    print(f"[IES - Full] Attempting to parse file: {ies_filepath}")
    try:
        with open(ies_filepath, 'r', encoding='ascii', errors='ignore') as f: # Added encoding
            lines = [line.strip() for line in f if line.strip()] # Read non-empty lines
    except FileNotFoundError:
        print(f"[IES - Full] Error: File not found at '{ies_filepath}'")
        return None, None, None, None
    except Exception as e:
        print(f"[IES - Full] Error reading file '{ies_filepath}': {e}")
        return None, None, None, None

    try:
        line_idx = 0
        # Skip header lines until TILT=
        while line_idx < len(lines) and not lines[line_idx].upper().startswith('TILT='):
            line_idx += 1
        if line_idx >= len(lines): raise ValueError("TILT= line not found")
        tilt_line = lines[line_idx]
        # Check for TILT=NONE or specific tilt data (for future use, currently ignored)
        if "NONE" not in tilt_line.upper():
            print(f"[IES - Full] Warning: TILT directive '{tilt_line}' found but ignored. Assuming standard orientation.")
        line_idx += 1

        # --- Parameter Line 1 ---
        if line_idx >= len(lines): raise ValueError("Missing parameter line 1")
        params1 = lines[line_idx].split()
        num_lamps = int(params1[0])
        lumens_per_lamp = float(params1[1]) # This is our normalization base
        multiplier = float(params1[2])
        num_v_angles = int(params1[3])
        num_h_angles = int(params1[4])
        photometric_type = int(params1[5]) # 1=C, 2=B, 3=A
        units_type = int(params1[6]) # 1=feet, 2=meters
        # width, length, height (indices 7, 8, 9) - ignored for point/strip sources
        line_idx += 1

        # --- Parameter Line 2 ---
        if line_idx >= len(lines): raise ValueError("Missing parameter line 2")
        params2 = lines[line_idx].split()
        ballast_factor = float(params2[0])
        # future_use = float(params2[1]) # Ignored
        input_watts = float(params2[2])
        line_idx += 1

        # --- Read Vertical Angles ---
        if line_idx >= len(lines): raise ValueError("Missing vertical angles")
        v_angles_list = []
        while len(v_angles_list) < num_v_angles:
            if line_idx >= len(lines): raise ValueError(f"File ended while reading vertical angles (expected {num_v_angles})")
            v_angles_list.extend([float(a) for a in lines[line_idx].split()])
            line_idx += 1
        vertical_angles = np.array(v_angles_list, dtype=np.float64)
        if len(vertical_angles) != num_v_angles:
            raise ValueError(f"Mismatch in vertical angle count (Expected {num_v_angles}, Found {len(vertical_angles)})")
        # Ensure sorted
        if not np.all(np.diff(vertical_angles) >= 0):
            print("[IES - Full] Warning: Vertical angles were not sorted, sorting them now.")
            vertical_angles = np.sort(vertical_angles)


        # --- Read Horizontal Angles ---
        if line_idx >= len(lines): raise ValueError("Missing horizontal angles")
        h_angles_list = []
        while len(h_angles_list) < num_h_angles:
             if line_idx >= len(lines): raise ValueError(f"File ended while reading horizontal angles (expected {num_h_angles})")
             h_angles_list.extend([float(a) for a in lines[line_idx].split()])
             line_idx += 1
        horizontal_angles = np.array(h_angles_list, dtype=np.float64)
        if len(horizontal_angles) != num_h_angles:
            raise ValueError(f"Mismatch in horizontal angle count (Expected {num_h_angles}, Found {len(horizontal_angles)})")
        # Ensure sorted
        if not np.all(np.diff(horizontal_angles) >= 0):
             print("[IES - Full] Warning: Horizontal angles were not sorted, sorting them now.")
             horizontal_angles = np.sort(horizontal_angles)

        # --- Read Candela Values ---
        if line_idx >= len(lines): raise ValueError("Missing candela values")
        candela_list_flat = []
        while line_idx < len(lines):
            try:
                candela_list_flat.extend([float(c) for c in lines[line_idx].split()])
            except ValueError:
                 # Check if it's a keyword line to ignore
                 current_line_upper = lines[line_idx].upper()
                 if any(kw in current_line_upper for kw in ['[', ']', 'END', 'LABEL']):
                      print(f"[IES - Full] Info: Skipping potential keyword line: {lines[line_idx]}")
                      line_idx += 1
                      continue
                 else:
                    print(f"[IES - Full] Warning: Encountered non-numeric data in candela section (line {line_idx + 1}). Stopping candela read: {lines[line_idx]}")
                    break # Stop reading if we hit unexpected non-numeric data
            line_idx += 1


        expected_candela_count = num_v_angles * num_h_angles
        if len(candela_list_flat) != expected_candela_count:
            # Try to handle potential extra trailing data/keywords
            if len(candela_list_flat) > expected_candela_count:
                print(f"[IES - Full] Warning: Found {len(candela_list_flat)} candela values, expected {expected_candela_count}. Truncating extra values.")
                candela_list_flat = candela_list_flat[:expected_candela_count]
            else:
                # If too few, it might be a file error or parsing stopped early.
                raise ValueError(f"Candela value count mismatch: Found {len(candela_list_flat)}, expected {expected_candela_count}. Check IES file format/content.")

        # Reshape based on IES standard order (V changes fastest for each H block)
        # Values are ordered: v0h0, v1h0, ..., vNh0, v0h1, v1h1, ..., vNh1, ...
        # So reshape directly into (num_h, num_v) then transpose for (num_v, num_h)
        candela_data_raw = np.array(candela_list_flat, dtype=np.float64)
        # Ensure candela values are non-negative
        if np.any(candela_data_raw < 0):
            print(f"[IES - Full] Warning: Negative candela values found ({np.min(candela_data_raw):.2f}). Clamping to 0.")
            candela_data_raw = np.maximum(candela_data_raw, 0.0)

        candela_data_2d = candela_data_raw.reshape((num_h_angles, num_v_angles)).T # Transpose needed -> shape (num_v, num_h)

        # Use lumens/lamp from header for normalization factor
        # The standard interpretation is that lumens_per_lamp is the reference base,
        # and the multiplier/ballast factor scale the output relative to that base.
        ies_file_lumens_norm = lumens_per_lamp * num_lamps * multiplier * ballast_factor
        if ies_file_lumens_norm <= 1e-6:
             print("[IES - Full] Warning: Calculated normalization lumens are near zero. Using 1.0 to avoid division errors.")
             ies_file_lumens_norm = 1.0


        print(f"[IES - Full] Successfully parsed data. Type: {photometric_type}")
        print(f"[IES - Full] Norm Lumens (calc): {ies_file_lumens_norm:.2f}, Input Watts: {input_watts:.2f}")
        print(f"[IES - Full] V Angles ({num_v_angles}): {vertical_angles[0]:.1f} to {vertical_angles[-1]:.1f}")
        print(f"[IES - Full] H Angles ({num_h_angles}): {horizontal_angles[0]:.1f} to {horizontal_angles[-1]:.1f}")
        print(f"[IES - Full] Candela Grid Shape: {candela_data_2d.shape}")

        # Return all necessary data arrays
        return vertical_angles, horizontal_angles, candela_data_2d, ies_file_lumens_norm

    except IndexError as e:
        print(f"[IES - Full] Error: File ended unexpectedly or list index out of range near line {line_idx+1}. {e}")
        return None, None, None, None
    except ValueError as ve:
        print(f"[IES - Full] Error: Could not convert data or count mismatch near line {line_idx+1}: {ve}")
        return None, None, None, None
    except Exception as e:
        print(f"[IES - Full] Unexpected error during parsing near line {line_idx+1}: {e}")
        import traceback # Optional: for more detailed debugging during development
        traceback.print_exc()
        return None, None, None, None

# --- Load STRIP IES Data ---
print("\n--- Loading Strip IES Data ---")
(STRIP_IES_V_ANGLES, STRIP_IES_H_ANGLES,
 STRIP_IES_CANDELAS_2D, STRIP_IES_FILE_LUMENS_NORM) = parse_ies_file_full(STRIP_IES_FILE)
if STRIP_IES_V_ANGLES is None:
     raise SystemExit("Failed to load or process strip IES file. Exiting.")

# --- Load COB IES Data ---
print("\n--- Loading COB IES Data ---")
(COB_IES_V_ANGLES, COB_IES_H_ANGLES,
 COB_IES_CANDELAS_2D, COB_IES_FILE_LUMENS_NORM) = parse_ies_file_full(COB_IES_FILE)
if COB_IES_V_ANGLES is None:
     raise SystemExit("Failed to load or process COB IES file. Exiting.")


# ------------------------------------
# 3) Compute SPD-based µmol/J Factor (No Changes Needed)
# ------------------------------------
def compute_conversion_factor(spd_file):
    try:
        # Load data, skipping header
        raw_spd = np.loadtxt(spd_file, delimiter=' ', skiprows=1)

        # Sort by wavelength and handle duplicates (average intensity)
        unique_wl, indices, counts = np.unique(raw_spd[:, 0], return_inverse=True, return_counts=True)
        # Ensure counts are not zero before dividing
        counts_nonzero = np.maximum(counts, 1) # Avoid division by zero if a wavelength appears 0 times (unlikely)
        avg_intens = np.bincount(indices, weights=raw_spd[:, 1]) / counts_nonzero
        spd = np.column_stack((unique_wl, avg_intens))

    except Exception as e:
        print(f"Error loading/processing SPD data from {spd_file}: {e}")
        print("Using default conversion factor.")
        return 0.0138 # Fallback value

    wl = spd[:, 0]
    intens = spd[:, 1]

    # Ensure wavelength is sorted for integration
    sort_idx = np.argsort(wl)
    wl = wl[sort_idx]
    intens = intens[sort_idx]

    # Calculate PAR fraction
    mask_par = (wl >= 400) & (wl <= 700)
    # Ensure we have points for integration
    if len(wl) < 2:
         print("[SPD Warning] Not enough data points for integration. Assuming PAR fraction = 1.0")
         PAR_fraction = 1.0
         tot = 1.0 # Avoid division by zero later
    else:
        tot = np.trapz(intens, wl)
        if tot <= 1e-9:
             print("[SPD Warning] Total integrated intensity is near zero. Assuming PAR fraction = 1.0")
             PAR_fraction = 1.0
        else:
             # Ensure PAR range has points for integration
             if np.count_nonzero(mask_par) < 2:
                  print("[SPD Warning] Not enough PAR data points for integration. Assuming PAR fraction = 1.0")
                  tot_par = tot
             else:
                  tot_par = np.trapz(intens[mask_par], wl[mask_par])
             PAR_fraction = tot_par / tot

    # Calculate conversion factor using effective photon energy in PAR range
    wl_m = wl * 1e-9
    h, c, N_A = 6.626e-34, 3.0e8, 6.022e23 # Planck, Speed of Light, Avogadro

    # Check if there are PAR points for weighted average
    if np.count_nonzero(mask_par) < 2:
         print("[SPD Warning] Not enough PAR data points for effective wavelength calc. Using simple average or fallback.")
         # Use simple average wavelength in PAR range if possible, else use midpoint
         if np.count_nonzero(mask_par) > 0:
             lambda_eff = np.mean(wl_m[mask_par])
         else:
             lambda_eff = (400e-9 + 700e-9) / 2.0 # Fallback: midpoint of PAR
    else:
        # Weighted average wavelength in PAR range
        numerator = np.trapz(wl_m[mask_par] * intens[mask_par], wl_m[mask_par])
        denominator = np.trapz(intens[mask_par], wl_m[mask_par])
        # Prevent division by zero if PAR intensity is zero
        lambda_eff = numerator / denominator if denominator > 1e-15 else 0.0

    # Energy per photon at effective wavelength
    E_photon = (h * c / lambda_eff) if lambda_eff > 1e-15 else 1.0 # Avoid division by zero

    # µmol/J = (photons/J) * (mol/photons) * (µmol/mol) * PAR_fraction
    conversion_factor = (1.0 / E_photon) * (1e6 / N_A) * PAR_fraction

    print(f"[INFO] SPD: Processed {len(wl)} unique points. PAR fraction={PAR_fraction:.3f}, conv_factor={conversion_factor:.5f} µmol/J")
    return conversion_factor

CONVERSION_FACTOR = compute_conversion_factor(SPD_FILE)


# ------------------------------------
# 4.6) Numba-Compatible 2D Intensity Functions (NEW/REVISED)
# ------------------------------------

@njit(cache=True)
def _interp_1d_linear_safe(x, xp, fp):
    """Safely interpolates, handling boundary conditions by clamping."""
    # Find insertion point
    idx = np.searchsorted(xp, x, side='right')

    # Handle boundary conditions
    if idx == 0:
        return fp[0]
    if idx == len(xp):
        return fp[-1]

    # Linear interpolation
    x0, x1 = xp[idx-1], xp[idx]
    y0, y1 = fp[idx-1], fp[idx]

    delta_x = x1 - x0
    if delta_x <= 1e-9: # Avoid division by zero if points are identical
        return y0 # or (y0+y1)/2 ? Sticking with y0.

    weight = (x - x0) / delta_x
    return y0 * (1.0 - weight) + y1 * weight

@njit(cache=True)
def interpolate_2d_bilinear(angle_v_deg, angle_h_deg,
                            ies_v_angles, ies_h_angles, ies_candelas_2d):
    """
    Performs bilinear interpolation on the 2D candela grid.
    Handles angle wrapping for horizontal angle (phi) assuming 0-360 range.
    Handles boundary conditions by clamping via safe 1D interpolation.
    Assumes ies_v_angles and ies_h_angles are sorted.
    """
    num_v, num_h = ies_candelas_2d.shape

    # --- Handle single point cases ---
    if num_v == 1 and num_h == 1:
        return ies_candelas_2d[0, 0]
    if num_v == 1: # Interpolate only horizontally
        return _interp_1d_linear_safe(angle_h_deg, ies_h_angles, ies_candelas_2d[0, :])
    if num_h == 1: # Interpolate only vertically
        return _interp_1d_linear_safe(angle_v_deg, ies_v_angles, ies_candelas_2d[:, 0])

    # --- Vertical Angle (Theta) ---
    # Find indices using searchsorted (more robust than manual loop)
    iv = np.searchsorted(ies_v_angles, angle_v_deg, side='right') - 1
    # Clamp indices to valid range [0, num_v - 2] for accessing iv and iv+1
    iv = max(0, min(iv, num_v - 2))

    # Get bracketing angles and calculate vertical weight tv
    v0, v1 = ies_v_angles[iv], ies_v_angles[iv+1]
    delta_v = v1 - v0
    if delta_v <= 1e-9: # Avoid division by zero if angles are identical
        tv = 0.0 if angle_v_deg <= v0 else 1.0
    else:
        tv = (angle_v_deg - v0) / delta_v
    tv = max(0.0, min(tv, 1.0)) # Clamp weight

    # --- Horizontal Angle (Phi) ---
    # Wrap input angle to match the IES horizontal angle range (e.g., 0-360)
    h_min, h_max = ies_h_angles[0], ies_h_angles[-1]
    angle_h_wrapped = angle_h_deg

    # Determine range and wrap logic
    h_range = h_max - h_min
    is_full_360 = abs(h_range - 360.0) < 5.0 # Check if range is close to 360
    is_symmetric_180 = abs(h_max - 180.0) < 5.0 and abs(h_min - 0.0) < 1e-6 # Check for 0-180
    is_symmetric_90 = abs(h_max - 90.0) < 5.0 and abs(h_min - 0.0) < 1e-6 # Check for 0-90

    if is_full_360:
         # Wrap angle to [h_min, h_max + epsilon] range, effectively 0-360
         angle_h_wrapped = h_min + ((angle_h_deg - h_min) % 360.0)
         # Handle result being exactly 360 when max angle is slightly less
         if abs(angle_h_wrapped - (h_min + 360.0)) < 1e-6:
             angle_h_wrapped = h_max
    elif is_symmetric_180:
         # Map angle to 0-180 range using symmetry
         angle_h_wrapped = angle_h_deg % 360.0
         if angle_h_wrapped > 180.0:
             angle_h_wrapped = 360.0 - angle_h_wrapped # Reflect back
    elif is_symmetric_90:
         # Map angle to 0-90 using quadrant symmetry
         angle_h_wrapped = angle_h_deg % 360.0
         if angle_h_wrapped > 270.0:
             angle_h_wrapped = 360.0 - angle_h_wrapped # Q4 -> Q1
         elif angle_h_wrapped > 180.0:
             angle_h_wrapped = angle_h_wrapped - 180.0 # Q3 -> Q1 (needs horizontal flip too, handled by interp?)
         elif angle_h_wrapped > 90.0:
             angle_h_wrapped = 180.0 - angle_h_wrapped # Q2 -> Q1
         # Ensure angle_h_wrapped is within [0, 90] after mapping
         angle_h_wrapped = max(0.0, min(angle_h_wrapped, 90.0))
    # Else: Assume contiguous range, no special wrapping (less common for Type C)
    # No change needed for angle_h_wrapped if not detected 360/180/90

    # Find indices using searchsorted
    ih = np.searchsorted(ies_h_angles, angle_h_wrapped, side='right') - 1
    # Clamp indices to valid range [0, num_h - 2] for accessing ih and ih+1
    ih = max(0, min(ih, num_h - 2))

    # Get bracketing angles and calculate horizontal weight th
    h0, h1 = ies_h_angles[ih], ies_h_angles[ih+1]
    delta_h = h1 - h0

    # Special handling if interpolation needs to wrap around 0/360 boundary
    needs_h_wrap_interp = is_full_360 and (angle_h_wrapped > h_max or angle_h_wrapped < h_min) and num_h > 1
    # This condition is tricky with searchsorted. Simpler: check if ih=num_h-2 and h1 is 'far' from h0.
    # Let's rely on clamping for now. A robust wrap requires interpolating between ih=num_h-1 and ih=0.

    if delta_h <= 1e-9: # Avoid division by zero
         th = 0.0 if angle_h_wrapped <= h0 else 1.0
    else:
        th = (angle_h_wrapped - h0) / delta_h
    th = max(0.0, min(th, 1.0)) # Clamp weight


    # --- Bilinear Interpolation ---
    # Get the 4 corner candela values using the clamped indices
    # Accessing ih+1 is safe because ih is clamped to num_h-2
    C00 = ies_candelas_2d[iv, ih]
    C10 = ies_candelas_2d[iv+1, ih]
    C01 = ies_candelas_2d[iv, ih+1]
    C11 = ies_candelas_2d[iv+1, ih+1]

    # Interpolate vertically first (along columns for a given H)
    C_h0 = C00 * (1.0 - tv) + C10 * tv
    C_h1 = C01 * (1.0 - tv) + C11 * tv

    # Interpolate horizontally second (between the results at h0 and h1)
    candela_raw = C_h0 * (1.0 - th) + C_h1 * th

    return max(0.0, candela_raw) # Ensure non-negative


@njit(cache=True)
def calculate_ies_intensity_2d(dx, dy, dz, dist, total_emitter_lumens,
                               ies_v_angles, ies_h_angles, ies_candelas_2d, ies_file_lumens_norm):
    """
    Calculates luminous intensity (candela) using full 2D IES data.
    Assumes standard IES Type C orientation (Nadir=0 deg V, 0 deg H along +X).
    Scales output based on the emitter's actual total lumens.

    Args:
        dx, dy, dz: Components of vector from source to target.
        dist: Magnitude of the vector (source to target distance).
        total_emitter_lumens: Actual lumens of the specific light source instance.
        ies_v_angles: Array of vertical angles (degrees, 0=down/nadir).
        ies_h_angles: Array of horizontal angles (degrees, 0-360 or symmetric range).
        ies_candelas_2d: 2D array of candela values (shape V x H).
        ies_file_lumens_norm: Normalization lumens from the IES file header.

    Returns:
        Scaled luminous intensity (candela) in the target direction.
    """
    epsilon = 1e-9
    if dist < epsilon:
        # If distance is tiny, direction is undefined. Use intensity at nadir (0,0)?
        # Need to find index for V=0. Assume ies_v_angles[0] is 0.
        # Need H=0? Assume ies_h_angles[0] is 0.
        # Fallback: return 0 or average intensity? Using 0 is safer.
        # More robust: Check if 0,0 exists and return that candela value.
        zero_v_idx = np.searchsorted(ies_v_angles, 0.0)
        zero_h_idx = np.searchsorted(ies_h_angles, 0.0)
        if zero_v_idx < len(ies_v_angles) and abs(ies_v_angles[zero_v_idx]) < epsilon \
           and zero_h_idx < len(ies_h_angles) and abs(ies_h_angles[zero_h_idx]) < epsilon:
           candela_raw = ies_candelas_2d[zero_v_idx, zero_h_idx]
        else:
           # If 0,0 not exactly present, maybe use first value? Or 0.
           candela_raw = ies_candelas_2d[0,0] # Use top-left corner as fallback

    else:
        # Calculate Vertical Angle (Theta) from Nadir (-Z axis)
        # cos(theta) = dot(V, Nadir) / |V| = dot((dx,dy,dz), (0,0,-1)) / dist = -dz / dist
        cos_theta_nadir = -dz / dist
        # Clamp due to potential floating point errors before acos
        cos_theta_nadir = max(-1.0, min(1.0, cos_theta_nadir))
        angle_v_deg = math.degrees(math.acos(cos_theta_nadir))

        # Calculate Horizontal Angle (Phi) in XY plane
        # atan2(Y, X) -> gives angle from +X axis. Matches IES Type C 0-deg H convention.
        angle_h_deg = math.degrees(math.atan2(dy, dx))
        # Wrap/adjust angle based on IES horizontal angle range (handled in interpolate_2d)
        # We just pass the raw 0-360 from atan2 (+ adjustment)
        angle_h_deg = (angle_h_deg + 360.0) % 360.0

        # Interpolate using the 2D function
        candela_raw = interpolate_2d_bilinear(angle_v_deg, angle_h_deg,
                                              ies_v_angles, ies_h_angles, ies_candelas_2d)

    # Scale by lumen ratio
    # Ensure norm_factor is not zero
    norm_factor = ies_file_lumens_norm if ies_file_lumens_norm > epsilon else 1.0
    scaling_factor = total_emitter_lumens / norm_factor
    return candela_raw * scaling_factor


# ------------------------------------
# 5) Geometry Building (No changes needed to the functions themselves)
# ------------------------------------
# Functions _get_cob_abstract_coords_and_transform, _apply_transform,
# cached_build_floor_grid, build_floor_grid, cached_build_patches, build_patches,
# build_all_light_sources, prepare_geometry remain IDENTICAL to version 1.0.
# (Copied here for file completeness)

@lru_cache(maxsize=32)
def cached_build_floor_grid(W: float, L: float):
    num_cells_x = int(round(W / FLOOR_GRID_RES))
    num_cells_y = int(round(L / FLOOR_GRID_RES))
    actual_res_x = W / num_cells_x if num_cells_x > 0 else W
    actual_res_y = L / num_cells_y if num_cells_y > 0 else L
    # Ensure at least one grid point
    num_cells_x = max(1, num_cells_x)
    num_cells_y = max(1, num_cells_y)
    xs = np.linspace(actual_res_x / 2.0, W - actual_res_x / 2.0, num_cells_x)
    ys = np.linspace(actual_res_y / 2.0, L - actual_res_y / 2.0, num_cells_y)
    # Ensure arrays are not empty if W/L or RES are very small
    if xs.size == 0: xs = np.array([W/2.0])
    if ys.size == 0: ys = np.array([L/2.0])
    X, Y = np.meshgrid(xs, ys)
    print(f"[Grid] Centered grid created: {X.shape[1]}x{X.shape[0]} points.") # Note shape order
    return X, Y

def build_floor_grid(W, L):
    return cached_build_floor_grid(W, L)

@lru_cache(maxsize=32) # Cache patch generation based on W, L, H
def cached_build_patches(W: float, L: float, H: float):
    patch_centers = []
    patch_areas = []
    patch_normals = []
    patch_refl = []

    # Floor (single patch) - Area should be > 0
    floor_area = W * L
    if floor_area > 1e-9:
        patch_centers.append((W/2, L/2, 0.0))
        patch_areas.append(floor_area)
        patch_normals.append((0.0, 0.0, 1.0)) # Normal pointing up (into room)
        patch_refl.append(REFL_FLOOR)
    else:
        print("[Warning] Floor area is zero or negative.")


    # Ceiling Patches
    xs_ceiling = np.linspace(0, W, CEIL_SUBDIVS_X + 1)
    ys_ceiling = np.linspace(0, L, CEIL_SUBDIVS_Y + 1)
    for i in range(CEIL_SUBDIVS_X):
        for j in range(CEIL_SUBDIVS_Y):
            cx = (xs_ceiling[i] + xs_ceiling[i+1]) / 2
            cy = (ys_ceiling[j] + ys_ceiling[j+1]) / 2
            area = (xs_ceiling[i+1] - xs_ceiling[i]) * (ys_ceiling[j+1] - ys_ceiling[j])
            if area > 1e-9: # Ensure patch has positive area
                patch_centers.append((cx, cy, H)) # Ceiling at height H
                patch_areas.append(area)
                patch_normals.append((0.0, 0.0, -1.0)) # Normal pointing down (into room)
                patch_refl.append(REFL_CEIL)

    # Wall Patches - Define parameters for each wall
    wall_params = [
        # (axis_coord, fixed_val, normal_vec, iter_axis_1_range, iter_axis_2_range, iter_axis_1_subdivs, iter_axis_2_subdivs)
        ('y', 0.0, (0.0, 1.0, 0.0), (0, W), (0, H), WALL_SUBDIVS_X, WALL_SUBDIVS_Y),  # Wall at y=0 (Normal +Y)
        ('y', L,   (0.0,-1.0, 0.0), (0, W), (0, H), WALL_SUBDIVS_X, WALL_SUBDIVS_Y),  # Wall at y=L (Normal -Y)
        ('x', 0.0, (1.0, 0.0, 0.0), (0, L), (0, H), WALL_SUBDIVS_X, WALL_SUBDIVS_Y),  # Wall at x=0 (Normal +X) - Assuming subdivs_x applies to L here
        ('x', W,  (-1.0, 0.0, 0.0), (0, L), (0, H), WALL_SUBDIVS_X, WALL_SUBDIVS_Y),  # Wall at x=W (Normal -X) - Assuming subdivs_x applies to L here
    ]

    for axis, fixed_val, normal, range1, range2, subdivs1, subdivs2 in wall_params:
        # Ensure subdivision counts are positive
        subdivs1 = max(1, subdivs1)
        subdivs2 = max(1, subdivs2)
        coords1 = np.linspace(range1[0], range1[1], subdivs1 + 1)
        coords2 = np.linspace(range2[0], range2[1], subdivs2 + 1) # Always Z axis (height)

        for i in range(subdivs1):
            for j in range(subdivs2):
                c1 = (coords1[i] + coords1[i+1]) / 2
                c2 = (coords2[j] + coords2[j+1]) / 2 # This is cz
                area = (coords1[i+1] - coords1[i]) * (coords2[j+1] - coords2[j])

                if area > 1e-9: # Ensure patch has positive area
                    if axis == 'y':
                        center = (c1, fixed_val, c2) # c1 is cx
                    else: # axis == 'x'
                        center = (fixed_val, c1, c2) # c1 is cy

                    patch_centers.append(center)
                    patch_areas.append(area)
                    patch_normals.append(normal) # Use the CORRECTED inward-pointing normal
                    patch_refl.append(REFL_WALL)

    if not patch_centers:
         print("[ERROR] No patches were generated. Check room dimensions and subdivisions.")
         # Return empty arrays to avoid errors later
         return (np.empty((0,3), dtype=np.float64), np.empty(0, dtype=np.float64),
                 np.empty((0,3), dtype=np.float64), np.empty(0, dtype=np.float64))

    return (np.array(patch_centers, dtype=np.float64),
            np.array(patch_areas, dtype=np.float64),
            np.array(patch_normals, dtype=np.float64),
            np.array(patch_refl, dtype=np.float64))

def build_patches(W, L, H):
    return cached_build_patches(W, L, H)

def _get_cob_abstract_coords_and_transform(W, L, H, num_total_layers):
    n = num_total_layers - 1
    abstract_positions = []
    # Abstract diamond coordinates (only store perimeter points needed for strips later)
    abstract_positions.append({'x': 0, 'y': 0, 'h': H, 'layer': 0, 'is_vertex': True}) # Center COB
    for i in range(1, n + 1):
        # Generate abstract coordinates in clockwise order for perimeter
        # Top-Right to Top-Left
        for x in range(i, 0, -1): abstract_positions.append({'x': x, 'y': i - x, 'h': H, 'layer': i, 'is_vertex': (x == i or x == 0)})
        # Top-Left to Bottom-Left
        for x in range(0, -i, -1): abstract_positions.append({'x': x, 'y': i + x, 'h': H, 'layer': i, 'is_vertex': (x == -i or x == 0)})
        # Bottom-Left to Bottom-Right
        for x in range(-i, 0, 1): abstract_positions.append({'x': x, 'y': -i - x, 'h': H, 'layer': i, 'is_vertex': (x == -i or x == 0)})
        # Bottom-Right to Top-Right
        for x in range(0, i + 1, 1): abstract_positions.append({'x': x, 'y': -i + x, 'h': H, 'layer': i, 'is_vertex': (x == i or x == 0)}) # Include closing vertex x=i

    # Remove duplicates (especially the corners which are generated twice)
    unique_positions = []
    seen_coords = set()
    for pos in abstract_positions:
        coord_tuple = (pos['x'], pos['y'], pos['layer'])
        if coord_tuple not in seen_coords:
            unique_positions.append(pos)
            seen_coords.add(coord_tuple)

    # Sort primarily by layer, then potentially by angle/position for consistency
    unique_positions.sort(key=lambda p: (p['layer'], math.atan2(p['y'], p['x']) if p['layer'] > 0 else 0))

    # Transformation parameters
    theta = math.radians(45)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    center_x, center_y = W / 2, L / 2
    # Scaling factor based on max layer index 'n' to reach near corners
    # Adjust scale slightly to prevent lights being exactly at W or L if needed
    scale_x = W / (n * math.sqrt(2)) if n > 0 else W / 2 # Avoid division by zero
    scale_y = L / (n * math.sqrt(2)) if n > 0 else L / 2 # Avoid division by zero
    # Ensure scale is positive
    scale_x = max(scale_x, 1e-6)
    scale_y = max(scale_y, 1e-6)


    transform_params = {
        'center_x': center_x, 'center_y': center_y,
        'scale_x': scale_x, 'scale_y': scale_y,
        'cos_t': cos_t, 'sin_t': sin_t,
        'H': H, 'Z_pos': H * 0.95 # Actual Z height for lights
    }
    return unique_positions, transform_params


def _apply_transform(abstract_pos, transform_params):
    """Applies rotation, scaling, translation to abstract coords."""
    ax, ay = abstract_pos['x'], abstract_pos['y']
    # Rotate
    rx = ax * transform_params['cos_t'] - ay * transform_params['sin_t']
    ry = ax * transform_params['sin_t'] + ay * transform_params['cos_t']
    # Scale and Translate
    px = transform_params['center_x'] + rx * transform_params['scale_x']
    py = transform_params['center_y'] + ry * transform_params['scale_y']
    pz = transform_params['Z_pos']
    return px, py, pz

def build_all_light_sources(W, L, H, cob_lumen_params, strip_module_lumen_params):
    """
    Generates positions, lumens, and types for all COBs and discrete LED strip MODULES.
    Uses cob_lumen_params for COB brightness per layer.
    Uses strip_module_lumen_params for Strip module brightness per layer.
    Assumes STRIP_MODULE_LENGTH is defined globally/accessibly.
    """
    num_total_layers_cob = len(cob_lumen_params)
    num_total_layers_strip = len(strip_module_lumen_params)
    if num_total_layers_cob != num_total_layers_strip:
         raise ValueError(f"Length mismatch between cob_lumen_params ({num_total_layers_cob}) and strip_module_lumen_params ({num_total_layers_strip})")
    num_total_layers = num_total_layers_cob # Use consistent layer count

    if num_total_layers == 0 : return np.empty((0,3)), np.empty((0,)), np.empty((0,)), np.empty((0,4)), {}

    abstract_cob_coords, transform_params = _get_cob_abstract_coords_and_transform(W, L, H, num_total_layers)

    all_positions_list = []
    all_lumens_list = []
    all_types_list = []
    cob_positions_only_list = []
    ordered_strip_vertices = {}

    # Process COBs (uses cob_lumen_params)
    for i, abs_cob in enumerate(abstract_cob_coords):
        layer = abs_cob['layer']
        # Ensure layer index is valid for lumen params
        if not (0 <= layer < num_total_layers):
             print(f"[Warning] Abstract COB coord has invalid layer {layer}. Skipping.")
             continue
        px, py, pz = _apply_transform(abs_cob, transform_params)
        lumens = cob_lumen_params[layer]
        if lumens > 1e-9: # Only add light source if it has significant lumens
            all_positions_list.append([px, py, pz])
            all_lumens_list.append(lumens)
            all_types_list.append(EMITTER_TYPE_COB)
            cob_positions_only_list.append([px, py, pz, layer])
        # else: print(f"[Debug] Skipping COB layer {layer} due to zero lumens.")


    # Process Strips (Layers 1 onwards) using strip_module_lumen_params
    print("[Geometry] Placing strip modules...")
    total_modules_placed = 0
    for layer_idx in range(1, num_total_layers):
        # Get the target lumens for modules in this specific layer
        target_module_lumens = strip_module_lumen_params[layer_idx]
        if target_module_lumens <= 1e-9: # Check target lumens ## MODIFY CHECK ##
             # print(f"[INFO] Layer {layer_idx}: Skipping modules due to zero/low target lumens.") # Reduce noise
             continue

        # Find vertices for this layer
        layer_vertices_abstract = [p for p in abstract_cob_coords if p['layer'] == layer_idx and p['is_vertex']]
        if not layer_vertices_abstract: continue # Skip if no vertices found for strip placement

        # Sort vertices by angle for consistent line segment generation
        layer_vertices_abstract.sort(key=lambda p: math.atan2(p['y'], p['x']))
        transformed_vertices = [_apply_transform(av, transform_params) for av in layer_vertices_abstract]
        ordered_strip_vertices[layer_idx] = transformed_vertices
        num_vertices = len(transformed_vertices)
        modules_this_layer = 0


        # Iterate through sections connecting the main COB vertices
        for i in range(num_vertices):
            p1 = np.array(transformed_vertices[i])
            p2 = np.array(transformed_vertices[(i + 1) % num_vertices]) # Wrap around
            direction_vec = p2 - p1
            section_length = np.linalg.norm(direction_vec)

            # Check if section is long enough for at least one module (with some buffer)
            if section_length < STRIP_MODULE_LENGTH * 0.9: continue

            direction_unit = direction_vec / section_length
            # Calculate how many modules fit
            num_modules = int(math.floor(section_length / STRIP_MODULE_LENGTH))
            if num_modules == 0: continue

            # Distribute modules evenly along the section
            total_gap_length = section_length - num_modules * STRIP_MODULE_LENGTH
            gap_length = total_gap_length / (num_modules + 1) # Gaps at start, end, and between modules

            for j in range(num_modules):
                # Position of the center of the j-th module
                dist_to_module_center = gap_length * (j + 1) + STRIP_MODULE_LENGTH * (j + 0.5)
                module_pos = p1 + direction_unit * dist_to_module_center
                all_positions_list.append(module_pos.tolist())
                # Assign the TARGET lumens for this layer's modules
                all_lumens_list.append(target_module_lumens)
                all_types_list.append(EMITTER_TYPE_STRIP)
                modules_this_layer += 1

        if modules_this_layer > 0:
             print(f"[INFO] Layer {layer_idx}: Placed {modules_this_layer} strip modules (Target Lumens={target_module_lumens:.1f}).") # Updated print
             total_modules_placed += modules_this_layer

    # Convert lists to NumPy arrays - check and error handling remains the same) ...
    try:
        # Ensure lists are not empty before converting
        if not all_positions_list:
            print("[Warning] No light sources generated with positive lumens.")
            light_positions = np.empty((0, 3), dtype=np.float64)
            light_lumens = np.empty(0, dtype=np.float64)
            light_types = np.empty(0, dtype=np.int32)
        else:
            light_positions = np.array(all_positions_list, dtype=np.float64)
            if not all(isinstance(l, (int, float)) for l in all_lumens_list):
                 print("[ERROR] Invalid data found in lumen list:", [type(l) for l in all_lumens_list if not isinstance(l, (int, float))])
                 raise TypeError("Lumen list contains non-numeric data")
            light_lumens = np.array(all_lumens_list, dtype=np.float64)
            light_types = np.array(all_types_list, dtype=np.int32)

        # Handle potentially empty cob_positions_only_list
        if not cob_positions_only_list:
            cob_positions_only = np.empty((0, 4), dtype=np.float64)
        else:
            cob_positions_only = np.array(cob_positions_only_list, dtype=np.float64)

    except Exception as e:
         print(f"[ERROR] Failed to convert geometry lists to NumPy arrays: {e}")
         raise

    # Use shape[0] for counts after conversion
    num_cobs_placed = cob_positions_only.shape[0]
    num_strips_placed = total_modules_placed # From counter during placement

    print(f"[INFO] Generated {num_cobs_placed} main COBs with positive lumens.")
    if num_strips_placed > 0:
         print(f"[INFO] Placed {num_strips_placed} strip modules across all layers.")

    return light_positions, light_lumens, light_types, cob_positions_only, ordered_strip_vertices

def prepare_geometry(W, L, H, cob_lumen_params, strip_module_lumen_params):
    """Prepares all geometry: light sources, floor grid, patches, strip vertices."""
    print("[Geometry] Building light sources (COBs + Strip Modules)...")
    # Pass both lumen parameter arrays to build_all_light_sources ## FIX CALL ##
    light_positions, light_lumens, light_types, cob_positions_only, ordered_strip_vertices = build_all_light_sources(
        W, L, H, cob_lumen_params, strip_module_lumen_params # Pass both here
    )
    print("[Geometry] Building floor grid...")
    X, Y = build_floor_grid(W, L)
    print("[Geometry] Building room patches (floor, ceiling, walls)...")
    patches = build_patches(W, L, H)
    return light_positions, light_lumens, light_types, cob_positions_only, ordered_strip_vertices, X, Y, patches


# ------------------------------------
# 6) Numba-JIT Computations (REVISED Kernels)
# ------------------------------------

@njit(parallel=True, cache=True)
def compute_direct_floor(light_positions, light_lumens, light_types,
                         X, Y,
                         # Pass full IES data for both types
                         cob_v_angles, cob_h_angles, cob_candelas_2d, cob_norm,
                         strip_v_angles, strip_h_angles, strip_candelas_2d, strip_norm
                         ):
    """ Computes direct illuminance on the floor grid using full 2D IES data. """
    min_dist2 = (FLOOR_GRID_RES / 4.0) ** 2 # Smaller minimum distance squared
    epsilon = 1e-9
    rows, cols = X.shape
    out = np.zeros_like(X, dtype=np.float64)
    num_lights = light_positions.shape[0]

    # Check if IES data arrays are valid (basic shape check)
    cob_ies_valid = cob_candelas_2d.ndim == 2 and cob_candelas_2d.size > 0
    strip_ies_valid = strip_candelas_2d.ndim == 2 and strip_candelas_2d.size > 0

    for r in prange(rows):
        for c in range(cols):
            fx, fy = X[r, c], Y[r, c] # Floor point (target)
            fz = 0.0 # Floor Z coordinate
            val = 0.0
            for k in range(num_lights):
                # Source light properties
                lx, ly, lz = light_positions[k, 0], light_positions[k, 1], light_positions[k, 2]
                lumens_k = light_lumens[k]
                type_k = light_types[k]

                # Optimization: Skip if lumens are negligible
                if lumens_k < epsilon: continue

                # Vector from source (l) to target (f)
                dx, dy, dz = fx - lx, fy - ly, fz - lz
                d2 = dx*dx + dy*dy + dz*dz
                d2 = max(d2, min_dist2) # Avoid singularity if source is very close/on floor point
                dist = math.sqrt(d2)

                # Cosine of angle with floor normal (0,0,1) -> incidence angle cosine
                # cos_in_floor = dot(Normal_floor, -V) / |V| = dot((0,0,1), (-dx,-dy,-dz)) / dist = -dz / dist
                cos_in_floor = -dz / dist
                # Check if light is below floor horizon or exactly horizontal
                if cos_in_floor < epsilon: continue
                cos_in_floor = min(cos_in_floor, 1.0) # Clamp (shouldn't exceed 1)


                # --- Calculate Intensity using 2D IES data ---
                I_val = 0.0
                # Select correct IES data based on type_k
                if type_k == EMITTER_TYPE_COB:
                    if cob_ies_valid:
                         I_val = calculate_ies_intensity_2d(dx, dy, dz, dist, lumens_k,
                                                            cob_v_angles, cob_h_angles, cob_candelas_2d, cob_norm)
                    # else: print warning handled implicitly by I_val = 0
                elif type_k == EMITTER_TYPE_STRIP:
                     if strip_ies_valid:
                         I_val = calculate_ies_intensity_2d(dx, dy, dz, dist, lumens_k,
                                                            strip_v_angles, strip_h_angles, strip_candelas_2d, strip_norm)
                    # else: print warning handled implicitly by I_val = 0
                # ---------------------------------------------

                # Calculate local illuminance E = (I / d^2) * cos(incidence_angle)
                # Only add if intensity is positive
                if I_val > epsilon:
                     E_local = (I_val / d2) * cos_in_floor
                     val += E_local
            out[r, c] = val
    return out


@njit(cache=True) # Only one loop over patches, parallel=True might add overhead
def compute_patch_direct(light_positions, light_lumens, light_types,
                         patch_centers, patch_normals, patch_areas,
                         # Pass full IES data for both types
                         cob_v_angles, cob_h_angles, cob_candelas_2d, cob_norm,
                         strip_v_angles, strip_h_angles, strip_candelas_2d, strip_norm
                         ):
    """ Computes direct illuminance on patch centers using full 2D IES data. """
    min_dist2 = (FLOOR_GRID_RES / 4.0) ** 2 # Use a small minimum distance squared
    epsilon = 1e-9
    Np = patch_centers.shape[0]
    num_lights = light_positions.shape[0]
    out = np.zeros(Np, dtype=np.float64)

    # Check if IES data arrays are valid
    cob_ies_valid = cob_candelas_2d.ndim == 2 and cob_candelas_2d.size > 0
    strip_ies_valid = strip_candelas_2d.ndim == 2 and strip_candelas_2d.size > 0

    for ip in range(Np): # Iterate through each patch (target)
        pc = patch_centers[ip] # Patch center (target)
        n_patch = patch_normals[ip] # Patch normal
        # Normalize patch normal once
        norm_n_patch_val = np.linalg.norm(n_patch)
        if norm_n_patch_val < epsilon: continue # Skip invalid patches (zero normal)
        n_patch_unit = n_patch / norm_n_patch_val

        accum_E = 0.0
        for k in range(num_lights): # Iterate through each light source
            # Source light properties
            lx, ly, lz = light_positions[k, 0], light_positions[k, 1], light_positions[k, 2]
            lumens_k = light_lumens[k]
            type_k = light_types[k]

            # Optimization: Skip if lumens are negligible
            if lumens_k < epsilon: continue

            # Vector from source (l) to target (pc)
            dx, dy, dz = pc[0] - lx, pc[1] - ly, pc[2] - lz
            d2 = dx*dx + dy*dy + dz*dz
            d2 = max(d2, min_dist2)
            dist = math.sqrt(d2)

            # Cosine of incidence angle at patch surface
            # Vector FROM light TO patch is V = (dx, dy, dz)
            # Incidence angle is between surface normal N and direction -V
            # cos_in_patch = dot(N_unit, -V_unit) = dot(N_unit, -V/dist)
            # V/dist = (dx/dist, dy/dist, dz/dist) = unit vector towards patch
            cos_in_patch = n_patch_unit[0]*(-dx/dist) + n_patch_unit[1]*(-dy/dist) + n_patch_unit[2]*(-dz/dist)

            # Check if light is behind patch or hitting edge-on
            if cos_in_patch < epsilon: continue
            cos_in_patch = min(cos_in_patch, 1.0) # Clamp


            # --- Calculate Intensity using 2D IES data ---
            I_val = 0.0
            if type_k == EMITTER_TYPE_COB:
                if cob_ies_valid:
                     I_val = calculate_ies_intensity_2d(dx, dy, dz, dist, lumens_k,
                                                        cob_v_angles, cob_h_angles, cob_candelas_2d, cob_norm)
            elif type_k == EMITTER_TYPE_STRIP:
                 if strip_ies_valid:
                     I_val = calculate_ies_intensity_2d(dx, dy, dz, dist, lumens_k,
                                                        strip_v_angles, strip_h_angles, strip_candelas_2d, strip_norm)
            # ---------------------------------------------

            # Calculate local illuminance E = (I / d^2) * cos(incidence_angle)
            if I_val > epsilon:
                E_local = (I_val / d2) * cos_in_patch
                accum_E += E_local
        out[ip] = accum_E
    return out


# --- Radiosity and Monte Carlo Reflection Functions (No Changes Needed structurally) ---
# iterative_radiosity_loop, compute_reflection_on_floor, compute_row_reflection
# remain IDENTICAL to version 1.0. Added minor epsilon checks.
# (Copied here for file completeness)

@njit(cache=True) # Radiosity loop
def iterative_radiosity_loop(patch_centers, patch_normals, patch_direct, patch_areas, patch_refl,
                             max_bounces, convergence_threshold):
    """ Calculates total incident irradiance on each patch via iterative radiosity. """
    Np = patch_direct.shape[0]
    if Np == 0: return np.empty(0, dtype=np.float64) # Handle no patches case

    # Initialize total incident irradiance with direct component
    patch_incident_total = patch_direct.copy()
    # Calculate initial exitance (radiosity B)
    patch_exitance = patch_incident_total * patch_refl
    epsilon = 1e-9 # Small number for checks

    for bounce in range(max_bounces):
        newly_incident_flux_indirect = np.zeros(Np, dtype=np.float64)

        # --- Flux Transfer Calculation ---
        # This part is computationally intensive. Parallelization is complex due to potential
        # race conditions updating newly_incident_flux_indirect.
        # Keeping it serial for simplicity and correctness.
        for j in range(Np): # Source patch j emitting flux
            # Calculate total flux leaving patch j (Area * Exitance)
            outgoing_flux_j = patch_areas[j] * patch_exitance[j]
            # Skip if patch emits negligible flux or has zero area/reflectance
            if outgoing_flux_j <= epsilon or patch_areas[j] <= epsilon: continue

            pj = patch_centers[j]; nj = patch_normals[j]
            # Normalize source normal once
            norm_nj_val = np.linalg.norm(nj)
            if norm_nj_val < epsilon: continue # Skip invalid source normal
            nj_unit = nj / norm_nj_val

            for i in range(Np): # Destination patch i receiving flux
                if i == j: continue # Patch doesn't transfer flux to itself directly
                if patch_areas[i] <= epsilon: continue # Skip if destination has zero area

                pi = patch_centers[i]; ni = patch_normals[i]
                 # Normalize destination normal once
                norm_ni_val = np.linalg.norm(ni)
                if norm_ni_val < epsilon: continue # Skip invalid destination normal
                ni_unit = ni / norm_ni_val

                # Vector from center of patch j to center of patch i
                vij = pi - pj
                dist2 = np.dot(vij, vij)
                # Avoid coincident patches or very close patches (use epsilon*epsilon)
                if dist2 < epsilon * epsilon: continue
                dist = math.sqrt(dist2)
                vij_unit = vij / dist

                # Cosines of angles relative to normals (using unit vectors)
                cos_j = np.dot(nj_unit, vij_unit)     # Angle at source patch j (emission)
                cos_i = np.dot(ni_unit, -vij_unit)    # Angle at destination patch i (incidence)

                # Check visibility and orientation: both cosines must be positive
                if cos_j <= epsilon or cos_i <= epsilon: continue

                # Form Factor approximation (point-to-point geometric term G_ji)
                # G_ji = (cos_j * cos_i) / (pi * dist^2)
                # Flux arriving at dA_i = B_j * dA_j * G_ji * dA_i / dA_j (?) No.
                # Differential Flux dPhi_{j->i} = L_j * cos_j * dAj * dOmega_{ji}
                # L_j = B_j / pi (for Lambertian)
                # dOmega_{ji} = cos_i * dAi / dist^2
                # dPhi_{j->i} = (B_j / pi) * cos_j * dAj * cos_i * dAi / dist^2
                # Total Phi_{j->i} = Integral over Aj, Ai [ (B_j / pi) * G_ji ] dAj dAi
                # Approximation: Phi_{j->i} approx B_j * Aj * F_{ji}
                # Where F_{ji} approx G_ji * Ai = (cos_j * cos_i * Ai) / (pi * dist^2)
                # Check units: B (W/m2), Aj (m2), Fji (dimensionless) -> Phi (W) - Correct.
                form_factor_approx_ji = (cos_j * cos_i * patch_areas[i]) / (math.pi * dist2)
                # Clamp form factor approximation to avoid issues, though should be positive
                form_factor_approx_ji = max(0.0, form_factor_approx_ji)

                flux_transfer = patch_exitance[j] * patch_areas[j] * form_factor_approx_ji # B_j * A_j * F_ji
                newly_incident_flux_indirect[i] += flux_transfer


        # --- Update Radiosity and Check Convergence ---
        max_rel_change = 0.0
        prev_patch_incident_total = patch_incident_total.copy() # Keep previous total for comparison

        for i in range(Np):
            # New indirect incident irradiance (flux density) on patch i
            # Protect against division by zero area
            incident_irradiance_indirect_i = newly_incident_flux_indirect[i] / patch_areas[i] if patch_areas[i] > epsilon else 0.0

            # Update total incident irradiance: Direct + New Indirect
            patch_incident_total[i] = patch_direct[i] + incident_irradiance_indirect_i

            # Update exitance for the *next* bounce calculation
            patch_exitance[i] = patch_incident_total[i] * patch_refl[i]

            # Check convergence based on relative change in *total incident* irradiance
            change = abs(patch_incident_total[i] - prev_patch_incident_total[i])
            # Use previous value as denominator base for relative change
            denom = abs(prev_patch_incident_total[i]) + epsilon # Add epsilon to avoid div by zero
            rel_change = change / denom
            if rel_change > max_rel_change:
                max_rel_change = rel_change

        # Check for convergence
        if max_rel_change < convergence_threshold:
            # print(f"Radiosity converged after {bounce+1} bounces. Max change: {max_rel_change:.2e}")
            break
    # else: # Optional: print if max bounces reached without converging
        # print(f"Radiosity loop reached max {max_bounces} bounces. Max change: {max_rel_change:.2e}")

    # Return the final total incident irradiance on each patch
    return patch_incident_total


@njit(cache=True) # Numba function for MC row calculation
def compute_row_reflection(r, X, Y, patch_centers, patch_normals, patch_areas, patch_exitance, mc_samples):
    """ Computes indirect illuminance for a single row of the floor grid using MC. """
    row_vals = np.empty(X.shape[1], dtype=np.float64)
    Np = patch_centers.shape[0]
    epsilon = 1e-9

    # Ensure mc_samples is positive
    valid_mc_samples = max(1, mc_samples)

    for c in range(X.shape[1]): # Iterate through columns (points in the row)
        fx, fy, fz = X[r, c], Y[r, c], 0.0 # Target floor point (z=0)
        total_indirect_E = 0.0

        for p in range(Np): # Iterate through all source patches
            exitance_p = patch_exitance[p]
            area_p = patch_areas[p]
            # Skip if patch contributes negligible flux or has zero area
            if exitance_p * area_p <= epsilon: continue

            pc = patch_centers[p]; n = patch_normals[p]
            # Normalize patch normal once
            norm_n_val = np.linalg.norm(n)
            if norm_n_val < epsilon: continue # Skip invalid patch normal
            n_unit = n / norm_n_val

            # --- Monte Carlo Sampling from Patch p to Floor Point (fx, fy) ---
            # Simplified patch sampling setup (assumes roughly square/compact)
            # Find tangent vectors robustly
            if abs(n_unit[0]) > 0.9: v_tmp = np.array((0.0, 1.0, 0.0))
            else: v_tmp = np.array((1.0, 0.0, 0.0))
            tangent1 = np.cross(n_unit, v_tmp); t1_norm = np.linalg.norm(tangent1)
            # Handle cases where tmp is parallel to normal
            if t1_norm < epsilon: tangent1 = np.cross(n_unit, np.array((0.0,0.0,1.0))); t1_norm = np.linalg.norm(tangent1)
            if t1_norm < epsilon: tangent1 = np.array((1.0,0.0,0.0)) # Failsafe if normal is Z-aligned
            else: tangent1 /= t1_norm
            # Create second tangent vector orthogonal to normal and tangent1
            tangent2 = np.cross(n_unit, tangent1) # Already normalized if n_unit and tangent1 are orthonormal
            # tangent2 /= np.linalg.norm(tangent2) # Re-normalize just in case

            # Approximate patch size for uniform random sampling offsets
            # More accuracy could come from knowing exact patch shape/dimensions
            half_side = math.sqrt(area_p) * 0.5

            sample_sum_integrand = 0.0 # Accumulator for the core term of the integral
            for _ in range(valid_mc_samples):
                # 1. Sample a random point on the patch surface (uniform sampling)
                # Improve this with importance sampling (e.g., cosine weighted) for efficiency
                off1 = np.random.uniform(-half_side, half_side)
                off2 = np.random.uniform(-half_side, half_side)
                sample_point = pc + off1 * tangent1 + off2 * tangent2

                # 2. Calculate geometry between sample_point and floor_point (fx, fy, 0)
                vec_sp_to_fp = np.array((fx - sample_point[0], fy - sample_point[1], fz - sample_point[2]))
                dist2 = np.dot(vec_sp_to_fp, vec_sp_to_fp)
                if dist2 < epsilon * epsilon: continue # Avoid singularity
                dist = math.sqrt(dist2)
                vec_sp_to_fp_unit = vec_sp_to_fp / dist

                # 3. Calculate cosines
                # Cosine at floor point (normal_floor = 0,0,1)
                # cos_f = dot(normal_floor, -vec_sp_to_fp_unit) = dot((0,0,1), -vec_sp_to_fp_unit)
                cos_f = -vec_sp_to_fp_unit[2]
                # Cosine at patch sample point (normal_patch = n_unit)
                # cos_p = dot(normal_patch, vec_sp_to_fp_unit)
                cos_p = np.dot(n_unit, vec_sp_to_fp_unit)

                # 4. Check visibility and orientation (both cosines must be positive)
                if cos_f > epsilon and cos_p > epsilon:
                     # Calculate the integrand term for this sample: (cos_p * cos_f) / dist^2
                     # This is part of the rendering equation (simplified for Lambertian)
                     # dE = L * cos_p * cos_f * dA / dist^2 = (B/pi) * cos_p * cos_f * dA / dist^2
                     # We sum the geometric term G = (cos_p * cos_f) / dist^2
                    integrand_term = (cos_p * cos_f) / dist2
                    sample_sum_integrand += integrand_term

            # 5. Estimate the integral using the MC average
            # E = Integral [ (B/pi) * G dA ] approx (Area / N) * Sum [ (B/pi) * G_sample ]
            # E = (Area_p * Exitance_p / pi) * (1 / N_samples) * Sum [ Integrand_term_sample ]
            avg_integrand = sample_sum_integrand / valid_mc_samples if valid_mc_samples > 0 else 0.0
            total_indirect_E += (area_p * exitance_p / math.pi) * avg_integrand

        row_vals[c] = total_indirect_E # Store calculated indirect illuminance for this point
    return row_vals

# Joblib wrapper for parallel row computation
def compute_reflection_on_floor(X, Y, patch_centers, patch_normals, patch_areas, patch_rad, patch_refl,
                                mc_samples=MC_SAMPLES):
    """ Calculates indirect illuminance on floor grid using MC via parallel rows. """
    rows, cols = X.shape
    if rows == 0 or cols == 0: return np.zeros((rows, cols), dtype=np.float64) # Handle empty grid

    # Calculate patch exitance B = Incident_Total * Reflectance
    patch_exitance = patch_rad * patch_refl

    # Parallel computation over rows of the floor grid
    results = Parallel(n_jobs=-1)(delayed(compute_row_reflection)(
        r, X, Y, patch_centers, patch_normals, patch_areas, patch_exitance, mc_samples
    ) for r in range(rows))

    # Assemble results into the output array
    out = np.zeros((rows, cols), dtype=np.float64)
    # Ensure results list is not empty and has correct structure
    if results and len(results) == rows:
        for r, row_vals in enumerate(results):
            if row_vals is not None and len(row_vals) == cols:
                 out[r, :] = row_vals
            else:
                 print(f"[Warning] Invalid results returned for row {r}. Setting to zero.")
    else:
         print("[Warning] Parallel computation for reflection returned unexpected results. Output might be incomplete.")

    return out


# ------------------------------------
# 7) Heatmap Plotting Function (No changes needed)
# ------------------------------------
# plot_heatmap function remains IDENTICAL to version 1.0.
# (Copied here for file completeness)
def plot_heatmap(floor_ppfd, X, Y, cob_marker_positions, ordered_strip_vertices, W, L, annotation_step=10):
    """ Plots the PPFD heatmap with light source overlays. """
    fig, ax = plt.subplots(figsize=(10, 8))
    extent = [0, W, 0, L]
    # Handle cases where ppfd is constant (vmin=vmax causes issues)
    vmin = np.min(floor_ppfd)
    vmax = np.max(floor_ppfd)
    if abs(vmin - vmax) < 1e-6: vmax = vmin + 1.0 # Add small range if all values are identical

    im = ax.imshow(floor_ppfd, cmap='viridis', interpolation='nearest', origin='lower', extent=extent, aspect='equal', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8); cbar.set_label('PPFD (µmol/m²/s)')
    # Annotations
    rows, cols = floor_ppfd.shape
    # Ensure X, Y shapes match floor_ppfd for annotation indexing
    if X.shape[0] != rows or X.shape[1] != cols or Y.shape[0] != rows or Y.shape[1] != cols:
         print("[Plot Warning] Mismatch between floor_ppfd and X/Y grid shapes. Skipping annotations.")
    elif annotation_step > 0 and rows > 0 and cols > 0:
        step_r = max(1, rows // annotation_step); step_c = max(1, cols // annotation_step)
        # Calculate cell size for centering text
        cell_w = (X[0,1] - X[0,0]) if cols > 1 else W
        cell_h = (Y[1,0] - Y[0,0]) if rows > 1 else L
        for r in range(0, rows, step_r):
             for c in range(0, cols, step_c):
                # Center text within the grid cell corresponding to (r, c)
                text_x = X[r, c] # X grid corresponds to columns
                text_y = Y[r, c] # Y grid corresponds to rows
                # Clip text position just in case? Should be within extent.
                text_x = np.clip(text_x, extent[0], extent[1]); text_y = np.clip(text_y, extent[2], extent[3])
                # Display value at floor_ppfd[r, c]
                ax.text(text_x, text_y, f"{floor_ppfd[r, c]:.1f}", ha="center", va="center", color="white", fontsize=6,
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.5))

    # Strip Lines
    strip_handles = []
    cmap_strips = plt.cm.cool; num_strip_layers = len(ordered_strip_vertices)
    colors_strips = cmap_strips(np.linspace(0, 1, max(1, num_strip_layers)))
    for layer_idx, vertices in ordered_strip_vertices.items():
        if not vertices: continue
        num_vertices = len(vertices); strip_color = colors_strips[layer_idx - 1] # Layer index starts at 1
        for i in range(num_vertices):
            p1, p2 = vertices[i], vertices[(i + 1) % num_vertices]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=strip_color, linewidth=2.0, linestyle='--', alpha=0.7, zorder=2)
        # Add proxy artist for legend only once per layer
        if num_vertices > 0 and layer_idx not in [h.get_label().split()[-1] for h in strip_handles]: # Check label added
             try:
                 layer_num_str = f'{int(layer_idx)}' # Ensure layer_idx is treated as number for label
                 strip_handles.append(plt.Line2D([0], [0], color=strip_color, lw=2.0, linestyle='--', label=f'Strip Layer {layer_num_str}'))
             except ValueError: pass # Ignore if layer_idx cannot be converted to int (shouldn't happen)

    # COB Markers
    # Check if cob_marker_positions is not None and has data
    if cob_marker_positions is not None and cob_marker_positions.shape[0] > 0:
        ax.scatter(cob_marker_positions[:, 0], cob_marker_positions[:, 1], marker='o',
                   color='red', edgecolors='black', s=50, label="Main COB positions", alpha=0.8, zorder=3)
    # Plot setup
    ax.set_title("Floor PPFD Distribution (COBs + Strips - Full IES)"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    handles, labels = ax.get_legend_handles_labels()
    # Ensure strip handles are not duplicated if layers share colors (unlikely here)
    unique_strip_handles = {h.get_label(): h for h in strip_handles}.values()
    # Only add legend if there are handles to show
    all_handles = handles + list(unique_strip_handles)
    if all_handles:
        ax.legend(handles=all_handles, fontsize='small', loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, linestyle=':', alpha=0.5); plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend

# ------------------------------------
# 8) CSV Output Function (No changes needed)
# ------------------------------------
# write_ppfd_to_csv function remains IDENTICAL to version 1.0. Added check for cob_positions
# (Copied here for file completeness)
def write_ppfd_to_csv(filename, floor_ppfd, X, Y, cob_positions, W, L):
    """
    Writes PPFD data to CSV, assigning each floor point to a distance-based ring (layer).
    Ensures layer 0 has a non-zero radius if FIXED_NUM_LAYERS > 1.
    """
    # Ensure floor_ppfd is not empty
    if floor_ppfd.size == 0:
        print("[CSV Warning] Floor PPFD data is empty. Skipping CSV write.")
        return

    layer_data = {i: [] for i in range(FIXED_NUM_LAYERS)} # Initialize list for each layer

    # Calculate layer radii based on COB positions
    center_x, center_y = W / 2, L / 2
    max_dist_per_layer = np.zeros(FIXED_NUM_LAYERS)

    # Check if cob_positions is valid and non-empty
    if cob_positions is not None and cob_positions.shape[0] > 0 and cob_positions.shape[1] >= 4:
        cob_layers = cob_positions[:, 3].astype(int)
        distances_from_center = np.sqrt((cob_positions[:, 0] - center_x)**2 +
                                        (cob_positions[:, 1] - center_y)**2)

        for i in range(FIXED_NUM_LAYERS):
            layer_mask = (cob_layers == i)
            if np.any(layer_mask):
                max_dist_per_layer[i] = np.max(distances_from_center[layer_mask])
            elif i > 0:
                max_dist_per_layer[i] = max_dist_per_layer[i-1] # If layer empty, use previous max radius
            # Layer 0 distance is handled correctly as 0 if it exists and is at center

    else: # No COBs defined, create arbitrary layers based on distance
        print("[CSV Info] No COB positions provided. Creating layers based on radial distance.")
        max_room_dist = np.sqrt((W/2)**2 + (L/2)**2)
        # Divide distance into FIXED_NUM_LAYERS rings
        max_dist_per_layer = np.linspace(0, max_room_dist, FIXED_NUM_LAYERS + 1)[1:]
        # Ensure layer 0 gets a small radius if layers > 1
        if FIXED_NUM_LAYERS > 1: max_dist_per_layer[0] = max(max_dist_per_layer[0] * 0.1, FLOOR_GRID_RES / 2.0)
        elif FIXED_NUM_LAYERS == 1: max_dist_per_layer[0] = max(max_dist_per_layer[0], FLOOR_GRID_RES / 2.0)


    # Define ring boundaries (outer radius of each layer's ring)
    layer_radii_outer = np.sort(np.unique(max_dist_per_layer)) # Sort unique radii
    # Ensure we have the correct number of boundaries, pad if necessary
    if len(layer_radii_outer) < FIXED_NUM_LAYERS and len(layer_radii_outer)>0:
         layer_radii_outer = np.pad(layer_radii_outer, (0, FIXED_NUM_LAYERS - len(layer_radii_outer)), mode='edge')
    elif len(layer_radii_outer) == 0: # Handle case where all radii were zero
         layer_radii_outer = np.linspace(FLOOR_GRID_RES/2.0, np.sqrt((W/2)**2+(L/2)**2), FIXED_NUM_LAYERS)


    # --- Ensure layer 0 has a small radius if it's currently 0 and other layers exist ---
    if FIXED_NUM_LAYERS > 1 and abs(layer_radii_outer[0]) < 1e-9:
        radius_next = layer_radii_outer[1] if len(layer_radii_outer) > 1 else FLOOR_GRID_RES
        min_radius = min(FLOOR_GRID_RES / 2.0, radius_next / 2.0)
        layer_radii_outer[0] = max(min_radius, 1e-6) # Ensure positive
    elif FIXED_NUM_LAYERS == 1: # If only one layer (layer 0)
         layer_radii_outer[0] = max(layer_radii_outer[0], FLOOR_GRID_RES / 2.0)

    # Add a small epsilon to the last radius to ensure the furthest points are included
    layer_radii_outer[-1] += 0.01 * FLOOR_GRID_RES

    # Assign each floor grid point to a layer
    rows, cols = floor_ppfd.shape
     # Ensure X, Y shapes match floor_ppfd
    if X.shape[0] != rows or X.shape[1] != cols or Y.shape[0] != rows or Y.shape[1] != cols:
         print("[CSV Warning] Mismatch between floor_ppfd and X/Y grid shapes. Cannot assign points to layers.")
    else:
        for r in range(rows):
            for c in range(cols):
                fx, fy = X[r, c], Y[r, c]
                dist_to_center = np.sqrt((fx - center_x)**2 + (fy - center_y)**2)

                # Find which ring the point falls into
                assigned_layer = -1
                for i in range(FIXED_NUM_LAYERS):
                    outer_radius = layer_radii_outer[i]
                    inner_radius = layer_radii_outer[i-1] if i > 0 else 0.0

                    # Check if distance is within the bounds [inner, outer]
                    # Handle layer 0 separately: dist <= outer_radius[0]
                    # Handle other layers: inner_radius[i-1] < dist <= outer_radius[i]
                    if (i == 0 and dist_to_center <= outer_radius + 1e-9): # Include boundary for layer 0
                        assigned_layer = 0
                        break
                    elif (i > 0 and inner_radius < dist_to_center <= outer_radius + 1e-9): # Include boundary
                        assigned_layer = i
                        break

                # Fallback: if slightly outside the largest radius due to grid/float issues, assign to the last layer
                if assigned_layer == -1 and dist_to_center > layer_radii_outer[-1] and dist_to_center < layer_radii_outer[-1] * 1.05:
                     assigned_layer = FIXED_NUM_LAYERS - 1

                if assigned_layer != -1:
                    # Ensure assigned_layer is a valid key
                    if assigned_layer in layer_data:
                        layer_data[assigned_layer].append(floor_ppfd[r, c])
                    # else: print(f"Warning: Assigned layer {assigned_layer} not in layer_data keys.") # Debugging
                # else: # Optional: Warn about points not assigned
                #      print(f"Warning: Point ({fx:.2f}, {fy:.2f}) with dist {dist_to_center:.3f} not assigned to any layer. Radii: {layer_radii_outer}")


    # Write to CSV
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Layer', 'PPFD']) # Header
            for layer_index in sorted(layer_data.keys()):
                # Only write if layer actually has data points
                if layer_data[layer_index]:
                    for ppfd_value in layer_data[layer_index]:
                        writer.writerow([layer_index, ppfd_value])
    except IOError as e:
        print(f"Error writing PPFD data to {filename}: {e}")


# ------------------------------------
# 9) Main Simulation Function (REVISED Call Signatures)
# ------------------------------------
def simulate_lighting(light_positions, light_lumens, light_types, X, Y, patches):
    """ Runs the full lighting simulation pipeline. """
    # Check if patches exist
    if not patches or len(patches) != 4 or patches[0].shape[0] == 0:
        print("[Error] No patches available for simulation. Returning zero PPFD.")
        return np.zeros_like(X, dtype=np.float64)

    p_centers, p_areas, p_normals, p_refl = patches

    # Check if lights exist
    if light_positions.shape[0] == 0:
         print("[Warning] No light sources defined. Direct illumination will be zero.")
         floor_lux_direct = np.zeros_like(X, dtype=np.float64)
         patch_direct_lux = np.zeros(p_centers.shape[0], dtype=np.float64)
    else:
        print("[Simulation] Calculating direct floor illuminance (using Full IES)...")
        floor_lux_direct = compute_direct_floor(
            light_positions, light_lumens, light_types, X, Y,
            # Pass full COB IES data
            COB_IES_V_ANGLES, COB_IES_H_ANGLES, COB_IES_CANDELAS_2D, COB_IES_FILE_LUMENS_NORM,
            # Pass full Strip IES data
            STRIP_IES_V_ANGLES, STRIP_IES_H_ANGLES, STRIP_IES_CANDELAS_2D, STRIP_IES_FILE_LUMENS_NORM
        )

        print("[Simulation] Calculating direct patch illuminance (using Full IES)...")
        patch_direct_lux = compute_patch_direct(
            light_positions, light_lumens, light_types,
            p_centers, p_normals, p_areas,
            # Pass full COB IES data
            COB_IES_V_ANGLES, COB_IES_H_ANGLES, COB_IES_CANDELAS_2D, COB_IES_FILE_LUMENS_NORM,
            # Pass full Strip IES data
            STRIP_IES_V_ANGLES, STRIP_IES_H_ANGLES, STRIP_IES_CANDELAS_2D, STRIP_IES_FILE_LUMENS_NORM
        )

    # Check for NaNs after direct calculation
    if np.any(np.isnan(floor_lux_direct)) or np.any(np.isnan(patch_direct_lux)):
         print("[ERROR] NaN detected after direct illumination calculation. Check IES data and geometry.")
         # Replace NaNs with 0 to attempt proceeding, but results might be compromised
         floor_lux_direct = np.nan_to_num(floor_lux_direct)
         patch_direct_lux = np.nan_to_num(patch_direct_lux)


    print("[Simulation] Running radiosity...")
    patch_total_incident_lux = iterative_radiosity_loop(
        p_centers, p_normals, patch_direct_lux, p_areas, p_refl,
        MAX_RADIOSITY_BOUNCES, RADIOSITY_CONVERGENCE_THRESHOLD
    )
    if np.any(np.isnan(patch_total_incident_lux)):
         print("[ERROR] NaN detected after radiosity calculation.")
         patch_total_incident_lux = np.nan_to_num(patch_total_incident_lux)

    print("[Simulation] Calculating indirect floor illuminance (Monte Carlo)...")
    floor_lux_indirect = compute_reflection_on_floor(
        X, Y, p_centers, p_normals, p_areas, patch_total_incident_lux, p_refl, MC_SAMPLES
    )
    if np.any(np.isnan(floor_lux_indirect)):
         print("[ERROR] NaN detected after indirect reflection calculation.")
         floor_lux_indirect = np.nan_to_num(floor_lux_indirect)

    # Combine and convert to PPFD (No changes here)
    total_floor_lux = floor_lux_direct + floor_lux_indirect
    # Ensure luminous efficacy is positive
    effic = LUMINOUS_EFFICACY if LUMINOUS_EFFICACY > 1e-9 else 1.0
    total_radiant_Wm2 = total_floor_lux / effic
    floor_ppfd = total_radiant_Wm2 * CONVERSION_FACTOR
    # Add final check for NaN/Inf values and replace with 0
    if np.any(np.isnan(floor_ppfd)) or np.any(np.isinf(floor_ppfd)):
        print("[ERROR] NaN or Inf detected in final floor_ppfd array. Replacing with 0.")
        floor_ppfd = np.nan_to_num(floor_ppfd, nan=0.0, posinf=0.0, neginf=0.0)

    return floor_ppfd

# ------------------------------------
# 10) Execution Block (Adjust parameter definitions)
# ------------------------------------
def main():
    # --- Simulation-Specific Parameters ---
    W = 6.096 # Room Width (m)
    L = 6.096 # Room Length (m)
    H = 0.9144 # Room Height (m) -> Light Z-pos is H * 0.95

    # Define number of layers based on desired pattern complexity
    global FIXED_NUM_LAYERS
    FIXED_NUM_LAYERS = 10 # Set the desired number of layers

    # COB Lumen parameters per layer (Length must match FIXED_NUM_LAYERS)
    # Example: All COBs at 10k lumens
    cob_params = np.array([10000.0] * FIXED_NUM_LAYERS, dtype=np.float64)

    # Strip Module Lumen parameters per layer (Length must match FIXED_NUM_LAYERS)
    # Layer 0 is ignored for strips, but needs a placeholder value.
    # Example: Increasing brightness for outer layers
    strip_lumens = [0.0] + list(np.linspace(1000, 8000, FIXED_NUM_LAYERS - 1))
    strip_module_lumen_params = np.array(strip_lumens, dtype=np.float64)

    # --- Runtime Parameter Adjustments (Optional, e.g., via command line) ---
    # Example: Could override MC_SAMPLES via args if implemented
    global MC_SAMPLES
    # global FLOOR_GRID_RES
    # (Add argparse here if needed to modify these at runtime)

    # --- Ensure IES data is loaded (Check globals using new names) ---
    # The check happens implicitly at load time now, raising SystemExit if failed.
    # We can add an explicit check here too if desired:
    if COB_IES_V_ANGLES is None or STRIP_IES_V_ANGLES is None:
        print("Error: Required IES data arrays are missing. Cannot proceed.")
        return

    print("Preparing geometry (COBs, Strip Modules, Room)...")
    # Call prepare_geometry (unchanged signature)
    light_positions, light_lumens, light_types, cob_positions_only, ordered_strip_vertices, X, Y, patches = prepare_geometry(
        W, L, H, cob_params, strip_module_lumen_params
    )

    total_lights = light_positions.shape[0]
    if total_lights == 0 and patches[0].shape[0] > 0: # Check if lights are zero but patches exist
        print("Warning: No light sources generated with positive lumens. Simulation will only calculate reflections of zero direct light.")
        # Proceed anyway, result should be zero.
    elif patches[0].shape[0] == 0:
         print("Error: No room patches generated. Cannot run simulation.")
         return # Stop if no patches

    print(f"\nStarting simulation for {total_lights} total light emitters...")
    # simulate_lighting uses the globally loaded IES data via the new variables
    floor_ppfd = simulate_lighting(light_positions, light_lumens, light_types, X, Y, patches)

    # Check if floor_ppfd is valid before statistics/plotting
    if floor_ppfd is None or floor_ppfd.size == 0:
         print("Error: Simulation did not produce valid floor PPFD data.")
         return

    print("Simulation complete. Calculating statistics...")
    # --- Statistics Calculation (Added check for zero mean) ---
    mean_ppfd = np.mean(floor_ppfd); std_dev = np.std(floor_ppfd)
    min_ppfd = np.min(floor_ppfd); max_ppfd = np.max(floor_ppfd)
    if mean_ppfd > 1e-9: # Check for non-zero mean before calculating MAD/RMSE based uniformity
        mad = np.mean(np.abs(floor_ppfd - mean_ppfd)); rmse = np.sqrt(np.mean((floor_ppfd - mean_ppfd)**2))
        cv_percent = (std_dev / mean_ppfd) * 100; min_max_ratio = min_ppfd / max_ppfd if max_ppfd > 0 else 0
        min_avg_ratio = min_ppfd / mean_ppfd; cu_percent = (1 - mad / mean_ppfd) * 100
        dou_percent = (1 - rmse / mean_ppfd) * 100
    else: # Handle zero mean case
         mad, rmse, cv_percent, min_max_ratio, min_avg_ratio, cu_percent, dou_percent = 0, 0, 0, 0, 0, 0, 0

    # --- Output Results (Unchanged) ---
    print(f"\n--- Results ---"); print(f"Room Dimensions (WxLxH): {W:.2f}m x {L:.2f}m x {H:.2f}m")
    print(f"Floor Grid Resolution: {FLOOR_GRID_RES:.3f}m ({X.shape[1]}x{X.shape[0]} points)")
    num_cobs = cob_positions_only.shape[0]
    print(f"Number of Main COBs: {num_cobs}")
    strip_emitters_count = total_lights - num_cobs
    print(f"Number of Strip Modules Placed: {strip_emitters_count}"); print(f"Total Emitters Simulated: {total_lights}")
    print(f"Average PPFD: {mean_ppfd:.2f} µmol/m²/s"); print(f"Std Deviation: {std_dev:.2f}"); print(f"RMSE: {rmse:.2f}"); print(f"MAD: {mad:.2f}")
    print(f"Min PPFD: {min_ppfd:.2f}"); print(f"Max PPFD: {max_ppfd:.2f}")
    print(f"\n--- Uniformity ---"); print(f"CV (%): {cv_percent:.2f}%"); print(f"Min/Max Ratio: {min_max_ratio:.3f}")
    print(f"Min/Avg Ratio: {min_avg_ratio:.3f}"); print(f"CU (%) (MAD-based): {cu_percent:.2f}%"); print(f"DOU (%) (RMSE-based): {dou_percent:.2f}%")

    # --- CSV Output (Changed filename slightly) ---
    csv_filename = "ppfd_layer_data_full_ies_v2.csv"
    print(f"\nWriting layer-based PPFD data to {csv_filename}...")
    write_ppfd_to_csv(csv_filename, floor_ppfd, X, Y, cob_positions_only, W, L)
    print("CSV writing complete.")

    # --- Command Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Lighting Simulation Script with COBs and Strips (Full IES)")
    parser.add_argument('--no-plot', action='store_true', help='Disable the heatmap plot')
    parser.add_argument('--anno', type=int, default=15, help='Annotation density step for plot (0 disables)')
    args = parser.parse_args()

    # --- Heatmap Plot (Optional - Unchanged) ---
    if not args.no_plot:
        print("\nGenerating heatmap plot...")
        # Check if floor_ppfd has valid data before plotting
        if np.all(np.isfinite(floor_ppfd)):
            plot_heatmap(floor_ppfd, X, Y, cob_positions_only, ordered_strip_vertices, W, L, annotation_step=args.anno)
            print("Plot window opened. Close plot window to exit.")
            plt.show()
        else:
            print("[Plot Error] Floor PPFD data contains non-finite values. Skipping plot.")

# --- Execution Guard ---
if __name__ == "__main__":
    # IES loading happens globally and raises SystemExit on failure
    # SPD loading happens globally
    main()

# --- END OF REVISED FILE lighting-simulation-data(2.0).py ---
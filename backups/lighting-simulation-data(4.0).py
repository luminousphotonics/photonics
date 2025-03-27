# --- START OF FILE lighting-simulation-data(4.3).py ---

#!/usr/bin/env python3
import csv
import math
import time # Import time for benchmarking
import numpy as np
from numba import njit, prange
from functools import lru_cache
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import argparse
import warnings

# Suppress NumbaPerformanceWarning
from numba import NumbaPerformanceWarning
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)


# ------------------------------------
# 1) Basic Config & Reflectances (Unchanged from 4.0)
# ------------------------------------
REFL_WALL = 0.8
REFL_CEIL = 0.8
REFL_FLOOR = 0.1

LUMINOUS_EFFICACY = 182.0  # lumens/W
SPD_FILE = "/Users/austinrouse/photonics/backups/spd_data.csv" # <<< SET PATH

MAX_RADIOSITY_BOUNCES = 10
RADIOSITY_CONVERGENCE_THRESHOLD = 1e-3

FLOOR_SUBDIVS_X = 10
FLOOR_SUBDIVS_Y = 10
WALL_SUBDIVS_X = 10
WALL_SUBDIVS_Y = 5
CEIL_SUBDIVS_X = 10
CEIL_SUBDIVS_Y = 10

FLOOR_GRID_RES = 0.08  # m (Fine grid for final results)
FIXED_NUM_LAYERS = 10 # Tied to the structure of build_cob_positions and params length
GEOMETRY_MARGIN = 0.95 # Margin to pull light sources away from walls
MC_SAMPLES = 28 # Monte Carlo samples for *indirect floor* illumination
N_FF_SAMPLES = 0 # Monte Carlo samples for *Form Factor* calculation

# --- LED Strip Module Configuration ---
STRIP_IES_FILE = "/Users/austinrouse/photonics/backups/Standard_Horti_G2.ies" # <<< SET PATH
STRIP_MODULE_LENGTH = 0.561

# --- Constants for emitter types ---
EMITTER_TYPE_COB = 0
EMITTER_TYPE_STRIP = 1

# --- COB Configuration ---
COB_IES_FILE = "/Users/austinrouse/photonics/backups/cob.ies" # <<< SET PATH

# --- Epsilon for numerical stability ---
EPSILON = 1e-9

# ------------------------------------
# 4.5) Load and Prepare Full IES Data (With added diagnostics)
# ------------------------------------
# Functions parse_ies_file_full, compute_conversion_factor remain IDENTICAL to v3.1
def parse_ies_file_full(ies_filepath):
    """ Parses an IESNA:LM-63-1995 file format (Full 2D). """
    print(f"[IES - Full] Attempting to parse file: {ies_filepath}")
    # ... (rest of function is identical to v3.1) ...
    try:
        with open(ies_filepath, 'r', encoding='ascii', errors='ignore') as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"[IES - Full] Error: File not found at '{ies_filepath}'")
        return None, None, None, None
    except Exception as e:
        print(f"[IES - Full] Error reading file '{ies_filepath}': {e}")
        return None, None, None, None

    try:
        line_idx = 0
        while line_idx < len(lines) and not lines[line_idx].upper().startswith('TILT='): line_idx += 1
        if line_idx >= len(lines): raise ValueError("TILT= line not found")
        tilt_line = lines[line_idx]; line_idx += 1
        if "NONE" not in tilt_line.upper(): print(f"[IES - Full] Warning: TILT directive ignored.")

        if line_idx >= len(lines): raise ValueError("Missing parameter line 1")
        params1 = lines[line_idx].split(); line_idx += 1
        num_lamps = int(params1[0]); lumens_per_lamp = float(params1[1]); multiplier = float(params1[2])
        num_v_angles = int(params1[3]); num_h_angles = int(params1[4])
        photometric_type = int(params1[5]); units_type = int(params1[6])

        if line_idx >= len(lines): raise ValueError("Missing parameter line 2")
        params2 = lines[line_idx].split(); line_idx += 1
        ballast_factor = float(params2[0]); input_watts = float(params2[2])

        v_angles_list = []; h_angles_list = []; candela_list_flat = []
        while len(v_angles_list) < num_v_angles:
            if line_idx >= len(lines): raise ValueError(f"File ended while reading vertical angles")
            v_angles_list.extend([float(a) for a in lines[line_idx].split()]); line_idx += 1
        vertical_angles = np.array(v_angles_list, dtype=np.float64)
        if len(vertical_angles) != num_v_angles: raise ValueError(f"Vertical angle count mismatch")
        if not np.all(np.diff(vertical_angles) >= 0): vertical_angles = np.sort(vertical_angles)

        while len(h_angles_list) < num_h_angles:
             if line_idx >= len(lines): raise ValueError(f"File ended while reading horizontal angles")
             h_angles_list.extend([float(a) for a in lines[line_idx].split()]); line_idx += 1
        horizontal_angles = np.array(h_angles_list, dtype=np.float64)
        if len(horizontal_angles) != num_h_angles: raise ValueError(f"Horizontal angle count mismatch")
        if not np.all(np.diff(horizontal_angles) >= 0): horizontal_angles = np.sort(horizontal_angles)

        expected_candela_count = num_v_angles * num_h_angles
        while line_idx < len(lines) and len(candela_list_flat) < expected_candela_count:
            try:
                candela_list_flat.extend([float(c) for c in lines[line_idx].split()])
            except ValueError:
                 if any(kw in lines[line_idx].upper() for kw in ['[', ']', 'END', 'LABEL']):
                      print(f"[IES - Full] Info: Skipping potential keyword line: {lines[line_idx]}")
                 else:
                      print(f"[IES - Full] Warning: Non-numeric data in candela section (line {line_idx + 1}). Stopping read: {lines[line_idx]}")
                      break
            line_idx += 1

        if len(candela_list_flat) != expected_candela_count:
            if len(candela_list_flat) > expected_candela_count:
                print(f"[IES - Full] Warning: Found {len(candela_list_flat)} candela values, expected {expected_candela_count}. Truncating.")
                candela_list_flat = candela_list_flat[:expected_candela_count]
            else: raise ValueError(f"Candela value count mismatch: Found {len(candela_list_flat)}, expected {expected_candela_count}.")

        candela_data_raw = np.array(candela_list_flat, dtype=np.float64)
        if np.any(candela_data_raw < 0):
             print(f"[IES - Full] Warning: Negative candela values found. Clamping to 0.")
             candela_data_raw = np.maximum(candela_data_raw, 0.0)
        candela_data_2d = candela_data_raw.reshape((num_h_angles, num_v_angles)).T # Shape (num_v, num_h)

        ies_file_lumens_norm = lumens_per_lamp * num_lamps * multiplier * ballast_factor
        if ies_file_lumens_norm <= EPSILON:
             print("[IES - Full] Warning: Calculated normalization lumens near zero. Using 1.0."); ies_file_lumens_norm = 1.0

        print(f"[IES - Full] Parsed: Type {photometric_type}, Norm Lumens {ies_file_lumens_norm:.2f}, Watts {input_watts:.2f}")
        print(f"[IES - Full] V Angles ({num_v_angles}), H Angles ({num_h_angles}), Candela Grid {candela_data_2d.shape}")
        return vertical_angles, horizontal_angles, candela_data_2d, ies_file_lumens_norm

    except Exception as e:
        print(f"[IES - Full] Unexpected error during parsing near line {line_idx+1}: {e}")
        import traceback; traceback.print_exc()
        return None, None, None, None

print("\n--- Loading Strip IES Data ---")
(STRIP_IES_V_ANGLES, STRIP_IES_H_ANGLES, STRIP_IES_CANDELAS_2D, STRIP_IES_FILE_LUMENS_NORM) = parse_ies_file_full(STRIP_IES_FILE)
if STRIP_IES_V_ANGLES is None: raise SystemExit("Failed to load strip IES file.")
# <<< ADDED H Angle Range Print >>>
if STRIP_IES_H_ANGLES is not None and STRIP_IES_H_ANGLES.size > 0:
    print(f"[IES - Strip] H Angle Range: {np.min(STRIP_IES_H_ANGLES):.1f} to {np.max(STRIP_IES_H_ANGLES):.1f}")
else:
    print("[IES - Strip] H Angles not loaded or empty.")
# <<< END PRINT >>>

print("\n--- Loading COB IES Data ---")
(COB_IES_V_ANGLES, COB_IES_H_ANGLES, COB_IES_CANDELAS_2D, COB_IES_FILE_LUMENS_NORM) = parse_ies_file_full(COB_IES_FILE)
if COB_IES_V_ANGLES is None: raise SystemExit("Failed to load COB IES file.")
# <<< ADDED H Angle Range Print >>>
if COB_IES_H_ANGLES is not None and COB_IES_H_ANGLES.size > 0:
    print(f"[IES - COB] H Angle Range: {np.min(COB_IES_H_ANGLES):.1f} to {np.max(COB_IES_H_ANGLES):.1f}")
else:
    print("[IES - COB] H Angles not loaded or empty.")
# <<< END PRINT >>>


def compute_conversion_factor(spd_file):
    """ Computes PPFD conversion factor from SPD data. """
    try:
        raw_spd = np.loadtxt(spd_file, delimiter=' ', skiprows=1)
        unique_wl, indices, counts = np.unique(raw_spd[:, 0], return_inverse=True, return_counts=True)
        counts_nonzero = np.maximum(counts, 1)
        avg_intens = np.bincount(indices, weights=raw_spd[:, 1]) / counts_nonzero
        spd = np.column_stack((unique_wl, avg_intens))
    except Exception as e: print(f"Error loading SPD: {e}"); return 0.0138 # Fallback value
    wl = spd[:, 0]; intens = spd[:, 1]; sort_idx = np.argsort(wl); wl = wl[sort_idx]; intens = intens[sort_idx]
    mask_par = (wl >= 400) & (wl <= 700); PAR_fraction = 1.0
    if len(wl) >= 2:
        tot = np.trapz(intens, wl)
        if tot > EPSILON:
             if np.count_nonzero(mask_par) >= 2: tot_par = np.trapz(intens[mask_par], wl[mask_par]); PAR_fraction = tot_par / tot
             else: print("[SPD Warning] Not enough PAR data points for fraction.")
        else: print("[SPD Warning] Zero total SPD intensity.")
    else: print("[SPD Warning] Not enough SPD points for integration.")
    wl_m = wl * 1e-9; h, c, N_A = 6.626e-34, 3.0e8, 6.022e23; lambda_eff = 0.0
    if np.count_nonzero(mask_par) >= 2:
        numerator = np.trapz(wl_m[mask_par] * intens[mask_par], wl_m[mask_par])
        denominator = np.trapz(intens[mask_par], wl_m[mask_par])
        if denominator > EPSILON: lambda_eff = numerator / denominator
    if lambda_eff <= EPSILON:
        print("[SPD Warning] Could not calculate effective PAR wavelength.")
        if np.count_nonzero(mask_par) > 0: lambda_eff = np.mean(wl_m[mask_par])
        else: lambda_eff = 550e-9 # Fallback to ~green
    E_photon = (h * c / lambda_eff) if lambda_eff > EPSILON else 1.0
    conversion_factor = (1.0 / E_photon) * (1e6 / N_A) * PAR_fraction
    print(f"[INFO] SPD: PAR fraction={PAR_fraction:.3f}, conv_factor={conversion_factor:.5f} µmol/J")
    return conversion_factor
CONVERSION_FACTOR = compute_conversion_factor(SPD_FILE)

# ------------------------------------
# 4.6) Numba-Compatible 2D Intensity Functions (REVISED Interpolation v4.9)
# ------------------------------------
# _interp_1d_linear_safe remains the same
# Ensure this is the v3.1/v4.0 version
# --- Restore _interp_1d_linear_safe to v3.1/v4.0 ---
@njit(cache=True)
def _interp_1d_linear_safe(x, xp, fp):
    """Safely interpolates 1D, clamping boundaries. (Restored v3.1/v4.0)"""
    idx = np.searchsorted(xp, x, side='right')
    if idx == 0: return fp[0]
    if idx == len(xp): return fp[-1]
    x0, x1 = xp[idx-1], xp[idx]; y0, y1 = fp[idx-1], fp[idx]
    delta_x = x1 - x0
    if delta_x <= EPSILON: return y0
    weight = (x - x0) / delta_x
    weight = max(0.0, min(weight, 1.0))
    return y0 * (1.0 - weight) + y1 * weight

# --- Restore interpolate_2d_bilinear to v3.1/v4.0 ---
@njit(cache=True)
def interpolate_2d_bilinear(angle_v_deg, angle_h_deg, ies_v_angles, ies_h_angles, ies_candelas_2d):
    """ Bilinear interpolation on 2D candela grid with angle wrapping/clamping. (Restored v3.1/v4.0) """
    num_v, num_h = ies_candelas_2d.shape
    if num_v == 0 or num_h == 0: return 0.0
    if num_v == 1 and num_h == 1: return ies_candelas_2d[0, 0]
    # --- Vertical Interpolation Setup ---
    if num_v == 1:
        iv = 0; tv = 0.0
    else:
        iv = max(0, min(np.searchsorted(ies_v_angles, angle_v_deg, side='right') - 1, num_v - 2))
        v0, v1 = ies_v_angles[iv], ies_v_angles[iv+1]; delta_v = v1 - v0
        tv = (angle_v_deg - v0) / delta_v if delta_v > EPSILON else (0.0 if angle_v_deg <= v0 else 1.0)
        tv = max(0.0, min(tv, 1.0))
    # --- Horizontal Interpolation Setup (with angle wrapping/symmetry) ---
    if num_h == 1:
         ih = 0; th = 0.0
         C0 = ies_candelas_2d[iv, 0]; C1 = ies_candelas_2d[iv+1, 0] if num_v > 1 else C0
         return C0 * (1.0 - tv) + C1 * tv
    else:
        h_min, h_max = ies_h_angles[0], ies_h_angles[-1]; angle_h_wrapped = angle_h_deg
        h_range = h_max - h_min
        is_full_360 = abs(h_range - 360.0) < 5.0 and abs(h_min) < EPSILON
        is_symmetric_180 = abs(h_max - 180.0) < 5.0 and abs(h_min) < EPSILON and not is_full_360
        is_symmetric_90 = abs(h_max - 90.0) < 5.0 and abs(h_min) < EPSILON and not is_full_360 and not is_symmetric_180
        if is_full_360:
            angle_h_wrapped = h_min + ((angle_h_deg - h_min) % 360.0)
            if abs(angle_h_wrapped - (h_min + 360.0)) < EPSILON: angle_h_wrapped = h_max
        elif is_symmetric_180:
            angle_h_wrapped_mod = angle_h_deg % 360.0 # Use temp var for clarity
            angle_h_wrapped = 360.0 - angle_h_wrapped_mod if angle_h_wrapped_mod > 180.0 else angle_h_wrapped_mod
            angle_h_wrapped = max(0.0, min(angle_h_wrapped, 180.0))
        elif is_symmetric_90:
            angle_h_wrapped_mod = angle_h_deg % 360.0 # Use temp var for clarity
            if angle_h_wrapped_mod > 270.0: angle_h_wrapped = 360.0 - angle_h_wrapped_mod
            elif angle_h_wrapped_mod > 180.0: angle_h_wrapped = angle_h_wrapped_mod - 180.0
            elif angle_h_wrapped_mod > 90.0: angle_h_wrapped = 180.0 - angle_h_wrapped_mod
            else: angle_h_wrapped = angle_h_wrapped_mod # Keep if in 0-90
            angle_h_wrapped = max(0.0, min(angle_h_wrapped, 90.0))

        ih = max(0, min(np.searchsorted(ies_h_angles, angle_h_wrapped, side='right') - 1, num_h - 2))
        h0, h1 = ies_h_angles[ih], ies_h_angles[ih+1]; delta_h = h1 - h0
        th = (angle_h_wrapped - h0) / delta_h if delta_h > EPSILON else (0.0 if angle_h_wrapped <= h0 else 1.0)
        th = max(0.0, min(th, 1.0))
        C00 = ies_candelas_2d[iv, ih]; C10 = ies_candelas_2d[iv+1, ih] if num_v > 1 else C00
        C01 = ies_candelas_2d[iv, ih+1]; C11 = ies_candelas_2d[iv+1, ih+1] if num_v > 1 else C01
        C_h0 = C00 * (1.0 - th) + C01 * th; C_h1 = C10 * (1.0 - th) + C11 * th
        candela_raw = C_h0 * (1.0 - tv) + C_h1 * tv
        return max(0.0, candela_raw)

# --- Restore calculate_ies_intensity_2d to v3.1/v4.0 (NO snapping logic) ---
@njit(cache=True)
def calculate_ies_intensity_2d(dx, dy, dz, dist, total_emitter_lumens, ies_v_angles, ies_h_angles, ies_candelas_2d, ies_file_lumens_norm):
    """ Calculates luminous intensity (cd) using full 2D IES data. (Restored v3.1/v4.0) """
    candela_raw = 0.0
    if dist < EPSILON:
        if ies_candelas_2d.size > 0: candela_raw = ies_candelas_2d[0, 0]
    else:
        cos_theta_nadir = max(-1.0, min(1.0, -dz / dist))
        angle_v_deg = math.degrees(math.acos(cos_theta_nadir))
        angle_h_deg = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        # Directly call the original interpolation
        candela_raw = interpolate_2d_bilinear(angle_v_deg, angle_h_deg, ies_v_angles, ies_h_angles, ies_candelas_2d)

    norm_factor = max(ies_file_lumens_norm, EPSILON)
    scaling_factor = total_emitter_lumens / norm_factor
    return candela_raw * scaling_factor


# ------------------------------------
# 5) Geometry Building (Unchanged from 4.0)
# ------------------------------------
# Functions cached_build_floor_grid, build_floor_grid, cached_build_patches,
# build_patches, _get_cob_abstract_coords_and_transform, _apply_transform,
# build_all_light_sources, prepare_geometry remain IDENTICAL to v4.0.

# --- Revised cached_build_floor_grid from previous step (keep) ---
@lru_cache(maxsize=32)
def cached_build_floor_grid(W: float, L: float):
    """Builds the FINE floor grid for final results (separate from radiosity patches).
       Revised to ensure symmetrical centering based on resolution.
    """
    num_cells_x = max(1, int(round(W / FLOOR_GRID_RES)))
    num_cells_y = max(1, int(round(L / FLOOR_GRID_RES)))
    actual_res_x = W / num_cells_x
    actual_res_y = L / num_cells_y
    xs = np.linspace(actual_res_x / 2.0, W - actual_res_x / 2.0, num_cells_x)
    ys = np.linspace(actual_res_y / 2.0, L - actual_res_y / 2.0, num_cells_y)
    X, Y = np.meshgrid(xs, ys)
    print(f"[Grid] Centered result grid created: {X.shape[1]}x{X.shape[0]} points ({actual_res_x:.3f}x{actual_res_y:.3f}m resolution).")
    print(f"[Grid] Bottom-Left Center: ({X[0,0]:.4f}, {Y[0,0]:.4f})")
    print(f"[Grid] Top-Right Center:   ({X[-1,-1]:.4f}, {Y[-1,-1]:.4f})")
    return X, Y

def build_floor_grid(W, L): return cached_build_floor_grid(W, L)


@lru_cache(maxsize=32)
def cached_build_patches(W: float, L: float, H: float):
    """ Builds patches (including subdivided floor) and returns center, area, normal, reflectance,
        tangent1, tangent2, half_len1, half_len2 """
    # ... (body identical to v3.1) ...
    patches_data = {
        'center': [], 'area': [], 'normal': [], 'refl': [],
        't1': [], 't2': [], 'hs1': [], 'hs2': []
    }

    def add_patch(center, area, normal, refl, t1, t2, hs1, hs2):
        if area > EPSILON:
            patches_data['center'].append(center)
            patches_data['area'].append(area)
            patches_data['normal'].append(normal)
            patches_data['refl'].append(refl)
            patches_data['t1'].append(t1)
            patches_data['t2'].append(t2)
            patches_data['hs1'].append(hs1)
            patches_data['hs2'].append(hs2)

    # --- Floor Patches (Subdivided) ---
    xs_f = np.linspace(0, W, FLOOR_SUBDIVS_X + 1)
    ys_f = np.linspace(0, L, FLOOR_SUBDIVS_Y + 1)
    dx_f = (xs_f[1]-xs_f[0]) if FLOOR_SUBDIVS_X > 0 else W
    dy_f = (ys_f[1]-ys_f[0]) if FLOOR_SUBDIVS_Y > 0 else L
    hs1_f, hs2_f = dx_f / 2.0, dy_f / 2.0
    t1_f, t2_f = np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0))
    normal_f = np.array((0.0, 0.0, 1.0))
    for i in range(FLOOR_SUBDIVS_X):
        for j in range(FLOOR_SUBDIVS_Y):
            cx = (xs_f[i] + xs_f[i+1]) / 2
            cy = (ys_f[j] + ys_f[j+1]) / 2
            area = dx_f * dy_f
            add_patch((cx, cy, 0.0), area, normal_f, REFL_FLOOR, t1_f, t2_f, hs1_f, hs2_f)

    # --- Ceiling Patches ---
    xs_c = np.linspace(0, W, CEIL_SUBDIVS_X + 1); ys_c = np.linspace(0, L, CEIL_SUBDIVS_Y + 1)
    dx_c = (xs_c[1]-xs_c[0]) if CEIL_SUBDIVS_X > 0 else W
    dy_c = (ys_c[1]-ys_c[0]) if CEIL_SUBDIVS_Y > 0 else L
    hs1_c, hs2_c = dx_c / 2.0, dy_c / 2.0
    t1_c, t2_c = np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0))
    normal_c = np.array((0.0, 0.0, -1.0))
    for i in range(CEIL_SUBDIVS_X):
        for j in range(CEIL_SUBDIVS_Y):
            cx = (xs_c[i] + xs_c[i+1]) / 2; cy = (ys_c[j] + ys_c[j+1]) / 2
            area = dx_c * dy_c
            add_patch((cx, cy, H), area, normal_c, REFL_CEIL, t1_c, t2_c, hs1_c, hs2_c)

    # --- Wall Patches ---
    wall_defs = [
        ('y', 0.0, (0.0, 1.0, 0.0), (0, W), (0, H), WALL_SUBDIVS_X, WALL_SUBDIVS_Y, 'x', 'z'), # y=0
        ('y', L,   (0.0,-1.0, 0.0), (0, W), (0, H), WALL_SUBDIVS_X, WALL_SUBDIVS_Y, 'x', 'z'), # y=L
        ('x', 0.0, (1.0, 0.0, 0.0), (0, L), (0, H), WALL_SUBDIVS_X, WALL_SUBDIVS_Y, 'y', 'z'), # x=0
        ('x', W,  (-1.0, 0.0, 0.0), (0, L), (0, H), WALL_SUBDIVS_X, WALL_SUBDIVS_Y, 'y', 'z'), # x=W
    ]
    axis_vec = {'x': np.array((1.0, 0.0, 0.0)), 'y': np.array((0.0, 1.0, 0.0)), 'z': np.array((0.0, 0.0, 1.0))}

    for axis, fixed_val, normal_tuple, r1, r2, sub1, sub2, ax1, ax2 in wall_defs:
        s1 = max(1, sub1); s2 = max(1, sub2)
        c1 = np.linspace(r1[0], r1[1], s1 + 1); c2 = np.linspace(r2[0], r2[1], s2 + 1)
        dc1 = (c1[1]-c1[0]) if s1 > 0 else (r1[1]-r1[0])
        dc2 = (c2[1]-c2[0]) if s2 > 0 else (r2[1]-r2[0])
        hs1_w, hs2_w = dc1 / 2.0, dc2 / 2.0
        t1_w, t2_w = axis_vec[ax1], axis_vec[ax2]
        normal_w = np.array(normal_tuple)

        for i in range(s1):
            for j in range(s2):
                pc1 = (c1[i] + c1[i+1]) / 2; pc2 = (c2[j] + c2[j+1]) / 2
                area = dc1 * dc2
                center = (pc1, fixed_val, pc2) if axis == 'y' else (fixed_val, pc1, pc2)
                add_patch(center, area, normal_w, REFL_WALL, t1_w, t2_w, hs1_w, hs2_w)

    if not patches_data['center']:
         print("[ERROR] No patches generated.")
         return {k: np.empty((0, 3) if k in ['center', 'normal', 't1', 't2'] else 0, dtype=np.float64) for k in patches_data}

    print(f"[Geometry] Generated {len(patches_data['center'])} patches (incl. subdivided floor).")
    return {k: np.array(v, dtype=np.float64) for k, v in patches_data.items()}

def build_patches(W, L, H): return cached_build_patches(W, L, H)

def _get_cob_abstract_coords_and_transform(W, L, H, num_total_layers):
    """Generates abstract layer/position coordinates and transform parameters."""
    # ... (body identical to v4.0) ...
    n = num_total_layers - 1; abstract_positions = []
    abstract_positions.append({'x': 0, 'y': 0, 'h': H, 'layer': 0, 'is_vertex': True})
    for i in range(1, n + 1):
        for x in range(i, 0, -1): abstract_positions.append({'x': x, 'y': i - x, 'h': H, 'layer': i, 'is_vertex': (x == i or x == 0)})
        for x in range(0, -i, -1): abstract_positions.append({'x': x, 'y': i + x, 'h': H, 'layer': i, 'is_vertex': (x == -i or x == 0)})
        for x in range(-i, 0, 1): abstract_positions.append({'x': x, 'y': -i - x, 'h': H, 'layer': i, 'is_vertex': (x == -i or x == 0)})
        for x in range(0, i + 1, 1): abstract_positions.append({'x': x, 'y': -i + x, 'h': H, 'layer': i, 'is_vertex': (x == i or x == 0)})

    unique_positions = []; seen_coords = set()
    for pos in abstract_positions:
        coord_tuple = (pos['x'], pos['y'], pos['layer'])
        if coord_tuple not in seen_coords:
            unique_positions.append(pos)
            seen_coords.add(coord_tuple)
    unique_positions.sort(key=lambda p: (p['layer'], math.atan2(p['y'], p['x']) if p['layer'] > 0 else 0))

    theta = math.radians(45); cos_t, sin_t = math.cos(theta), math.sin(theta)
    center_x, center_y = W / 2, L / 2
    effective_W = W * GEOMETRY_MARGIN; effective_L = L * GEOMETRY_MARGIN
    scale_x = max(effective_W / (n * math.sqrt(2)) if n > 0 else effective_W / 2, EPSILON)
    scale_y = max(effective_L / (n * math.sqrt(2)) if n > 0 else effective_L / 2, EPSILON)
    transform_params = {'center_x': center_x, 'center_y': center_y,
                       'scale_x': scale_x, 'scale_y': scale_y,
                       'cos_t': cos_t, 'sin_t': sin_t, 'H': H, 'Z_pos': H * 0.95 }
    return unique_positions, transform_params

def _apply_transform(abstract_pos, transform_params):
    """Applies scaling, rotation, and translation to abstract coordinates."""
    # ... (body identical to v3.1) ...
    ax, ay = abstract_pos['x'], abstract_pos['y']
    rx = ax * transform_params['cos_t'] - ay * transform_params['sin_t']
    ry = ax * transform_params['sin_t'] + ay * transform_params['cos_t']
    px = transform_params['center_x'] + rx * transform_params['scale_x']
    py = transform_params['center_y'] + ry * transform_params['scale_y']
    pz = transform_params['Z_pos']
    return px, py, pz

def build_all_light_sources(W, L, H, cob_lumen_params, strip_module_lumen_params):
    """Builds COB and Strip light source positions and lumens based on layer parameters."""
    # ... (body identical to v3.1) ...
    num_total_layers_cob = len(cob_lumen_params)
    num_total_layers_strip = len(strip_module_lumen_params)
    if num_total_layers_cob != num_total_layers_strip:
        raise ValueError(f"Lumen parameter array lengths mismatch: COB ({num_total_layers_cob}) vs Strip ({num_total_layers_strip})")
    num_total_layers = num_total_layers_cob
    if num_total_layers == 0 :
        print("[Warning] Zero layers specified for light sources.")
        return np.empty((0,3)), np.empty((0,)), np.empty((0,), dtype=np.int32), np.empty((0,4)), {}

    abstract_cob_coords, transform_params = _get_cob_abstract_coords_and_transform(W, L, H, num_total_layers)

    all_positions_list = []
    all_lumens_list = []
    all_types_list = []
    cob_positions_only_list = []
    ordered_strip_vertices = {}

    print("[Geometry] Placing COBs...")
    cobs_placed_count = 0
    for i, abs_cob in enumerate(abstract_cob_coords):
        layer = abs_cob['layer']
        if not (0 <= layer < num_total_layers):
            print(f"[Warning] Abstract COB coord has invalid layer {layer}. Skipping.")
            continue
        px, py, pz = _apply_transform(abs_cob, transform_params)
        lumens = cob_lumen_params[layer]
        if lumens > EPSILON:
            all_positions_list.append([px, py, pz])
            all_lumens_list.append(lumens)
            all_types_list.append(EMITTER_TYPE_COB)
            cob_positions_only_list.append([px, py, pz, layer])
            cobs_placed_count += 1
    print(f"[INFO] Placed {cobs_placed_count} COBs.")

    print("[Geometry] Placing strip modules...")
    total_modules_placed = 0
    for layer_idx in range(1, num_total_layers):
        target_module_lumens = strip_module_lumen_params[layer_idx]
        if target_module_lumens <= EPSILON: continue

        layer_vertices_abstract = [p for p in abstract_cob_coords if p['layer'] == layer_idx and p['is_vertex']]
        if not layer_vertices_abstract: continue

        layer_vertices_abstract.sort(key=lambda p: math.atan2(p['y'], p['x']))
        transformed_vertices = [_apply_transform(av, transform_params) for av in layer_vertices_abstract]
        ordered_strip_vertices[layer_idx] = transformed_vertices

        num_vertices = len(transformed_vertices)
        modules_this_layer = 0
        for i in range(num_vertices):
            p1 = np.array(transformed_vertices[i])
            p2 = np.array(transformed_vertices[(i + 1) % num_vertices])
            direction_vec = p2 - p1
            section_length = np.linalg.norm(direction_vec)
            if section_length < STRIP_MODULE_LENGTH * 0.9: continue

            direction_unit = direction_vec / section_length
            num_modules = int(math.floor(section_length / STRIP_MODULE_LENGTH))
            if num_modules == 0: continue

            total_gap_length = section_length - num_modules * STRIP_MODULE_LENGTH
            gap_length = total_gap_length / (num_modules + 1)

            for j in range(num_modules):
                dist_to_module_center = gap_length * (j + 1) + STRIP_MODULE_LENGTH * (j + 0.5)
                module_pos = p1 + direction_unit * dist_to_module_center
                all_positions_list.append(module_pos.tolist())
                all_lumens_list.append(target_module_lumens)
                all_types_list.append(EMITTER_TYPE_STRIP)
                modules_this_layer += 1
        if modules_this_layer > 0:
            print(f"[INFO] Layer {layer_idx}: Placed {modules_this_layer} strip modules (Lumens per module={target_module_lumens:.1f}).")
            total_modules_placed += modules_this_layer

    try:
        if not all_positions_list:
            light_positions = np.empty((0, 3), dtype=np.float64)
            light_lumens = np.empty(0, dtype=np.float64)
            light_types = np.empty(0, dtype=np.int32)
        else:
            light_positions = np.array(all_positions_list, dtype=np.float64)
            light_lumens = np.array(all_lumens_list, dtype=np.float64)
            light_types = np.array(all_types_list, dtype=np.int32)
        cob_positions_only = np.array(cob_positions_only_list, dtype=np.float64) if cob_positions_only_list else np.empty((0, 4), dtype=np.float64)
    except Exception as e:
        print(f"[ERROR] Failed converting geometry lists to NumPy arrays: {e}")
        raise

    num_cobs_final = cob_positions_only.shape[0]
    num_strips_final = total_modules_placed
    print(f"[INFO] Total generated emitters: {num_cobs_final} COBs, {num_strips_final} strip modules.")
    return light_positions, light_lumens, light_types, cob_positions_only, ordered_strip_vertices

def prepare_geometry(W, L, H, cob_lumen_params, strip_module_lumen_params):
    """Prepares all geometry: lights, floor grid, patches (with subdivided floor)."""
    # ... (body identical to v3.1) ...
    print("[Geometry] Building light sources (COBs + Strip Modules)...")
    light_positions, light_lumens, light_types, cob_positions_only, ordered_strip_vertices = build_all_light_sources(
        W, L, H, cob_lumen_params, strip_module_lumen_params )
    print("[Geometry] Building fine floor grid for results...")
    X, Y = build_floor_grid(W, L)
    print("[Geometry] Building room patches (incl. subdivided floor) for radiosity...")
    patches_dict = build_patches(W, L, H)
    return light_positions, light_lumens, light_types, cob_positions_only, ordered_strip_vertices, X, Y, patches_dict


# ------------------------------------
# 6) Numba-JIT Computations (RESTORED @njit on compute_direct_floor)
# ------------------------------------
# Functions compute_patch_direct, compute_form_factor_mc_pair,
# compute_form_factor_matrix, iterative_radiosity_loop_ff, compute_row_reflection,
# compute_reflection_on_floor remain IDENTICAL to v3.1.

# --- v4.3: RESTORED @njit ---
@njit(parallel=True, cache=True)
def compute_direct_floor(light_positions, light_lumens, light_types, X, Y,
                         cob_v_angles, cob_h_angles, cob_candelas_2d, cob_norm,
                         strip_v_angles, strip_h_angles, strip_candelas_2d, strip_norm):
    """ Computes direct floor illuminance (Full IES) on the fine grid. """
    min_dist2 = (FLOOR_GRID_RES / 4.0) ** 2; rows, cols = X.shape
    out = np.zeros_like(X); num_lights = light_positions.shape[0]
    cob_ies_valid = cob_candelas_2d.ndim == 2 and cob_candelas_2d.size > 0
    strip_ies_valid = strip_candelas_2d.ndim == 2 and strip_candelas_2d.size > 0

    # --- Use prange for parallel execution ---
    for r in prange(rows): # Use prange here
        for c in range(cols):
            fx, fy, fz = X[r, c], Y[r, c], 0.0; val = 0.0
            for k in range(num_lights): # Iterate through each light source
                lx, ly, lz = light_positions[k, 0:3]; lumens_k = light_lumens[k]; type_k = light_types[k]
                if lumens_k < EPSILON: continue

                dx, dy, dz = fx - lx, fy - ly, fz - lz
                d2 = max(dx*dx + dy*dy + dz*dz, min_dist2); dist = math.sqrt(d2)

                cos_in_floor = -dz / dist
                if cos_in_floor < EPSILON: continue
                cos_in_floor = min(cos_in_floor, 1.0)

                I_val = 0.0
                # --- NO PRINTING INSIDE NJIT ---

                # Calculate intensity using IES data
                if type_k == EMITTER_TYPE_COB:
                    if cob_ies_valid:
                        I_val = calculate_ies_intensity_2d(dx, dy, dz, dist, lumens_k, cob_v_angles, cob_h_angles, cob_candelas_2d, cob_norm)
                elif type_k == EMITTER_TYPE_STRIP:
                     if strip_ies_valid:
                         I_val = calculate_ies_intensity_2d(dx, dy, dz, dist, lumens_k, strip_v_angles, strip_h_angles, strip_candelas_2d, strip_norm)

                if I_val > EPSILON:
                    val += (I_val / d2) * cos_in_floor

            out[r, c] = val
    return out

# ------------------------------------
# Optimized compute_patch_direct
# (Added parallel=True and prange)
# ------------------------------------
@njit(parallel=True, cache=True) # Added parallel=True
def compute_patch_direct(light_positions, light_lumens, light_types,
                         patch_centers, patch_normals, patch_areas,
                         cob_v_angles, cob_h_angles, cob_candelas_2d, cob_norm,
                         strip_v_angles, strip_h_angles, strip_candelas_2d, strip_norm):
    """ Computes direct patch illuminance (Full IES) for radiosity patches.
        Optimized with Numba parallelization.
    """
    min_dist2 = (FLOOR_GRID_RES / 4.0) ** 2; Np = patch_centers.shape[0]
    out = np.zeros(Np); num_lights = light_positions.shape[0]
    cob_ies_valid = cob_candelas_2d.ndim == 2 and cob_candelas_2d.size > 0
    strip_ies_valid = strip_candelas_2d.ndim == 2 and strip_candelas_2d.size > 0

    # Use prange for parallel execution over patches
    for ip in prange(Np): # Changed range to prange
        pc = patch_centers[ip]; n_patch = patch_normals[ip]
        norm_n_patch_val = np.linalg.norm(n_patch)
        if norm_n_patch_val < EPSILON: continue # Skip patches with zero normal

        n_patch_unit = n_patch / norm_n_patch_val; accum_E = 0.0

        # Loop through each light source for this patch
        for k in range(num_lights):
            lx, ly, lz = light_positions[k, 0:3]; lumens_k = light_lumens[k]; type_k = light_types[k]
            if lumens_k < EPSILON: continue # Skip lights with zero lumens

            dx, dy, dz = pc[0] - lx, pc[1] - ly, pc[2] - lz
            d2 = max(dx*dx + dy*dy + dz*dz, min_dist2); dist = math.sqrt(d2)

            # Vector from light source towards patch center
            light_to_patch_vec_unit = np.array((-dx/dist, -dy/dist, -dz/dist))

            # Cosine of the angle between patch normal and direction to light
            cos_in_patch = np.dot(n_patch_unit, light_to_patch_vec_unit)

            # Only consider light arriving at the front face of the patch
            if cos_in_patch < EPSILON: continue
            cos_in_patch = min(cos_in_patch, 1.0) # Clamp to 1

            I_val = 0.0
            # Calculate intensity based on emitter type and IES data
            if type_k == EMITTER_TYPE_COB:
                if cob_ies_valid:
                    I_val = calculate_ies_intensity_2d(dx, dy, dz, dist, lumens_k, cob_v_angles, cob_h_angles, cob_candelas_2d, cob_norm)
            elif type_k == EMITTER_TYPE_STRIP:
                 if strip_ies_valid:
                     I_val = calculate_ies_intensity_2d(dx, dy, dz, dist, lumens_k, strip_v_angles, strip_h_angles, strip_candelas_2d, strip_norm)

            # Add contribution to patch illuminance if intensity is positive
            if I_val > EPSILON:
                accum_E += (I_val / d2) * cos_in_patch

        # Store the total direct illuminance for the patch
        out[ip] = accum_E
    return out

@njit(cache=True)
def compute_form_factor_mc_pair(
    pj, Aj, nj_unit, t1j, t2j, hs1j, hs2j, # Source Patch j
    pi, Ai, ni_unit, t1i, t2i, hs1i, hs2i, # Destination Patch i
    num_samples):
    """ Computes form factor F_ji using Monte Carlo. """
    # ... (body identical to v3.1) ...
    if Aj <= EPSILON or Ai <= EPSILON or abs(np.linalg.norm(nj_unit) - 1.0) > EPSILON or abs(np.linalg.norm(ni_unit) - 1.0) > EPSILON:
        return 0.0
    sum_kernel = 0.0; valid_samples = max(1, num_samples)
    for _ in range(valid_samples):
        u1j = np.random.uniform(-hs1j, hs1j); u2j = np.random.uniform(-hs2j, hs2j)
        xj = pj + u1j * t1j + u2j * t2j
        u1i = np.random.uniform(-hs1i, hs1i); u2i = np.random.uniform(-hs2i, hs2i)
        xi = pi + u1i * t1i + u2i * t2i

        vij = xi - xj; dist2 = np.dot(vij, vij)
        if dist2 < EPSILON * EPSILON: continue

        dist = math.sqrt(dist2); vij_unit = vij / dist
        cos_j = np.dot(nj_unit, vij_unit)
        cos_i = np.dot(ni_unit, -vij_unit)

        if cos_j > EPSILON and cos_i > EPSILON:
            kernel = (cos_j * cos_i) / (math.pi * dist2)
            sum_kernel += kernel

    avg_kernel = sum_kernel / valid_samples
    form_factor_ji = avg_kernel * Ai # Note: Aj used in exitance later, F_ji = integral(kernel * dAi)
    return max(0.0, form_factor_ji)

# ------------------------------------
# Optimized compute_form_factor_matrix
# (Using symmetry A_j * F_ji = A_i * F_ij)
# ------------------------------------
@njit(parallel=True, cache=True)
def compute_form_factor_matrix(
    patch_centers, patch_areas, patch_normals, patch_t1, patch_t2, patch_hs1, patch_hs2,
    num_samples):
    """ Computes the full Np x Np form factor matrix F[j, i] = F_ji using MC.
        Optimized by calculating only the upper triangle + diagonal via MC
        and deriving the lower triangle using the reciprocity relationship.
    """
    Np = patch_centers.shape[0]
    form_factor_matrix = np.zeros((Np, Np), dtype=np.float64)

    # Pre-calculate unit normals
    normals_unit = np.empty_like(patch_normals)
    for k in range(Np):
        norm_n = np.linalg.norm(patch_normals[k])
        if norm_n > EPSILON:
            normals_unit[k] = patch_normals[k] / norm_n
        else:
            # Assign a default zero vector if normal is zero to avoid errors later
            # Patches with zero normals should also have zero area and be skipped
            normals_unit[k] = np.array((0.0, 0.0, 0.0))

    # Parallelize the outer loop over source patches 'j'
    for j in prange(Np):
        pj = patch_centers[j]; Aj = patch_areas[j]
        nj_unit = normals_unit[j]
        t1j = patch_t1[j]; t2j = patch_t2[j]
        hs1j = patch_hs1[j]; hs2j = patch_hs2[j]

        # Skip source patch if its area is negligible or normal is invalid
        if Aj <= EPSILON or np.linalg.norm(nj_unit) < EPSILON:
             continue

        # Inner loop computes F_ji only for i >= j (upper triangle including diagonal)
        for i in range(j, Np):
            # Diagonal form factor F_jj is 0 for planar/convex patches
            if i == j:
                form_factor_matrix[j, i] = 0.0
                continue

            pi = patch_centers[i]; Ai = patch_areas[i]
            ni_unit = normals_unit[i]
            t1i = patch_t1[i]; t2i = patch_t2[i]
            hs1i = patch_hs1[i]; hs2i = patch_hs2[i]

            # Skip destination patch if its area is negligible or normal is invalid
            if Ai <= EPSILON or np.linalg.norm(ni_unit) < EPSILON:
                 form_factor_matrix[j, i] = 0.0
                 # Also set the symmetric element F_ij to 0
                 form_factor_matrix[i, j] = 0.0
                 continue

            # Compute F_ji using Monte Carlo
            F_ji = compute_form_factor_mc_pair(
                pj, Aj, nj_unit, t1j, t2j, hs1j, hs2j, # Source j
                pi, Ai, ni_unit, t1i, t2i, hs1i, hs2i, # Destination i
                num_samples
            )
            form_factor_matrix[j, i] = F_ji

            # Apply reciprocity: F_ij = (Aj / Ai) * F_ji
            # Check Ai > EPSILON again for safety, though done above
            if Ai > EPSILON:
                F_ij = (Aj / Ai) * F_ji
                form_factor_matrix[i, j] = F_ij # Assign the symmetric counterpart
            else:
                 # Should not happen if check above works, but as safety
                 form_factor_matrix[i, j] = 0.0

    return form_factor_matrix

# ------------------------------------
# Optimized iterative_radiosity_loop_ff
# (Using np.dot for flux transfer)
# ------------------------------------
@njit(cache=True)
def iterative_radiosity_loop_ff(patch_direct, patch_areas, patch_refl, form_factor_matrix,
                                max_bounces, convergence_threshold):
    """ Calculates total incident irradiance using precomputed Form Factors F_ji.
        Optimized using np.dot for indirect flux calculation.
    """
    Np = patch_direct.shape[0]
    if Np == 0: return np.empty(0, dtype=np.float64)

    # Initialize total incident irradiance with direct component
    patch_incident_total = patch_direct.copy()
    # Calculate initial exitance (flux density leaving the surface)
    patch_exitance = patch_incident_total * patch_refl

    # --- Radiosity Iteration ---
    for bounce in range(max_bounces):
        # Store previous total incident for convergence check
        prev_patch_incident_total = patch_incident_total.copy()

        # --- Calculate total indirect flux arriving at each patch 'i' ---
        # Flux leaving patch j: Exit_Flux_j = B_j * A_j = patch_exitance[j] * patch_areas[j]
        exit_flux_j = patch_exitance * patch_areas

        # Flux arriving at patch i from all patches j: Sum_j [ Exit_Flux_j * F_ji ]
        # This is a matrix multiplication: exit_flux_j @ form_factor_matrix
        # where form_factor_matrix[j, i] = F_ji
        # Note: np.dot handles row vector * matrix correctly in Numba
        newly_incident_flux_indirect = np.dot(exit_flux_j, form_factor_matrix)

        # --- Update total incident irradiance and exitance for each patch ---
        max_rel_change = 0.0
        any_change = False
        for i in range(Np):
            # Convert incident flux to incident irradiance (flux per unit area)
            incident_irradiance_indirect_i = 0.0
            if patch_areas[i] > EPSILON:
                # Use the pre-calculated total indirect flux arriving at patch i
                incident_irradiance_indirect_i = newly_incident_flux_indirect[i] / patch_areas[i]

            # Calculate the *new* total incident irradiance = Direct + Indirect
            new_total_incident = patch_direct[i] + incident_irradiance_indirect_i
            old_total_incident = prev_patch_incident_total[i]

            # Update total incident and exitance for the *next* bounce
            patch_incident_total[i] = new_total_incident
            patch_exitance[i] = new_total_incident * patch_refl[i] # B = rho * E_inc_total

            # --- Check convergence for this patch ---
            if old_total_incident > EPSILON: # Avoid division by zero or large relative change for near-zero values
                change = abs(new_total_incident - old_total_incident)
                rel_change = change / old_total_incident
                if rel_change > max_rel_change:
                    max_rel_change = rel_change
                # Track if *any* patch changed significantly (more robust than just max_rel_change < threshold)
                if rel_change > convergence_threshold :
                    any_change = True

        # --- Overall convergence check for this bounce ---
        # If no patch had a relative change above threshold, converged.
        if not any_change:
            # Optional: Add print statement for debugging convergence
            # print(f"Radiosity converged after {bounce+1} bounces. Max rel change: {max_rel_change:.2e}")
            break
    # else: # Optional: Add print statement if max bounces reached
        # if bounce == max_bounces - 1:
            # print(f"Radiosity loop reached max {max_bounces} bounces. Max rel change: {max_rel_change:.2e}")

    # Return the final total incident irradiance on each patch
    return patch_incident_total


@njit(cache=True)
def compute_row_reflection(r, X, Y, patch_centers, patch_normals, patch_areas, patch_exitance, mc_samples):
    """ Computes indirect illuminance for a single floor row using MC,
        EXCLUDING floor patches as sources. """
    # ... (body identical to v3.1) ...
    row_vals = np.empty(X.shape[1]); Np = patch_centers.shape[0]
    valid_mc_samples = max(1, mc_samples)
    min_dist2_indirect = (FLOOR_GRID_RES / 4.0) ** 2 # Minimum distance clamp

    for c in range(X.shape[1]):
        fx, fy, fz = X[r, c], Y[r, c], 0.0; total_indirect_E = 0.0
        for p in range(Np):
            # Skip floor patches (normal ~ (0, 0, 1))
            if patch_normals[p, 2] > 0.9: continue

            exitance_p = patch_exitance[p]; area_p = patch_areas[p]
            if exitance_p * area_p <= EPSILON: continue

            pc = patch_centers[p]; n = patch_normals[p]
            norm_n_val = np.linalg.norm(n)
            if norm_n_val < EPSILON: continue
            n_unit = n / norm_n_val

            # Simplified tangents for sampling
            if abs(n_unit[0]) > 0.9: v_tmp = np.array((0.0, 1.0, 0.0))
            else: v_tmp = np.array((1.0, 0.0, 0.0))
            tangent1 = np.cross(n_unit, v_tmp); t1_norm = np.linalg.norm(tangent1)
            if t1_norm < EPSILON: tangent1 = np.cross(n_unit, np.array((0.0,0.0,1.0))); t1_norm = np.linalg.norm(tangent1)
            if t1_norm < EPSILON: tangent1 = np.array((1.0,0.0,0.0))
            else: tangent1 /= t1_norm
            tangent2 = np.cross(n_unit, tangent1) # Should be normalized

            half_side = math.sqrt(area_p) * 0.5
            sample_sum_integrand = 0.0
            for _ in range(valid_mc_samples):
                off1 = np.random.uniform(-half_side, half_side); off2 = np.random.uniform(-half_side, half_side)
                sample_point = pc + off1 * tangent1 + off2 * tangent2

                vec_sp_to_fp = np.array((fx - sample_point[0], fy - sample_point[1], fz - sample_point[2]))
                dist2 = np.dot(vec_sp_to_fp, vec_sp_to_fp)
                dist2 = max(dist2, min_dist2_indirect)

                dist = math.sqrt(dist2); vec_sp_to_fp_unit = vec_sp_to_fp / dist
                cos_f = -vec_sp_to_fp_unit[2]
                cos_p = np.dot(n_unit, vec_sp_to_fp_unit)

                if cos_f > EPSILON and cos_p > EPSILON and dist2 > EPSILON:
                   integrand_term = (cos_p * cos_f) / dist2
                   sample_sum_integrand += integrand_term

            avg_integrand = sample_sum_integrand / valid_mc_samples
            total_indirect_E += (area_p * exitance_p / math.pi) * avg_integrand

        row_vals[c] = total_indirect_E
    return row_vals

def compute_reflection_on_floor(X, Y, patch_centers, patch_normals, patch_areas, patch_rad, patch_refl, mc_samples=MC_SAMPLES):
    """ Calculates indirect floor illuminance using MC via parallel rows,
        implicitly excluding floor patches via compute_row_reflection. """
    # ... (body identical to v3.1) ...
    rows, cols = X.shape
    if rows == 0 or cols == 0: return np.zeros((rows, cols))
    patch_exitance = patch_rad * patch_refl
    results = Parallel(n_jobs=-1)(delayed(compute_row_reflection)(
        r, X, Y, patch_centers, patch_normals, patch_areas, patch_exitance, mc_samples
    ) for r in range(rows))
    out = np.zeros((rows, cols))
    if results and len(results) == rows:
        for r, row_vals in enumerate(results):
            if row_vals is not None and len(row_vals) == cols:
                out[r, :] = row_vals
            else:
                print(f"[Warning] Invalid results for reflection row {r}.")
    else:
        print("[Warning] Parallel reflection computation returned unexpected results structure.")
    return out


# ------------------------------------
# 7) Heatmap Plotting Function (Restore v4.8 - Corners Drawn First)
# ------------------------------------
def plot_heatmap(floor_ppfd, X, Y, cob_marker_positions, ordered_strip_vertices, W, L, annotation_step=10):
    """ Generates a heatmap of the floor PPFD.
        Restored v4.8 - Draws corners first, standard loop skips corner indices.
    """
    fig, ax = plt.subplots(figsize=(10, 8)); extent = [0, W, 0, L]
    vmin = np.min(floor_ppfd); vmax = np.max(floor_ppfd)
    if abs(vmin - vmax) < EPSILON: vmax = vmin + 1.0
    im = ax.imshow(floor_ppfd, cmap='viridis', interpolation='nearest', origin='lower', extent=extent, aspect='equal', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8); cbar.set_label('PPFD (µmol/m²/s)')
    rows, cols = floor_ppfd.shape

    # --- Standard black background for text boxes ---
    standard_bbox = dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.5)

    # --- Annotation ---
    if rows > 0 and cols > 0 and X.shape == floor_ppfd.shape and Y.shape == floor_ppfd.shape:
        # 1. ALWAYS annotate the four corners FIRST
        corner_indices = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
        corner_indices_set = set(corner_indices) # Use a set for fast checking later

        for r, c in corner_indices:
             text_x = X[r, c]; text_y = Y[r, c]
             ax.text(text_x, text_y, f"{floor_ppfd[r, c]:.1f}", ha="center", va="center", color="white", fontsize=6,
                     bbox=standard_bbox) # Use standard_bbox

        # 2. Annotate non-corner points based on step size and threshold
        if annotation_step > 0:
            annotation_thresh = vmin + 0.01 * (vmax - vmin) # Threshold for non-corner points
            step_r = max(1, rows // annotation_step)
            step_c = max(1, cols // annotation_step)

            for r in range(0, rows, step_r):
                 for c in range(0, cols, step_c):
                    # --- Skip if this index is one of the corners we already annotated ---
                    if (r, c) in corner_indices_set:
                        continue
                    # --- End Skip ---

                    # Annotate if above threshold (and not a corner)
                    if floor_ppfd[r, c] > annotation_thresh:
                        text_x = X[r, c]; text_y = Y[r, c]
                        ax.text(text_x, text_y, f"{floor_ppfd[r, c]:.1f}", ha="center", va="center", color="white", fontsize=6,
                                bbox=standard_bbox) # Use standard_bbox
        # --- End Step-based Annotation ---

    elif annotation_step > 0: print("[Plot Warning] Skipping annotations due to grid mismatch or zero steps.")

    # --- Plot Strip/COB markers (identical) ---
    # ... (rest of plotting code is identical to v4.8) ...
    strip_handles = []
    cmap_strips = plt.cm.cool; num_strip_layers = len(ordered_strip_vertices)
    colors_strips = cmap_strips(np.linspace(0, 0.9, max(1, num_strip_layers)))
    for layer_idx, vertices in ordered_strip_vertices.items():
        if not vertices: continue
        num_vertices = len(vertices)
        if layer_idx-1 < len(colors_strips): strip_color = colors_strips[layer_idx - 1]
        else: strip_color = 'gray'
        for i in range(num_vertices):
            p1, p2 = vertices[i], vertices[(i + 1) % num_vertices]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=strip_color, linewidth=2.0, linestyle='--', alpha=0.7, zorder=2)
        if num_vertices > 0:
            label = f'Strip Layer {int(layer_idx)}'
            if label not in [h.get_label() for h in strip_handles]:
                 strip_handles.append(plt.Line2D([0], [0], color=strip_color, lw=2.0, linestyle='--', label=label))

    if cob_marker_positions is not None and cob_marker_positions.shape[0] > 0:
        ax.scatter(cob_marker_positions[:, 0], cob_marker_positions[:, 1], marker='o',
                   color='red', edgecolors='black', s=50, label="Main COB positions", alpha=0.8, zorder=3)

    # Update title to v4.9
    ax.set_title("Floor PPFD (v4.9 - Safe Interp / Corners First)"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    handles, labels = ax.get_legend_handles_labels()
    all_handles = handles + strip_handles
    if all_handles: ax.legend(handles=all_handles, fontsize='small', loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, linestyle=':', alpha=0.5); plt.tight_layout(rect=[0, 0, 0.85, 1])

# ------------------------------------
# 8) CSV Output Function (Unchanged from v3.1)
# ------------------------------------
def write_ppfd_to_csv(filename, floor_ppfd, X, Y, cob_positions, W, L):
    """ Writes PPFD data categorized by distance-based layers to a CSV file. """
    # ... (body identical to v3.1) ...
    if floor_ppfd.size == 0: print("[CSV Warning] Empty PPFD data. Skipping CSV."); return

    layer_data = {i: [] for i in range(FIXED_NUM_LAYERS)}
    center_x, center_y = W / 2, L / 2
    max_dist_per_layer = np.zeros(FIXED_NUM_LAYERS)
    if cob_positions is not None and cob_positions.shape[0] > 0 and cob_positions.shape[1] >= 4:
        cob_layers = cob_positions[:, 3].astype(int)
        distances = np.sqrt((cob_positions[:, 0] - center_x)**2 + (cob_positions[:, 1] - center_y)**2)
        for i in range(FIXED_NUM_LAYERS):
            mask = (cob_layers == i)
            if np.any(mask): max_dist_per_layer[i] = np.max(distances[mask])
            elif i > 0:
                found_prev = False
                for prev_i in range(i - 1, -1, -1):
                    if max_dist_per_layer[prev_i] > EPSILON:
                        max_dist_per_layer[i] = max_dist_per_layer[prev_i]; found_prev = True; break
                if not found_prev: max_dist_per_layer[i] = max_dist_per_layer[0]
    else:
        print("[CSV Warning] No COB positions provided. Using linear distance spacing for layers.")
        max_room_dist = np.sqrt((W/2)**2 + (L/2)**2)
        max_dist_per_layer = np.linspace(max_room_dist/max(1, FIXED_NUM_LAYERS), max_room_dist, FIXED_NUM_LAYERS) if FIXED_NUM_LAYERS > 0 else np.array([])

    layer_radii_outer = np.sort(np.unique(max_dist_per_layer[max_dist_per_layer > EPSILON]))
    if len(layer_radii_outer) < FIXED_NUM_LAYERS:
        if len(layer_radii_outer) > 0:
            last_radius = layer_radii_outer[-1]
            padding_needed = FIXED_NUM_LAYERS - len(layer_radii_outer)
            padding_step = max(FLOOR_GRID_RES, last_radius * 0.05 / max(1, padding_needed))
            padding = last_radius + np.cumsum(np.full(padding_needed, padding_step))
            layer_radii_outer = np.concatenate((layer_radii_outer, padding))
        elif FIXED_NUM_LAYERS > 0:
            print("[CSV Warning] No valid COB distances found. Reverting to linear spacing.")
            max_room_dist = np.sqrt((W/2)**2 + (L/2)**2)
            layer_radii_outer = np.linspace(max_room_dist/FIXED_NUM_LAYERS, max_room_dist, FIXED_NUM_LAYERS)
        else: layer_radii_outer = np.array([])

    min_sep = FLOOR_GRID_RES / 10.0
    for i in range(len(layer_radii_outer)):
        if i == 0: layer_radii_outer[i] = max(layer_radii_outer[i], min_sep)
        else: layer_radii_outer[i] = max(layer_radii_outer[i], layer_radii_outer[i-1] + min_sep)
    if len(layer_radii_outer) > 0: layer_radii_outer[-1] += min_sep * 0.5

    rows, cols = floor_ppfd.shape
    points_assigned = 0
    if X.shape == floor_ppfd.shape and Y.shape == floor_ppfd.shape and len(layer_radii_outer) == FIXED_NUM_LAYERS:
        print(f"[CSV] Using layer radii (outer bounds): {[f'{r:.3f}' for r in layer_radii_outer]}")
        for r in range(rows):
            for c in range(cols):
                fx, fy = X[r, c], Y[r, c]
                dist = np.sqrt((fx - center_x)**2 + (fy - center_y)**2)
                assigned_layer = -1
                for i in range(FIXED_NUM_LAYERS):
                    outer = layer_radii_outer[i]
                    inner = layer_radii_outer[i-1] if i > 0 else 0.0
                    if dist > inner - EPSILON and dist <= outer + EPSILON:
                        assigned_layer = i; break
                if assigned_layer == -1 and FIXED_NUM_LAYERS > 0 and dist > layer_radii_outer[-1] and dist < layer_radii_outer[-1] * 1.01:
                     assigned_layer = FIXED_NUM_LAYERS - 1
                if assigned_layer != -1:
                    layer_data[assigned_layer].append(floor_ppfd[r, c])
                    points_assigned += 1
        print(f"[CSV] Assigned {points_assigned}/{floor_ppfd.size} grid points to layers.")
        unassigned_count = floor_ppfd.size - points_assigned
        if unassigned_count > 0: print(f"[CSV Warning] {unassigned_count} grid points were not assigned to any layer.")
    else: print("[CSV Warning] Grid shape mismatch or invalid layer radii. Cannot assign points to layers.")

    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Layer', 'PPFD'])
            for layer_index in sorted(layer_data.keys()):
                if layer_data[layer_index]:
                    for ppfd_value in layer_data[layer_index]:
                        writer.writerow([layer_index, ppfd_value])
    except IOError as e: print(f"Error writing CSV {filename}: {e}")


# ------------------------------------
# 9) Main Simulation Function (Unchanged from v3.1)
# ------------------------------------
def simulate_lighting(light_positions, light_lumens, light_types, X, Y, patches_dict):
    """ Runs the full lighting simulation pipeline. """
    # ... (body identical to v3.1) ...
    if X is None or Y is None or X.size == 0 or Y.size == 0:
        print("[Error] Invalid floor grid (X or Y). Cannot simulate.")
        return None

    if not isinstance(patches_dict, dict) or not patches_dict.get('center', np.array([])).shape[0]:
        print("[Error] No valid patches available for simulation. Returning zero PPFD.")
        return np.zeros_like(X, dtype=np.float64)

    p_centers = patches_dict.get('center', np.empty((0,3), dtype=np.float64))
    p_areas   = patches_dict.get('area',   np.empty(0, dtype=np.float64))
    p_normals = patches_dict.get('normal', np.empty((0,3), dtype=np.float64))
    p_refl    = patches_dict.get('refl',   np.empty(0, dtype=np.float64))
    p_t1      = patches_dict.get('t1',     np.empty((0,3), dtype=np.float64))
    p_t2      = patches_dict.get('t2',     np.empty((0,3), dtype=np.float64))
    p_hs1     = patches_dict.get('hs1',    np.empty(0, dtype=np.float64))
    p_hs2     = patches_dict.get('hs2',    np.empty(0, dtype=np.float64))

    if p_centers.shape[0] == 0 or p_areas.shape[0] == 0 or p_normals.shape[0] == 0:
         print("[Error] Essential patch arrays (centers, areas, normals) are empty. Cannot simulate.")
         return np.zeros_like(X, dtype=np.float64)
    num_patches = p_centers.shape[0]

    if light_positions.shape[0] == 0:
         print("[Warning] No light sources defined. Direct illumination will be zero.")
         floor_lux_direct = np.zeros_like(X, dtype=np.float64)
         patch_direct_lux = np.zeros(num_patches, dtype=np.float64)
    else:
        print("[Simulation] Calculating direct floor illuminance (Full IES)...")
        start_direct_floor_time = time.time()
        floor_lux_direct = compute_direct_floor(
            light_positions, light_lumens, light_types, X, Y,
            COB_IES_V_ANGLES, COB_IES_H_ANGLES, COB_IES_CANDELAS_2D, COB_IES_FILE_LUMENS_NORM,
            STRIP_IES_V_ANGLES, STRIP_IES_H_ANGLES, STRIP_IES_CANDELAS_2D, STRIP_IES_FILE_LUMENS_NORM
        )
        end_direct_floor_time = time.time()
        print(f"  > Direct floor finished in {end_direct_floor_time - start_direct_floor_time:.2f} seconds.")

        print("[Simulation] Calculating direct patch illuminance (Full IES)...")
        start_direct_patch_time = time.time()
        patch_direct_lux = compute_patch_direct(
            light_positions, light_lumens, light_types,
            p_centers, p_normals, p_areas,
            COB_IES_V_ANGLES, COB_IES_H_ANGLES, COB_IES_CANDELAS_2D, COB_IES_FILE_LUMENS_NORM,
            STRIP_IES_V_ANGLES, STRIP_IES_H_ANGLES, STRIP_IES_CANDELAS_2D, STRIP_IES_FILE_LUMENS_NORM
        )
        end_direct_patch_time = time.time()
        print(f"  > Direct patch finished in {end_direct_patch_time - start_direct_patch_time:.2f} seconds.")

    floor_lux_direct = np.nan_to_num(floor_lux_direct, nan=0.0)
    patch_direct_lux = np.nan_to_num(patch_direct_lux, nan=0.0)

    print(f"[Simulation] Pre-calculating Form Factor matrix ({num_patches}x{num_patches}) using MC ({N_FF_SAMPLES} samples)...")
    start_ff_time = time.time()
    form_factor_matrix = compute_form_factor_matrix(
        p_centers, p_areas, p_normals, p_t1, p_t2, p_hs1, p_hs2,
        N_FF_SAMPLES
    )
    end_ff_time = time.time()
    print(f"  > Form Factor matrix calculation finished in {end_ff_time - start_ff_time:.2f} seconds.")

    if form_factor_matrix is None or np.any(np.isnan(form_factor_matrix)):
        print("[ERROR] NaN or None detected in Form Factor matrix! Check MC sampling or geometry. Aborting.")
        return np.zeros_like(X, dtype=np.float64)

    print("[Simulation] Running radiosity using precomputed Form Factors...")
    start_rad_time = time.time()
    patch_total_incident_lux = iterative_radiosity_loop_ff(
        patch_direct_lux, p_areas, p_refl, form_factor_matrix,
        MAX_RADIOSITY_BOUNCES, RADIOSITY_CONVERGENCE_THRESHOLD
    )
    end_rad_time = time.time()
    print(f"  > Radiosity loop finished in {end_rad_time - start_rad_time:.2f} seconds.")
    patch_total_incident_lux = np.nan_to_num(patch_total_incident_lux, nan=0.0)

    print("[Simulation] Calculating indirect floor illuminance (Monte Carlo)...")
    start_indirect_time = time.time()
    floor_lux_indirect = compute_reflection_on_floor(
        X, Y, p_centers, p_normals, p_areas, patch_total_incident_lux, p_refl, MC_SAMPLES
    )
    end_indirect_time = time.time()
    print(f"  > Indirect floor calculation finished in {end_indirect_time - start_indirect_time:.2f} seconds.")
    floor_lux_indirect = np.nan_to_num(floor_lux_indirect, nan=0.0)

    total_floor_lux = floor_lux_direct + floor_lux_indirect
    effic = max(LUMINOUS_EFFICACY, EPSILON)
    total_radiant_Wm2 = total_floor_lux / effic
    floor_ppfd = total_radiant_Wm2 * CONVERSION_FACTOR
    floor_ppfd = np.nan_to_num(floor_ppfd, nan=0.0, posinf=0.0, neginf=0.0)

    return floor_ppfd


# ------------------------------------
# 10) Execution Block (Modified for Lumen Totals)
# ------------------------------------
def main():
    # --- Simulation-Specific Parameters ---
    W = 6.096; L = 6.096; H = 0.9144 # Approx 20x20x3 ft
    global FIXED_NUM_LAYERS; FIXED_NUM_LAYERS = 10 # Must match param array lengths below

    # --- Lumen Definitions (Same as 4.0) ---
    if FIXED_NUM_LAYERS != 10:
         print(f"[Warning] FIXED_NUM_LAYERS is {FIXED_NUM_LAYERS}, not 10. Using generated linear lumen profiles.")
         # ... (default lumen generation logic remains the same) ...
         default_cob_lumens = np.linspace(10000, 5000, FIXED_NUM_LAYERS)
         default_strip_lumens = np.concatenate(([0.0], np.linspace(1000, 8000, max(0, FIXED_NUM_LAYERS - 1))))
         if len(default_strip_lumens) < FIXED_NUM_LAYERS:
             default_strip_lumens = np.pad(default_strip_lumens, (0, FIXED_NUM_LAYERS - len(default_strip_lumens)), mode='edge')
         cob_lumen_per_layer = default_cob_lumens
         strip_lumen_per_module_per_layer = default_strip_lumens
    else:
        cob_lumen_per_layer = np.array([
             2500.0,  # Layer 0
             7000.0,  # Layer 1
             4500.0,  # Layer 2
             6500.0,  # Layer 3
             7000.0,  # Layer 4
             5000.0,  # Layer 5
             4000.0,  # Layer 6
             2000.0,  # Layer 7
             8000.0,  # Layer 8
             12000.0  # Layer 9
        ])
        strip_lumen_per_module_per_layer = np.array([
             2000.0,     # Layer 0
             2000.0,     # Layer 1
             2000.0,     # Layer 2
             0.0,     # Layer 3
             0.0,     # Layer 4
             0.0,     # Layer 5
             0.0,     # Layer 6
             0.0,     # Layer 7
             8000.0,  # Layer 8
             8000.0   # Layer 9
        ])

    if len(cob_lumen_per_layer) != FIXED_NUM_LAYERS:
        raise ValueError(f"Length of cob_lumen_per_layer ({len(cob_lumen_per_layer)}) does not match FIXED_NUM_LAYERS ({FIXED_NUM_LAYERS})")
    if len(strip_lumen_per_module_per_layer) != FIXED_NUM_LAYERS:
         raise ValueError(f"Length of strip_lumen_per_module_per_layer ({len(strip_lumen_per_module_per_layer)}) does not match FIXED_NUM_LAYERS ({FIXED_NUM_LAYERS})")

    print("[Config] COB Lumens per Layer:", [f"{l:.1f}" for l in cob_lumen_per_layer])
    print("[Config] Strip Module Lumens per Layer:", [f"{l:.1f}" for l in strip_lumen_per_module_per_layer])

    # --- Runtime Parameter Adjustments ---
    global N_FF_SAMPLES, MC_SAMPLES, FLOOR_SUBDIVS_X, FLOOR_SUBDIVS_Y
    parser = argparse.ArgumentParser(description="Lighting Simulation (v4.5 - Standard Annotation)")
    # ... (argparse setup remains the same) ...
    parser.add_argument('--no-plot', action='store_true', help='Disable the heatmap plot')
    parser.add_argument('--anno', type=int, default=15, help='Annotation density step (0 disables)')
    parser.add_argument('--ffsamples', type=int, default=N_FF_SAMPLES, help=f'MC samples for Form Factor calculation (default: {N_FF_SAMPLES})')
    parser.add_argument('--mcsamples', type=int, default=MC_SAMPLES, help=f'MC samples for indirect floor illumination (default: {MC_SAMPLES})')
    parser.add_argument('--floor_sub_x', type=int, default=FLOOR_SUBDIVS_X, help=f'Floor subdivisions X for radiosity (default: {FLOOR_SUBDIVS_X})')
    parser.add_argument('--floor_sub_y', type=int, default=FLOOR_SUBDIVS_Y, help=f'Floor subdivisions Y for radiosity (default: {FLOOR_SUBDIVS_Y})')

    args = parser.parse_args()
    N_FF_SAMPLES = args.ffsamples; MC_SAMPLES = args.mcsamples
    FLOOR_SUBDIVS_X = args.floor_sub_x; FLOOR_SUBDIVS_Y = args.floor_sub_y
    print(f"[Config] Using N_FF_SAMPLES = {N_FF_SAMPLES}, MC_SAMPLES = {MC_SAMPLES}")
    print(f"[Config] Radiosity Floor Subdivisions: {FLOOR_SUBDIVS_X}x{FLOOR_SUBDIVS_Y}")

    # --- Geometry ---
    print("Preparing geometry...")
    start_geom_time = time.time()
    light_positions, light_lumens, light_types, cob_positions_only, ordered_strip_vertices, X, Y, patches_dict = prepare_geometry(
        W, L, H, cob_lumen_per_layer, strip_lumen_per_module_per_layer
    )
    end_geom_time = time.time()
    total_lights = light_positions.shape[0]
    num_patches = patches_dict.get('center', np.array([])).shape[0]
    print(f"  > Geometry prepared in {end_geom_time - start_geom_time:.2f} seconds.")
    if num_patches == 0: print("Error: No patches generated."); return
    if total_lights == 0: print("Warning: No light sources generated.")

    # --- Simulation ---
    print(f"\nStarting simulation: {total_lights} emitters, {num_patches} patches...")
    start_sim_time = time.time()
    # --- Use the optimized simulation functions ---
    floor_ppfd = simulate_lighting(light_positions, light_lumens, light_types, X, Y, patches_dict)
    end_sim_time = time.time()
    print(f"\nTotal simulation time: {end_sim_time - start_sim_time:.2f} seconds.")

    if floor_ppfd is None or floor_ppfd.size == 0: print("Error: Simulation failed."); return

    # --- Corner Value Diagnostics ---
    # ... (corner diagnostics remain the same) ...
    print("\n--- Corner Value Diagnostics ---")
    if floor_ppfd.shape[0] > 0 and floor_ppfd.shape[1] > 0:
        bl_val = floor_ppfd[0, 0]    # Bottom-left (origin='lower')
        br_val = floor_ppfd[0, -1]   # Bottom-right
        tl_val = floor_ppfd[-1, 0]   # Top-left
        tr_val = floor_ppfd[-1, -1]  # Top-right
        vmin = np.min(floor_ppfd)
        vmax = np.max(floor_ppfd)
        annotation_thresh = vmin + 0.01 * (vmax - vmin)
        print(f"Bottom-Left (0,0):   {bl_val:.2f}")
        print(f"Bottom-Right (0,-1): {br_val:.2f}")
        print(f"Top-Left (-1,0):   {tl_val:.2f}")
        print(f"Top-Right (-1,-1):  {tr_val:.2f}")
        print(f"Annotation Threshold (> {annotation_thresh:.2f})")
        if bl_val <= annotation_thresh: print(">>> Bottom-Left value IS below annotation threshold.")
        if tl_val <= annotation_thresh: print(">>> Top-Left value IS below annotation threshold.")
    else: print("Could not retrieve corner values (invalid shape).")

    # --- Statistics & Output ---
    print("\nCalculating statistics...")
    mean_ppfd = np.mean(floor_ppfd); std_dev = np.std(floor_ppfd)
    min_ppfd = np.min(floor_ppfd); max_ppfd = np.max(floor_ppfd)
    mad = rmse = cv_percent = min_max_ratio = min_avg_ratio = cu_percent = dou_percent = 0.0
    if mean_ppfd > EPSILON:
        mad = np.mean(np.abs(floor_ppfd - mean_ppfd)); rmse = np.sqrt(np.mean((floor_ppfd - mean_ppfd)**2))
        cv_percent = (std_dev / mean_ppfd) * 100; min_max_ratio = min_ppfd / max_ppfd if max_ppfd > 0 else 0
        min_avg_ratio = min_ppfd / mean_ppfd; cu_percent = (1 - mad / mean_ppfd) * 100 if mean_ppfd > 0 else 0
        dou_percent = (1 - rmse / mean_ppfd) * 100 if mean_ppfd > 0 else 0

    # --- ADDED: Calculate total lumens per type ---
    total_cob_lumens = 0.0
    total_strip_lumens = 0.0
    if light_lumens.size > 0 and light_types.size == light_lumens.size:
        is_cob = (light_types == EMITTER_TYPE_COB)
        is_strip = (light_types == EMITTER_TYPE_STRIP)
        total_cob_lumens = np.sum(light_lumens[is_cob])
        total_strip_lumens = np.sum(light_lumens[is_strip])
    # --- END ADDED ---

    print(f"\n--- Results ---");
    print(f"Room: {W:.2f}x{L:.2f}x{H:.2f}m, Grid: {X.shape[1]}x{X.shape[0]} ({FLOOR_GRID_RES:.3f}m)")
    num_cobs = cob_positions_only.shape[0] # Count COBs from the dedicated array
    strip_emitters_count = total_lights - num_cobs # Derive strip count
    print(f"Emitters: {num_cobs} COBs, {strip_emitters_count} Strips ({total_lights} total)")
    # --- ADDED: Print total lumens ---
    print(f"Total COB Luminous Flux: {total_cob_lumens:.1f} lm")
    print(f"Total Strip Luminous Flux: {total_strip_lumens:.1f} lm")
    # --- END ADDED ---
    print(f"Patches: {num_patches} (Floor: {FLOOR_SUBDIVS_X}x{FLOOR_SUBDIVS_Y})")
    print(f"Avg PPFD: {mean_ppfd:.2f} µmol/m²/s | Std Dev: {std_dev:.2f} | Min: {min_ppfd:.2f} | Max: {max_ppfd:.2f}")
    print(f"RMSE: {rmse:.2f} | MAD: {mad:.2f} | CV: {cv_percent:.2f}% | Min/Max: {min_max_ratio:.3f} | Min/Avg: {min_avg_ratio:.3f}")
    print(f"CU (MAD): {cu_percent:.2f}% | DOU (RMSE): {dou_percent:.2f}%")

    # --- CSV ---
    csv_filename = "ppfd_layer_data.csv"
    print(f"\nWriting layer data to {csv_filename}...")
    write_ppfd_to_csv(csv_filename, floor_ppfd, X, Y, cob_positions_only, W, L)
    print("CSV write complete.")

    # --- Plotting ---
    if not args.no_plot:
        print("\nGenerating heatmap plot...")
        if np.all(np.isfinite(floor_ppfd)):
            plot_heatmap(floor_ppfd, X, Y, cob_positions_only, ordered_strip_vertices, W, L, annotation_step=args.anno)
            print("Plot window opened. Close plot window to exit.")
            plt.show()
        else: print("[Plot Error] Cannot plot non-finite PPFD data.")

if __name__ == "__main__":
    # ... (rest of the __main__ block remains the same) ...
    main()
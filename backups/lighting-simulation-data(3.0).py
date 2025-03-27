# --- START OF REVISED FILE lighting-simulation-data(3.0).py ---

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
# 1) Basic Config & Reflectances
# ------------------------------------
REFL_WALL = 0.8
REFL_CEIL = 0.8
REFL_FLOOR = 0.1

LUMINOUS_EFFICACY = 182.0  # lumens/W
SPD_FILE = "/Users/austinrouse/photonics/backups/spd_data.csv" # <<< SET PATH

MAX_RADIOSITY_BOUNCES = 10
RADIOSITY_CONVERGENCE_THRESHOLD = 1e-3

WALL_SUBDIVS_X = 10
WALL_SUBDIVS_Y = 5
CEIL_SUBDIVS_X = 10
CEIL_SUBDIVS_Y = 10

FLOOR_GRID_RES = 0.08  # m
FIXED_NUM_LAYERS = 10 # Tied to the structure of build_cob_positions and params length
MC_SAMPLES = 64 # Monte Carlo samples for *indirect floor* illumination
N_FF_SAMPLES = 128 # Monte Carlo samples for *Form Factor* calculation (NEW)

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
# 4.5) Load and Prepare Full IES Data (Unchanged from v2.0)
# ------------------------------------
# Functions parse_ies_file_full, compute_conversion_factor remain IDENTICAL

def parse_ies_file_full(ies_filepath):
    """ Parses an IESNA:LM-63-1995 file format (Full 2D). """
    print(f"[IES - Full] Attempting to parse file: {ies_filepath}")
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

# --- Load IES Data --- (Error handling within parse function)
print("\n--- Loading Strip IES Data ---")
(STRIP_IES_V_ANGLES, STRIP_IES_H_ANGLES, STRIP_IES_CANDELAS_2D, STRIP_IES_FILE_LUMENS_NORM) = parse_ies_file_full(STRIP_IES_FILE)
if STRIP_IES_V_ANGLES is None: raise SystemExit("Failed to load strip IES file.")
print("\n--- Loading COB IES Data ---")
(COB_IES_V_ANGLES, COB_IES_H_ANGLES, COB_IES_CANDELAS_2D, COB_IES_FILE_LUMENS_NORM) = parse_ies_file_full(COB_IES_FILE)
if COB_IES_V_ANGLES is None: raise SystemExit("Failed to load COB IES file.")

# --- SPD Conversion Factor --- (Unchanged from v2.0)
def compute_conversion_factor(spd_file):
    try:
        raw_spd = np.loadtxt(spd_file, delimiter=' ', skiprows=1)
        unique_wl, indices, counts = np.unique(raw_spd[:, 0], return_inverse=True, return_counts=True)
        counts_nonzero = np.maximum(counts, 1)
        avg_intens = np.bincount(indices, weights=raw_spd[:, 1]) / counts_nonzero
        spd = np.column_stack((unique_wl, avg_intens))
    except Exception as e: print(f"Error loading SPD: {e}"); return 0.0138
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
# 4.6) Numba-Compatible 2D Intensity Functions (Unchanged from v2.0)
# ------------------------------------
# Functions _interp_1d_linear_safe, interpolate_2d_bilinear, calculate_ies_intensity_2d remain IDENTICAL

@njit(cache=True)
def _interp_1d_linear_safe(x, xp, fp):
    """Safely interpolates 1D, clamping boundaries."""
    idx = np.searchsorted(xp, x, side='right')
    if idx == 0: return fp[0]
    if idx == len(xp): return fp[-1]
    x0, x1 = xp[idx-1], xp[idx]; y0, y1 = fp[idx-1], fp[idx]
    delta_x = x1 - x0
    if delta_x <= EPSILON: return y0
    weight = (x - x0) / delta_x
    return y0 * (1.0 - weight) + y1 * weight

@njit(cache=True)
def interpolate_2d_bilinear(angle_v_deg, angle_h_deg, ies_v_angles, ies_h_angles, ies_candelas_2d):
    """ Bilinear interpolation on 2D candela grid with angle wrapping/clamping. """
    num_v, num_h = ies_candelas_2d.shape
    if num_v == 1 and num_h == 1: return ies_candelas_2d[0, 0]
    if num_v == 1: return _interp_1d_linear_safe(angle_h_deg, ies_h_angles, ies_candelas_2d[0, :])
    if num_h == 1: return _interp_1d_linear_safe(angle_v_deg, ies_v_angles, ies_candelas_2d[:, 0])

    iv = max(0, min(np.searchsorted(ies_v_angles, angle_v_deg, side='right') - 1, num_v - 2))
    v0, v1 = ies_v_angles[iv], ies_v_angles[iv+1]; delta_v = v1 - v0
    tv = (angle_v_deg - v0) / delta_v if delta_v > EPSILON else (0.0 if angle_v_deg <= v0 else 1.0)
    tv = max(0.0, min(tv, 1.0))

    h_min, h_max = ies_h_angles[0], ies_h_angles[-1]; angle_h_wrapped = angle_h_deg
    h_range = h_max - h_min
    is_full_360 = abs(h_range - 360.0) < 5.0; is_symmetric_180 = abs(h_max - 180.0) < 5.0 and abs(h_min) < EPSILON
    is_symmetric_90 = abs(h_max - 90.0) < 5.0 and abs(h_min) < EPSILON
    if is_full_360: angle_h_wrapped = h_min + ((angle_h_deg - h_min) % 360.0); angle_h_wrapped = h_max if abs(angle_h_wrapped - (h_min + 360.0)) < EPSILON else angle_h_wrapped
    elif is_symmetric_180: angle_h_wrapped = angle_h_deg % 360.0; angle_h_wrapped = 360.0 - angle_h_wrapped if angle_h_wrapped > 180.0 else angle_h_wrapped
    elif is_symmetric_90: angle_h_wrapped = angle_h_deg % 360.0; angle_h_wrapped = 360.0 - angle_h_wrapped if angle_h_wrapped > 270.0 else (angle_h_wrapped - 180.0 if angle_h_wrapped > 180.0 else (180.0 - angle_h_wrapped if angle_h_wrapped > 90.0 else angle_h_wrapped)); angle_h_wrapped = max(0.0, min(angle_h_wrapped, 90.0))

    ih = max(0, min(np.searchsorted(ies_h_angles, angle_h_wrapped, side='right') - 1, num_h - 2))
    h0, h1 = ies_h_angles[ih], ies_h_angles[ih+1]; delta_h = h1 - h0
    th = (angle_h_wrapped - h0) / delta_h if delta_h > EPSILON else (0.0 if angle_h_wrapped <= h0 else 1.0)
    th = max(0.0, min(th, 1.0))

    C00 = ies_candelas_2d[iv, ih]; C10 = ies_candelas_2d[iv+1, ih]
    C01 = ies_candelas_2d[iv, ih+1]; C11 = ies_candelas_2d[iv+1, ih+1]
    C_h0 = C00 * (1.0 - tv) + C10 * tv; C_h1 = C01 * (1.0 - tv) + C11 * tv
    candela_raw = C_h0 * (1.0 - th) + C_h1 * th
    return max(0.0, candela_raw)

@njit(cache=True)
def calculate_ies_intensity_2d(dx, dy, dz, dist, total_emitter_lumens, ies_v_angles, ies_h_angles, ies_candelas_2d, ies_file_lumens_norm):
    """ Calculates luminous intensity (cd) using full 2D IES data. """
    candela_raw = 0.0
    if dist < EPSILON:
        zero_v_idx = np.searchsorted(ies_v_angles, 0.0); zero_h_idx = np.searchsorted(ies_h_angles, 0.0)
        if zero_v_idx < len(ies_v_angles) and abs(ies_v_angles[zero_v_idx]) < EPSILON and zero_h_idx < len(ies_h_angles) and abs(ies_h_angles[zero_h_idx]) < EPSILON:
            candela_raw = ies_candelas_2d[zero_v_idx, zero_h_idx]
        elif ies_candelas_2d.size > 0: candela_raw = ies_candelas_2d[0,0] # Fallback
    else:
        cos_theta_nadir = max(-1.0, min(1.0, -dz / dist))
        angle_v_deg = math.degrees(math.acos(cos_theta_nadir))
        angle_h_deg = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        candela_raw = interpolate_2d_bilinear(angle_v_deg, angle_h_deg, ies_v_angles, ies_h_angles, ies_candelas_2d)

    norm_factor = max(ies_file_lumens_norm, EPSILON)
    scaling_factor = total_emitter_lumens / norm_factor
    return candela_raw * scaling_factor


# ------------------------------------
# 5) Geometry Building (REVISED build_patches)
# ------------------------------------
# Functions _get_cob_abstract_coords_and_transform, _apply_transform,
# cached_build_floor_grid, build_floor_grid, build_all_light_sources, prepare_geometry
# remain IDENTICAL to version 2.0.

@lru_cache(maxsize=32)
def cached_build_floor_grid(W: float, L: float):
    num_cells_x = max(1, int(round(W / FLOOR_GRID_RES)))
    num_cells_y = max(1, int(round(L / FLOOR_GRID_RES)))
    actual_res_x = W / num_cells_x; actual_res_y = L / num_cells_y
    xs = np.linspace(actual_res_x / 2.0, W - actual_res_x / 2.0, num_cells_x)
    ys = np.linspace(actual_res_y / 2.0, L - actual_res_y / 2.0, num_cells_y)
    X, Y = np.meshgrid(xs, ys)
    print(f"[Grid] Centered grid created: {X.shape[1]}x{X.shape[0]} points.")
    return X, Y
def build_floor_grid(W, L): return cached_build_floor_grid(W, L)


# REVISED cached_build_patches to include basis vectors and dimensions
@lru_cache(maxsize=32)
def cached_build_patches(W: float, L: float, H: float):
    """ Builds patches and returns center, area, normal, reflectance,
        tangent1, tangent2, half_len1, half_len2 """
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

    # Floor (single patch)
    floor_area = W * L
    hs1_f, hs2_f = W / 2.0, L / 2.0
    t1_f, t2_f = np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0))
    add_patch((W/2, L/2, 0.0), floor_area, (0.0, 0.0, 1.0), REFL_FLOOR, t1_f, t2_f, hs1_f, hs2_f)

    # Ceiling Patches
    xs_c = np.linspace(0, W, CEIL_SUBDIVS_X + 1); ys_c = np.linspace(0, L, CEIL_SUBDIVS_Y + 1)
    dx_c = (xs_c[1]-xs_c[0]) if CEIL_SUBDIVS_X > 0 else W
    dy_c = (ys_c[1]-ys_c[0]) if CEIL_SUBDIVS_Y > 0 else L
    hs1_c, hs2_c = dx_c / 2.0, dy_c / 2.0
    t1_c, t2_c = np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0)) # Align with grid iter
    for i in range(CEIL_SUBDIVS_X):
        for j in range(CEIL_SUBDIVS_Y):
            cx = (xs_c[i] + xs_c[i+1]) / 2; cy = (ys_c[j] + ys_c[j+1]) / 2
            area = dx_c * dy_c
            add_patch((cx, cy, H), area, (0.0, 0.0, -1.0), REFL_CEIL, t1_c, t2_c, hs1_c, hs2_c)

    # Wall Patches
    wall_defs = [ # axis, fixed_val, normal, range1(W/L), range2(H), sub1, sub2, t1_axis, t2_axis
        ('y', 0.0, (0.0, 1.0, 0.0), (0, W), (0, H), WALL_SUBDIVS_X, WALL_SUBDIVS_Y, 'x', 'z'), # y=0
        ('y', L,   (0.0,-1.0, 0.0), (0, W), (0, H), WALL_SUBDIVS_X, WALL_SUBDIVS_Y, 'x', 'z'), # y=L
        ('x', 0.0, (1.0, 0.0, 0.0), (0, L), (0, H), WALL_SUBDIVS_X, WALL_SUBDIVS_Y, 'y', 'z'), # x=0
        ('x', W,  (-1.0, 0.0, 0.0), (0, L), (0, H), WALL_SUBDIVS_X, WALL_SUBDIVS_Y, 'y', 'z'), # x=W
    ]
    axis_vec = {'x': np.array((1.0, 0.0, 0.0)), 'y': np.array((0.0, 1.0, 0.0)), 'z': np.array((0.0, 0.0, 1.0))}

    for axis, fixed_val, normal, r1, r2, sub1, sub2, ax1, ax2 in wall_defs:
        s1 = max(1, sub1); s2 = max(1, sub2)
        c1 = np.linspace(r1[0], r1[1], s1 + 1); c2 = np.linspace(r2[0], r2[1], s2 + 1)
        dc1 = (c1[1]-c1[0]) if s1 > 0 else (r1[1]-r1[0])
        dc2 = (c2[1]-c2[0]) if s2 > 0 else (r2[1]-r2[0])
        hs1_w, hs2_w = dc1 / 2.0, dc2 / 2.0
        t1_w, t2_w = axis_vec[ax1], axis_vec[ax2]

        for i in range(s1):
            for j in range(s2):
                pc1 = (c1[i] + c1[i+1]) / 2; pc2 = (c2[j] + c2[j+1]) / 2
                area = dc1 * dc2
                center = (pc1, fixed_val, pc2) if axis == 'y' else (fixed_val, pc1, pc2)
                add_patch(center, area, normal, REFL_WALL, t1_w, t2_w, hs1_w, hs2_w)

    if not patches_data['center']:
         print("[ERROR] No patches generated.")
         # Return structure with empty arrays
         return {k: np.empty((0, 3) if k in ['center', 'normal', 't1', 't2'] else 0, dtype=np.float64) for k in patches_data}

    # Convert lists to numpy arrays
    return {k: np.array(v, dtype=np.float64) for k, v in patches_data.items()}

def build_patches(W, L, H): return cached_build_patches(W, L, H)

# --- COB/Strip Placement (Unchanged from v2.0) ---
def _get_cob_abstract_coords_and_transform(W, L, H, num_total_layers):
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
        if coord_tuple not in seen_coords: unique_positions.append(pos); seen_coords.add(coord_tuple)
    unique_positions.sort(key=lambda p: (p['layer'], math.atan2(p['y'], p['x']) if p['layer'] > 0 else 0))
    theta = math.radians(45); cos_t, sin_t = math.cos(theta), math.sin(theta)
    center_x, center_y = W / 2, L / 2
    scale_x = max(W / (n * math.sqrt(2)) if n > 0 else W / 2, EPSILON)
    scale_y = max(L / (n * math.sqrt(2)) if n > 0 else L / 2, EPSILON)
    transform_params = {'center_x': center_x, 'center_y': center_y, 'scale_x': scale_x, 'scale_y': scale_y, 'cos_t': cos_t, 'sin_t': sin_t, 'H': H, 'Z_pos': H * 0.95 }
    return unique_positions, transform_params
def _apply_transform(abstract_pos, transform_params):
    ax, ay = abstract_pos['x'], abstract_pos['y']
    rx = ax * transform_params['cos_t'] - ay * transform_params['sin_t']
    ry = ax * transform_params['sin_t'] + ay * transform_params['cos_t']
    px = transform_params['center_x'] + rx * transform_params['scale_x']
    py = transform_params['center_y'] + ry * transform_params['scale_y']
    pz = transform_params['Z_pos']
    return px, py, pz
def build_all_light_sources(W, L, H, cob_lumen_params, strip_module_lumen_params):
    num_total_layers_cob = len(cob_lumen_params); num_total_layers_strip = len(strip_module_lumen_params)
    if num_total_layers_cob != num_total_layers_strip: raise ValueError(f"Lumen param length mismatch")
    num_total_layers = num_total_layers_cob
    if num_total_layers == 0 : return np.empty((0,3)), np.empty((0,)), np.empty((0,)), np.empty((0,4)), {}
    abstract_cob_coords, transform_params = _get_cob_abstract_coords_and_transform(W, L, H, num_total_layers)
    all_positions_list = []; all_lumens_list = []; all_types_list = []
    cob_positions_only_list = []; ordered_strip_vertices = {}
    for i, abs_cob in enumerate(abstract_cob_coords):
        layer = abs_cob['layer']
        if not (0 <= layer < num_total_layers): continue
        px, py, pz = _apply_transform(abs_cob, transform_params); lumens = cob_lumen_params[layer]
        if lumens > EPSILON:
            all_positions_list.append([px, py, pz]); all_lumens_list.append(lumens); all_types_list.append(EMITTER_TYPE_COB)
            cob_positions_only_list.append([px, py, pz, layer])
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
        num_vertices = len(transformed_vertices); modules_this_layer = 0
        for i in range(num_vertices):
            p1 = np.array(transformed_vertices[i]); p2 = np.array(transformed_vertices[(i + 1) % num_vertices])
            direction_vec = p2 - p1; section_length = np.linalg.norm(direction_vec)
            if section_length < STRIP_MODULE_LENGTH * 0.9: continue
            direction_unit = direction_vec / section_length
            num_modules = int(math.floor(section_length / STRIP_MODULE_LENGTH));
            if num_modules == 0: continue
            total_gap_length = section_length - num_modules * STRIP_MODULE_LENGTH
            gap_length = total_gap_length / (num_modules + 1)
            for j in range(num_modules):
                dist_to_module_center = gap_length * (j + 1) + STRIP_MODULE_LENGTH * (j + 0.5)
                module_pos = p1 + direction_unit * dist_to_module_center
                all_positions_list.append(module_pos.tolist()); all_lumens_list.append(target_module_lumens)
                all_types_list.append(EMITTER_TYPE_STRIP); modules_this_layer += 1
        if modules_this_layer > 0: print(f"[INFO] Layer {layer_idx}: Placed {modules_this_layer} strip modules (Lumens={target_module_lumens:.1f})."); total_modules_placed += modules_this_layer
    try:
        if not all_positions_list: light_positions, light_lumens, light_types = np.empty((0, 3)), np.empty(0), np.empty(0, dtype=np.int32)
        else: light_positions = np.array(all_positions_list); light_lumens = np.array(all_lumens_list); light_types = np.array(all_types_list, dtype=np.int32)
        cob_positions_only = np.array(cob_positions_only_list) if cob_positions_only_list else np.empty((0, 4))
    except Exception as e: print(f"[ERROR] Failed converting geometry lists: {e}"); raise
    num_cobs_placed = cob_positions_only.shape[0]; num_strips_placed = total_modules_placed
    print(f"[INFO] Generated {num_cobs_placed} COBs. Placed {num_strips_placed} strip modules.")
    return light_positions, light_lumens, light_types, cob_positions_only, ordered_strip_vertices

def prepare_geometry(W, L, H, cob_lumen_params, strip_module_lumen_params):
    """Prepares all geometry: lights, floor grid, patches (with basis vectors)."""
    print("[Geometry] Building light sources (COBs + Strip Modules)...")
    light_positions, light_lumens, light_types, cob_positions_only, ordered_strip_vertices = build_all_light_sources(
        W, L, H, cob_lumen_params, strip_module_lumen_params )
    print("[Geometry] Building floor grid...")
    X, Y = build_floor_grid(W, L)
    print("[Geometry] Building room patches (incl. basis vectors)...")
    patches_dict = build_patches(W, L, H) # Returns dictionary of arrays
    return light_positions, light_lumens, light_types, cob_positions_only, ordered_strip_vertices, X, Y, patches_dict


# ------------------------------------
# 6) Numba-JIT Computations (REVISED Radiosity Loop, NEW Form Factor Calc)
# ------------------------------------

# --- Direct Illumination Kernels (Unchanged from v2.0) ---
@njit(parallel=True, cache=True)
def compute_direct_floor(light_positions, light_lumens, light_types, X, Y,
                         cob_v_angles, cob_h_angles, cob_candelas_2d, cob_norm,
                         strip_v_angles, strip_h_angles, strip_candelas_2d, strip_norm):
    """ Computes direct floor illuminance (Full IES). """
    min_dist2 = (FLOOR_GRID_RES / 4.0) ** 2; rows, cols = X.shape
    out = np.zeros_like(X); num_lights = light_positions.shape[0]
    cob_ies_valid = cob_candelas_2d.ndim == 2 and cob_candelas_2d.size > 0
    strip_ies_valid = strip_candelas_2d.ndim == 2 and strip_candelas_2d.size > 0
    for r in prange(rows):
        for c in range(cols):
            fx, fy, fz = X[r, c], Y[r, c], 0.0; val = 0.0
            for k in range(num_lights):
                lx, ly, lz = light_positions[k, 0:3]; lumens_k = light_lumens[k]; type_k = light_types[k]
                if lumens_k < EPSILON: continue
                dx, dy, dz = fx - lx, fy - ly, fz - lz; d2 = max(dx*dx + dy*dy + dz*dz, min_dist2); dist = math.sqrt(d2)
                cos_in_floor = -dz / dist
                if cos_in_floor < EPSILON: continue
                cos_in_floor = min(cos_in_floor, 1.0)
                I_val = 0.0
                if type_k == EMITTER_TYPE_COB:
                    if cob_ies_valid: I_val = calculate_ies_intensity_2d(dx, dy, dz, dist, lumens_k, cob_v_angles, cob_h_angles, cob_candelas_2d, cob_norm)
                elif type_k == EMITTER_TYPE_STRIP:
                     if strip_ies_valid: I_val = calculate_ies_intensity_2d(dx, dy, dz, dist, lumens_k, strip_v_angles, strip_h_angles, strip_candelas_2d, strip_norm)
                if I_val > EPSILON: val += (I_val / d2) * cos_in_floor
            out[r, c] = val
    return out

@njit(cache=True) # Parallel might be overkill here
def compute_patch_direct(light_positions, light_lumens, light_types,
                         patch_centers, patch_normals, patch_areas,
                         cob_v_angles, cob_h_angles, cob_candelas_2d, cob_norm,
                         strip_v_angles, strip_h_angles, strip_candelas_2d, strip_norm):
    """ Computes direct patch illuminance (Full IES). """
    min_dist2 = (FLOOR_GRID_RES / 4.0) ** 2; Np = patch_centers.shape[0]
    out = np.zeros(Np); num_lights = light_positions.shape[0]
    cob_ies_valid = cob_candelas_2d.ndim == 2 and cob_candelas_2d.size > 0
    strip_ies_valid = strip_candelas_2d.ndim == 2 and strip_candelas_2d.size > 0
    for ip in range(Np):
        pc = patch_centers[ip]; n_patch = patch_normals[ip]
        norm_n_patch_val = np.linalg.norm(n_patch)
        if norm_n_patch_val < EPSILON: continue
        n_patch_unit = n_patch / norm_n_patch_val; accum_E = 0.0
        for k in range(num_lights):
            lx, ly, lz = light_positions[k, 0:3]; lumens_k = light_lumens[k]; type_k = light_types[k]
            if lumens_k < EPSILON: continue
            dx, dy, dz = pc[0] - lx, pc[1] - ly, pc[2] - lz; d2 = max(dx*dx + dy*dy + dz*dz, min_dist2); dist = math.sqrt(d2)
            cos_in_patch = n_patch_unit[0]*(-dx/dist) + n_patch_unit[1]*(-dy/dist) + n_patch_unit[2]*(-dz/dist)
            if cos_in_patch < EPSILON: continue
            cos_in_patch = min(cos_in_patch, 1.0)
            I_val = 0.0
            if type_k == EMITTER_TYPE_COB:
                if cob_ies_valid: I_val = calculate_ies_intensity_2d(dx, dy, dz, dist, lumens_k, cob_v_angles, cob_h_angles, cob_candelas_2d, cob_norm)
            elif type_k == EMITTER_TYPE_STRIP:
                 if strip_ies_valid: I_val = calculate_ies_intensity_2d(dx, dy, dz, dist, lumens_k, strip_v_angles, strip_h_angles, strip_candelas_2d, strip_norm)
            if I_val > EPSILON: accum_E += (I_val / d2) * cos_in_patch
        out[ip] = accum_E
    return out


# --- NEW: Monte Carlo Form Factor Calculation ---
# (compute_form_factor_mc_pair remains unchanged)
@njit(cache=True)
def compute_form_factor_mc_pair(
    pj, Aj, nj_unit, t1j, t2j, hs1j, hs2j, # Source Patch j
    pi, Ai, ni_unit, t1i, t2i, hs1i, hs2i, # Destination Patch i
    num_samples):
    # ... (body is identical to version 3.0) ...
    if abs(np.linalg.norm(nj_unit) - 1.0) > EPSILON or abs(np.linalg.norm(ni_unit) - 1.0) > EPSILON:
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
    form_factor_ji = avg_kernel * Ai
    return max(0.0, form_factor_ji)


# MODIFIED Signature and body accesses
@njit(parallel=True, cache=True)
def compute_form_factor_matrix(
    patch_centers, patch_areas, patch_normals, patch_t1, patch_t2, patch_hs1, patch_hs2, # Separate arrays
    num_samples):
    """ Computes the full Np x Np form factor matrix F[j, i] = F_ji using MC. """
    Np = patch_centers.shape[0] # Use patch_centers to get Np
    form_factor_matrix = np.zeros((Np, Np), dtype=np.float64)

    # Pre-normalize all normals
    normals_unit = np.empty_like(patch_normals) # Use patch_normals
    for k in range(Np):
        norm_n = np.linalg.norm(patch_normals[k]) # Use patch_normals
        if norm_n > EPSILON:
            normals_unit[k] = patch_normals[k] / norm_n # Use patch_normals
        else:
            normals_unit[k] = np.array((0.0, 0.0, 0.0))

    # Parallel loop over source patches (j)
    for j in prange(Np):
        # Extract source patch j parameters directly from input arrays
        pj = patch_centers[j]; Aj = patch_areas[j]
        nj_unit = normals_unit[j]
        t1j = patch_t1[j]; t2j = patch_t2[j]
        hs1j = patch_hs1[j]; hs2j = patch_hs2[j]

        if Aj <= EPSILON or np.linalg.norm(nj_unit) < EPSILON: continue

        # Inner loop over destination patches (i)
        for i in range(Np):
            if i == j: continue

            # Extract destination patch i parameters directly from input arrays
            pi = patch_centers[i]; Ai = patch_areas[i]
            ni_unit = normals_unit[i]
            t1i = patch_t1[i]; t2i = patch_t2[i]
            hs1i = patch_hs1[i]; hs2i = patch_hs2[i]

            if Ai <= EPSILON or np.linalg.norm(ni_unit) < EPSILON: continue

            # Compute F_ji using the pair function
            F_ji = compute_form_factor_mc_pair(
                pj, Aj, nj_unit, t1j, t2j, hs1j, hs2j,
                pi, Ai, ni_unit, t1i, t2i, hs1i, hs2i,
                num_samples
            )
            form_factor_matrix[j, i] = F_ji

    return form_factor_matrix


# --- REVISED Radiosity Loop ---
@njit(cache=True)
def iterative_radiosity_loop_ff(patch_direct, patch_areas, patch_refl, form_factor_matrix,
                                max_bounces, convergence_threshold):
    """ Calculates total incident irradiance using precomputed Form Factors F_ji. """
    Np = patch_direct.shape[0]
    if Np == 0: return np.empty(0, dtype=np.float64)

    # Initialize total incident irradiance E_inc_total = E_direct
    patch_incident_total = patch_direct.copy()
    # Initial exitance B = rho * E_inc_total (assuming no emission E=0)
    patch_exitance = patch_incident_total * patch_refl

    for bounce in range(max_bounces):
        newly_incident_flux_indirect = np.zeros(Np, dtype=np.float64)
        prev_patch_incident_total = patch_incident_total.copy() # For convergence check

        # Calculate indirect flux transfer using the FF matrix
        # Loop through destination patches i
        for i in range(Np):
            flux_sum_to_i = 0.0
            # Sum contributions from all source patches j
            for j in range(Np):
                if i == j: continue
                # Flux arriving at i from j = B_j * A_j * F_ji
                # F_ji is stored in form_factor_matrix[j, i]
                flux_ji = patch_exitance[j] * patch_areas[j] * form_factor_matrix[j, i]
                # Accumulate positive flux contributions
                if flux_ji > EPSILON:
                    flux_sum_to_i += flux_ji
            newly_incident_flux_indirect[i] = flux_sum_to_i

        # --- Update Radiosity and Check Convergence ---
        max_rel_change = 0.0
        for i in range(Np):
            # New indirect incident irradiance on patch i
            incident_irradiance_indirect_i = 0.0
            if patch_areas[i] > EPSILON:
                incident_irradiance_indirect_i = newly_incident_flux_indirect[i] / patch_areas[i]

            # Update total incident irradiance: E_inc_total = E_direct + E_indirect
            patch_incident_total[i] = patch_direct[i] + incident_irradiance_indirect_i

            # Update exitance for the *next* bounce: B = rho * E_inc_total
            patch_exitance[i] = patch_incident_total[i] * patch_refl[i]

            # Check convergence based on relative change in total incident irradiance
            change = abs(patch_incident_total[i] - prev_patch_incident_total[i])
            denom = abs(prev_patch_incident_total[i]) + EPSILON
            rel_change = change / denom
            if rel_change > max_rel_change:
                max_rel_change = rel_change

        # Check for convergence
        if max_rel_change < convergence_threshold:
            # print(f"Radiosity converged after {bounce+1} bounces. Max change: {max_rel_change:.2e}")
            break
    # else: # Optional: print if max bounces reached
        # print(f"Radiosity loop reached max {max_bounces} bounces. Max change: {max_rel_change:.2e}")

    return patch_incident_total


# --- Monte Carlo Reflection onto Floor (Unchanged from v2.0) ---
# Functions compute_row_reflection, compute_reflection_on_floor remain IDENTICAL
@njit(cache=True)
def compute_row_reflection(r, X, Y, patch_centers, patch_normals, patch_areas, patch_exitance, mc_samples):
    """ Computes indirect illuminance for a single floor row using MC. """
    row_vals = np.empty(X.shape[1]); Np = patch_centers.shape[0]
    valid_mc_samples = max(1, mc_samples)
    # Define minimum distance squared within the function or pass it (using local definition for simplicity here)
    min_dist2_indirect = (FLOOR_GRID_RES / 4.0) ** 2 # Minimum distance clamp

    for c in range(X.shape[1]):
        fx, fy, fz = X[r, c], Y[r, c], 0.0; total_indirect_E = 0.0
        for p in range(Np):
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
            if t1_norm < EPSILON: tangent1 = np.array((1.0,0.0,0.0)) # Failsafe
            else: tangent1 /= t1_norm
            tangent2 = np.cross(n_unit, tangent1) # Should be normalized
            half_side = math.sqrt(area_p) * 0.5

            sample_sum_integrand = 0.0
            for _ in range(valid_mc_samples):
                off1 = np.random.uniform(-half_side, half_side); off2 = np.random.uniform(-half_side, half_side)
                sample_point = pc + off1 * tangent1 + off2 * tangent2
                vec_sp_to_fp = np.array((fx - sample_point[0], fy - sample_point[1], fz - sample_point[2]))
                dist2 = np.dot(vec_sp_to_fp, vec_sp_to_fp)

                # --- ADD THIS CLAMP ---
                dist2 = max(dist2, min_dist2_indirect)
                # --- END ADDITION ---

                # Continue if effectively zero distance was the only issue before clamping (optional, but safe)
                # if dist2 <= min_dist2_indirect + EPSILON and np.dot(vec_sp_to_fp, vec_sp_to_fp) < EPSILON * EPSILON:
                #    continue # Skip if the original distance was pathologically small

                dist = math.sqrt(dist2); vec_sp_to_fp_unit = vec_sp_to_fp / dist
                cos_f = -vec_sp_to_fp_unit[2] # Floor normal (0,0,1)
                cos_p = np.dot(n_unit, vec_sp_to_fp_unit) # Patch normal
                if cos_f > EPSILON and cos_p > EPSILON:
                    # Prevent division by zero if dist2 was exactly min_dist2_indirect and small
                    if dist2 > EPSILON: # Check dist2 again after clamping
                       integrand_term = (cos_p * cos_f) / dist2
                       sample_sum_integrand += integrand_term
                    # else: This case should ideally not happen if min_dist2_indirect is reasonably large

            avg_integrand = sample_sum_integrand / valid_mc_samples
            total_indirect_E += (area_p * exitance_p / math.pi) * avg_integrand
        row_vals[c] = total_indirect_E
    return row_vals

# Note: compute_reflection_on_floor doesn't need changes as it just calls compute_row_reflection.

def compute_reflection_on_floor(X, Y, patch_centers, patch_normals, patch_areas, patch_rad, patch_refl, mc_samples=MC_SAMPLES):
    """ Calculates indirect floor illuminance using MC via parallel rows. """
    rows, cols = X.shape
    if rows == 0 or cols == 0: return np.zeros((rows, cols))
    patch_exitance = patch_rad * patch_refl
    results = Parallel(n_jobs=-1)(delayed(compute_row_reflection)(
        r, X, Y, patch_centers, patch_normals, patch_areas, patch_exitance, mc_samples
    ) for r in range(rows))
    out = np.zeros((rows, cols))
    if results and len(results) == rows:
        for r, row_vals in enumerate(results):
            if row_vals is not None and len(row_vals) == cols: out[r, :] = row_vals
            else: print(f"[Warning] Invalid results for reflection row {r}.")
    else: print("[Warning] Parallel reflection computation returned unexpected results.")
    return out


# ------------------------------------
# 7) Heatmap Plotting Function (Unchanged from v2.0)
# ------------------------------------
# plot_heatmap function remains IDENTICAL
def plot_heatmap(floor_ppfd, X, Y, cob_marker_positions, ordered_strip_vertices, W, L, annotation_step=10):
    fig, ax = plt.subplots(figsize=(10, 8)); extent = [0, W, 0, L]
    vmin = np.min(floor_ppfd); vmax = np.max(floor_ppfd)
    if abs(vmin - vmax) < EPSILON: vmax = vmin + 1.0
    im = ax.imshow(floor_ppfd, cmap='viridis', interpolation='nearest', origin='lower', extent=extent, aspect='equal', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8); cbar.set_label('PPFD (µmol/m²/s)')
    rows, cols = floor_ppfd.shape
    if annotation_step > 0 and rows > 0 and cols > 0 and X.shape == floor_ppfd.shape and Y.shape == floor_ppfd.shape:
        step_r = max(1, rows // annotation_step); step_c = max(1, cols // annotation_step)
        for r in range(0, rows, step_r):
             for c in range(0, cols, step_c):
                text_x = X[r, c]; text_y = Y[r, c]
                ax.text(text_x, text_y, f"{floor_ppfd[r, c]:.1f}", ha="center", va="center", color="white", fontsize=6,
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.5))
    else: print("[Plot Warning] Skipping annotations due to grid mismatch or zero steps.")
    strip_handles = []
    cmap_strips = plt.cm.cool; num_strip_layers = len(ordered_strip_vertices)
    colors_strips = cmap_strips(np.linspace(0, 1, max(1, num_strip_layers)))
    for layer_idx, vertices in ordered_strip_vertices.items():
        if not vertices: continue
        num_vertices = len(vertices); strip_color = colors_strips[layer_idx - 1]
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
    ax.set_title("Floor PPFD (COBs + Strips - MC Form Factors)"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    handles, labels = ax.get_legend_handles_labels()
    all_handles = handles + strip_handles # Assumes strip handles are unique enough
    if all_handles: ax.legend(handles=all_handles, fontsize='small', loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, linestyle=':', alpha=0.5); plt.tight_layout(rect=[0, 0, 0.85, 1])

# ------------------------------------
# 8) CSV Output Function (Unchanged from v2.0)
# ------------------------------------
# write_ppfd_to_csv function remains IDENTICAL
def write_ppfd_to_csv(filename, floor_ppfd, X, Y, cob_positions, W, L):
    if floor_ppfd.size == 0: print("[CSV Warning] Empty PPFD data. Skipping CSV."); return
    layer_data = {i: [] for i in range(FIXED_NUM_LAYERS)}; center_x, center_y = W / 2, L / 2
    max_dist_per_layer = np.zeros(FIXED_NUM_LAYERS)
    if cob_positions is not None and cob_positions.shape[0] > 0 and cob_positions.shape[1] >= 4:
        cob_layers = cob_positions[:, 3].astype(int); distances = np.sqrt((cob_positions[:, 0] - center_x)**2 + (cob_positions[:, 1] - center_y)**2)
        for i in range(FIXED_NUM_LAYERS):
            mask = (cob_layers == i)
            if np.any(mask): max_dist_per_layer[i] = np.max(distances[mask])
            elif i > 0: max_dist_per_layer[i] = max_dist_per_layer[i-1]
    else: max_room_dist = np.sqrt((W/2)**2 + (L/2)**2); max_dist_per_layer = np.linspace(0, max_room_dist, FIXED_NUM_LAYERS + 1)[1:]
    layer_radii_outer = np.sort(np.unique(max_dist_per_layer))
    if len(layer_radii_outer) < FIXED_NUM_LAYERS and len(layer_radii_outer)>0: layer_radii_outer = np.pad(layer_radii_outer, (0, FIXED_NUM_LAYERS - len(layer_radii_outer)), mode='edge')
    elif len(layer_radii_outer) == 0: layer_radii_outer = np.linspace(FLOOR_GRID_RES/2.0, np.sqrt((W/2)**2+(L/2)**2), FIXED_NUM_LAYERS) if FIXED_NUM_LAYERS > 0 else np.array([])
    if FIXED_NUM_LAYERS > 1 and abs(layer_radii_outer[0]) < EPSILON: layer_radii_outer[0] = max(min(FLOOR_GRID_RES / 2.0, (layer_radii_outer[1] / 2.0 if len(layer_radii_outer)>1 else FLOOR_GRID_RES)), EPSILON)
    elif FIXED_NUM_LAYERS == 1: layer_radii_outer[0] = max(layer_radii_outer[0], FLOOR_GRID_RES / 2.0)
    if len(layer_radii_outer) > 0: layer_radii_outer[-1] += 0.01 * FLOOR_GRID_RES
    rows, cols = floor_ppfd.shape
    if X.shape == floor_ppfd.shape and Y.shape == floor_ppfd.shape and len(layer_radii_outer) == FIXED_NUM_LAYERS:
        for r in range(rows):
            for c in range(cols):
                fx, fy = X[r, c], Y[r, c]; dist = np.sqrt((fx - center_x)**2 + (fy - center_y)**2)
                assigned_layer = -1
                for i in range(FIXED_NUM_LAYERS):
                    outer = layer_radii_outer[i]; inner = layer_radii_outer[i-1] if i > 0 else 0.0
                    if (i == 0 and dist <= outer + EPSILON) or (i > 0 and inner < dist <= outer + EPSILON):
                        assigned_layer = i; break
                if assigned_layer == -1 and dist > layer_radii_outer[-1] and dist < layer_radii_outer[-1] * 1.05: assigned_layer = FIXED_NUM_LAYERS - 1
                if assigned_layer != -1: layer_data[assigned_layer].append(floor_ppfd[r, c])
    else: print("[CSV Warning] Grid/Layer mismatch. Cannot assign points.")
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile); writer.writerow(['Layer', 'PPFD'])
            for layer_index in sorted(layer_data.keys()):
                if layer_data[layer_index]:
                    for ppfd_value in layer_data[layer_index]: writer.writerow([layer_index, ppfd_value])
    except IOError as e: print(f"Error writing CSV {filename}: {e}")


# ------------------------------------
# 9) Main Simulation Function (REVISED Call Structure)
# ------------------------------------
def simulate_lighting(light_positions, light_lumens, light_types, X, Y, patches_dict):
    """ Runs the full lighting simulation pipeline with MC Form Factors. """

    # --- Setup & Input Validation ---
    # Check if X or Y grids are valid
    if X is None or Y is None or X.size == 0 or Y.size == 0:
        print("[Error] Invalid floor grid (X or Y). Cannot simulate.")
        return None # Return None to indicate failure

    # Extract patch data from dictionary and validate
    if not isinstance(patches_dict, dict) or not patches_dict.get('center', np.array([])).shape[0]:
        print("[Error] No valid patches available for simulation. Returning zero PPFD.")
        return np.zeros_like(X, dtype=np.float64)

    # Safely get patch arrays, provide defaults for safety if keys missing (though build_patches should ensure they exist)
    p_centers = patches_dict.get('center', np.empty((0,3), dtype=np.float64))
    p_areas   = patches_dict.get('area',   np.empty(0, dtype=np.float64))
    p_normals = patches_dict.get('normal', np.empty((0,3), dtype=np.float64))
    p_refl    = patches_dict.get('refl',   np.empty(0, dtype=np.float64))
    p_t1      = patches_dict.get('t1',     np.empty((0,3), dtype=np.float64))
    p_t2      = patches_dict.get('t2',     np.empty((0,3), dtype=np.float64))
    p_hs1     = patches_dict.get('hs1',    np.empty(0, dtype=np.float64))
    p_hs2     = patches_dict.get('hs2',    np.empty(0, dtype=np.float64))

    # Further check if essential arrays are populated
    if p_centers.shape[0] == 0 or p_areas.shape[0] == 0 or p_normals.shape[0] == 0:
         print("[Error] Essential patch arrays (centers, areas, normals) are empty. Cannot simulate.")
         return np.zeros_like(X, dtype=np.float64)

    num_patches = p_centers.shape[0]

    # --- Direct Illumination Calculation ---
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
            p_centers, p_normals, p_areas, # Pass arrays extracted earlier
            COB_IES_V_ANGLES, COB_IES_H_ANGLES, COB_IES_CANDELAS_2D, COB_IES_FILE_LUMENS_NORM,
            STRIP_IES_V_ANGLES, STRIP_IES_H_ANGLES, STRIP_IES_CANDELAS_2D, STRIP_IES_FILE_LUMENS_NORM
        )
        end_direct_patch_time = time.time()
        print(f"  > Direct patch finished in {end_direct_patch_time - start_direct_patch_time:.2f} seconds.")

    # Handle potential NaNs from direct calculations
    if np.any(np.isnan(floor_lux_direct)) or np.any(np.isnan(patch_direct_lux)):
        print("[Warning] NaN detected after direct illumination. Replacing with 0.")
        floor_lux_direct = np.nan_to_num(floor_lux_direct, nan=0.0)
        patch_direct_lux = np.nan_to_num(patch_direct_lux, nan=0.0)


    # --- Form Factor Calculation ---
    print(f"[Simulation] Pre-calculating Form Factor matrix ({num_patches}x{num_patches}) using MC ({N_FF_SAMPLES} samples)...")
    start_ff_time = time.time()
    # Pass individual NumPy arrays required by the Numba function
    form_factor_matrix = compute_form_factor_matrix(
        p_centers, p_areas, p_normals, p_t1, p_t2, p_hs1, p_hs2, # Pass required arrays
        N_FF_SAMPLES
    )
    end_ff_time = time.time()
    print(f"  > Form Factor matrix calculation finished in {end_ff_time - start_ff_time:.2f} seconds.")

    # Sanity check the computed Form Factor matrix
    if form_factor_matrix is None or np.any(np.isnan(form_factor_matrix)):
        print("[ERROR] NaN or None detected in Form Factor matrix! Check MC sampling or geometry. Aborting.")
        # Return zeros or None to indicate failure clearly
        return np.zeros_like(X, dtype=np.float64)


    # --- Radiosity Calculation ---
    print("[Simulation] Running radiosity using precomputed Form Factors...")
    start_rad_time = time.time()
    # Call the radiosity loop using the computed form factors
    patch_total_incident_lux = iterative_radiosity_loop_ff(
        patch_direct_lux, p_areas, p_refl, form_factor_matrix, # Pass necessary arrays
        MAX_RADIOSITY_BOUNCES, RADIOSITY_CONVERGENCE_THRESHOLD
    )
    end_rad_time = time.time()
    print(f"  > Radiosity loop finished in {end_rad_time - start_rad_time:.2f} seconds.")

    # Handle potential NaNs from radiosity
    if np.any(np.isnan(patch_total_incident_lux)):
        print("[Warning] NaN detected after radiosity calculation. Replacing with 0.")
        patch_total_incident_lux = np.nan_to_num(patch_total_incident_lux, nan=0.0)


    # --- Indirect Floor Illumination Calculation ---
    print("[Simulation] Calculating indirect floor illuminance (Monte Carlo)...")
    start_indirect_time = time.time()
    # Call the reflection calculation using results from radiosity
    floor_lux_indirect = compute_reflection_on_floor(
        X, Y, p_centers, p_normals, p_areas, patch_total_incident_lux, p_refl, MC_SAMPLES
    )
    end_indirect_time = time.time()
    print(f"  > Indirect floor calculation finished in {end_indirect_time - start_indirect_time:.2f} seconds.")

    # Handle potential NaNs from indirect calculation
    if np.any(np.isnan(floor_lux_indirect)):
        print("[Warning] NaN detected after indirect reflection calculation. Replacing with 0.")
        floor_lux_indirect = np.nan_to_num(floor_lux_indirect, nan=0.0)


    # --- Combine Results and Convert to PPFD ---
    total_floor_lux = floor_lux_direct + floor_lux_indirect

    # Ensure luminous efficacy is positive
    effic = max(LUMINOUS_EFFICACY, EPSILON)

    # Convert total lux (lm/m^2) to radiant flux density (W/m^2)
    total_radiant_Wm2 = total_floor_lux / effic

    # Convert radiant flux density (W/m^2) to PPFD (µmol/m²/s)
    floor_ppfd = total_radiant_Wm2 * CONVERSION_FACTOR

    # Final check for non-finite values (NaN, Inf) in the result
    if np.any(~np.isfinite(floor_ppfd)):
        print("[Warning] Non-finite values (NaN/Inf) detected in final floor_ppfd. Replacing with 0.")
        floor_ppfd = np.nan_to_num(floor_ppfd, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Return Final Result ---
    return floor_ppfd


# ------------------------------------
# 10) Execution Block (Adjust parameter definitions)
# ------------------------------------
def main():
    # --- Simulation-Specific Parameters ---
    W = 6.096; L = 6.096; H = 0.9144
    global FIXED_NUM_LAYERS; FIXED_NUM_LAYERS = 10 # Must match param array lengths
    cob_params = np.array([10000.0] * FIXED_NUM_LAYERS)
    strip_lumens = [0.0] + list(np.linspace(1000, 8000, FIXED_NUM_LAYERS - 1))
    strip_module_lumen_params = np.array(strip_lumens)

    # --- Runtime Parameter Adjustments ---
    global N_FF_SAMPLES, MC_SAMPLES
    # Example: Add argparse to override N_FF_SAMPLES, MC_SAMPLES, etc.
    parser = argparse.ArgumentParser(description="Lighting Simulation (Full IES, MC Form Factors)")
    parser.add_argument('--no-plot', action='store_true', help='Disable the heatmap plot')
    parser.add_argument('--anno', type=int, default=15, help='Annotation density step (0 disables)')
    parser.add_argument('--ffsamples', type=int, default=N_FF_SAMPLES, help=f'MC samples for Form Factor calculation (default: {N_FF_SAMPLES})')
    parser.add_argument('--mcsamples', type=int, default=MC_SAMPLES, help=f'MC samples for indirect floor illumination (default: {MC_SAMPLES})')
    args = parser.parse_args()
    N_FF_SAMPLES = args.ffsamples; MC_SAMPLES = args.mcsamples
    print(f"[Config] Using N_FF_SAMPLES = {N_FF_SAMPLES}, MC_SAMPLES = {MC_SAMPLES}")

    # --- Geometry ---
    print("Preparing geometry...")
    light_positions, light_lumens, light_types, cob_positions_only, ordered_strip_vertices, X, Y, patches_dict = prepare_geometry(
        W, L, H, cob_params, strip_module_lumen_params
    )
    total_lights = light_positions.shape[0]
    num_patches = patches_dict.get('center', np.array([])).shape[0]
    if num_patches == 0: print("Error: No patches generated."); return
    if total_lights == 0: print("Warning: No light sources generated.")

    # --- Simulation ---
    print(f"\nStarting simulation: {total_lights} emitters, {num_patches} patches...")
    start_sim_time = time.time()
    floor_ppfd = simulate_lighting(light_positions, light_lumens, light_types, X, Y, patches_dict)
    end_sim_time = time.time()
    print(f"\nTotal simulation time: {end_sim_time - start_sim_time:.2f} seconds.")

    if floor_ppfd is None or floor_ppfd.size == 0: print("Error: Simulation failed."); return

    # --- Statistics & Output ---
    print("Calculating statistics...")
    mean_ppfd = np.mean(floor_ppfd); std_dev = np.std(floor_ppfd)
    min_ppfd = np.min(floor_ppfd); max_ppfd = np.max(floor_ppfd)
    mad = rmse = cv_percent = min_max_ratio = min_avg_ratio = cu_percent = dou_percent = 0.0
    if mean_ppfd > EPSILON:
        mad = np.mean(np.abs(floor_ppfd - mean_ppfd)); rmse = np.sqrt(np.mean((floor_ppfd - mean_ppfd)**2))
        cv_percent = (std_dev / mean_ppfd) * 100; min_max_ratio = min_ppfd / max_ppfd if max_ppfd > 0 else 0
        min_avg_ratio = min_ppfd / mean_ppfd; cu_percent = (1 - mad / mean_ppfd) * 100
        dou_percent = (1 - rmse / mean_ppfd) * 100

    print(f"\n--- Results ---"); print(f"Room: {W:.2f}x{L:.2f}x{H:.2f}m, Grid: {X.shape[1]}x{X.shape[0]} ({FLOOR_GRID_RES:.3f}m)")
    num_cobs = cob_positions_only.shape[0]; strip_emitters_count = total_lights - num_cobs
    print(f"Emitters: {num_cobs} COBs, {strip_emitters_count} Strips ({total_lights} total)")
    print(f"Avg PPFD: {mean_ppfd:.2f} µmol/m²/s | Std Dev: {std_dev:.2f} | Min: {min_ppfd:.2f} | Max: {max_ppfd:.2f}")
    print(f"RMSE: {rmse:.2f} | MAD: {mad:.2f} | CV: {cv_percent:.2f}% | Min/Max: {min_max_ratio:.3f} | Min/Avg: {min_avg_ratio:.3f}")
    print(f"CU (MAD): {cu_percent:.2f}% | DOU (RMSE): {dou_percent:.2f}%")

    # --- CSV ---
    csv_filename = "ppfd_layer_data_mc_ff_v3.csv"
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
    # Global loads happen before main
    main()

# --- END OF REVISED FILE lighting-simulation-data(3.0).py ---
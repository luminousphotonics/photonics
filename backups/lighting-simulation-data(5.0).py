# --- START OF REVISED FILE lighting-simulation-simplified.py ---

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
REFL_WALL = 0.7
REFL_CEIL = 0.7
REFL_FLOOR = 0.1

LUMINOUS_EFFICACY = 160.0  # lumens/W - Still needed for Lux -> W/m^2
SPD_FILE = "/Users/austinrouse/photonics/backups/spd_data.csv" # <<< SET PATH - Still needed for W/m^2 -> PPFD

MAX_RADIOSITY_BOUNCES = 10
RADIOSITY_CONVERGENCE_THRESHOLD = 1e-3

WALL_SUBDIVS_X = 10
WALL_SUBDIVS_Y = 5
CEIL_SUBDIVS_X = 10
CEIL_SUBDIVS_Y = 10

FLOOR_GRID_RES = 0.08  # m
FIXED_NUM_LAYERS = 10 # Tied to the structure of build_cob_positions and params length
MC_SAMPLES = 64 # Monte Carlo samples for *indirect floor* illumination
N_FF_SAMPLES = 128 # Monte Carlo samples for *Form Factor* calculation

# --- Epsilon for numerical stability ---
EPSILON = 1e-9

# ------------------------------------
# 2) COB Datasheet Angular Data
# ------------------------------------
COB_ANGLE_DATA = np.array([
    [  0, 1.00], [ 10, 0.98], [ 20, 0.95], [ 30, 0.88], [ 40, 0.78],
    [ 50, 0.65], [ 60, 0.50], [ 70, 0.30], [ 80, 0.10], [ 90, 0.00],
], dtype=np.float64)

COB_angles_deg = COB_ANGLE_DATA[:, 0]
COB_shape = COB_ANGLE_DATA[:, 1] # Relative intensity shape factor

# ------------------------------------
# 3) SPD Conversion Factor & COB Normalization
# ------------------------------------
def compute_conversion_factor(spd_file):
    """ Calculates PPFD conversion factor (µmol/J) from SPD data. """
    try:
        # Basic loading, replace with more robust parsing if needed
        raw_spd = np.loadtxt(spd_file, delimiter=' ', skiprows=1)
        # Handle duplicates by averaging (simple approach)
        unique_wl, indices, counts = np.unique(raw_spd[:, 0], return_inverse=True, return_counts=True)
        counts_nonzero = np.maximum(counts, 1)
        avg_intens = np.bincount(indices, weights=raw_spd[:, 1]) / counts_nonzero
        spd = np.column_stack((unique_wl, avg_intens))
    except Exception as e:
        print(f"[SPD Error] Error loading/processing SPD file '{spd_file}': {e}")
        print("[SPD Error] Using default fallback conversion factor.")
        return 0.0138 # Example fallback value (corresponds roughly to ~72 lm/W_PAR)

    wl = spd[:, 0]
    intens = spd[:, 1]
    # Ensure sorted by wavelength for integration
    sort_idx = np.argsort(wl)
    wl = wl[sort_idx]
    intens = intens[sort_idx]

    # Calculate PAR fraction (400-700nm)
    mask_par = (wl >= 400) & (wl <= 700)
    PAR_fraction = 1.0
    if len(wl) >= 2:
        total_integral = np.trapz(intens, wl)
        if total_integral > EPSILON:
            if np.count_nonzero(mask_par) >= 2:
                par_integral = np.trapz(intens[mask_par], wl[mask_par])
                PAR_fraction = par_integral / total_integral
            else:
                print("[SPD Warning] Not enough PAR data points (>=2) for accurate PAR fraction.")
        else:
            print("[SPD Warning] Zero total SPD intensity found.")
            PAR_fraction = 0.0 # Cannot determine PAR fraction
    else:
        print("[SPD Warning] Not enough SPD data points (>=2) for integration.")
        PAR_fraction = 0.0 # Cannot determine PAR fraction

    # Calculate effective PAR wavelength for quantum conversion
    wl_m = wl * 1e-9 # Wavelength in meters
    h = 6.626e-34  # Planck's constant (J*s)
    c = 3.0e8      # Speed of light (m/s)
    N_A = 6.022e23 # Avogadro's number (mol^-1)
    lambda_eff_par = 0.0 # Effective PAR wavelength in meters

    if np.count_nonzero(mask_par) >= 2:
        # Calculate intensity-weighted average wavelength within PAR range
        numerator = np.trapz(wl_m[mask_par] * intens[mask_par], wl_m[mask_par])
        denominator = np.trapz(intens[mask_par], wl_m[mask_par])
        if denominator > EPSILON:
            lambda_eff_par = numerator / denominator

    if lambda_eff_par <= EPSILON:
        print("[SPD Warning] Could not calculate effective PAR wavelength (denominator near zero or no PAR data).")
        # Fallback: use the arithmetic mean wavelength in PAR range if available, else 550nm
        if np.count_nonzero(mask_par) > 0:
            lambda_eff_par = np.mean(wl_m[mask_par])
            print(f"[SPD Warning] Falling back to mean PAR wavelength: {lambda_eff_par*1e9:.1f} nm")
        else:
            lambda_eff_par = 550e-9 # Fallback to ~green light
            print(f"[SPD Warning] Falling back to default wavelength: {lambda_eff_par*1e9:.1f} nm")

    # Energy per photon at the effective PAR wavelength
    E_photon_par = (h * c / lambda_eff_par) if lambda_eff_par > EPSILON else 1.0 # Avoid division by zero

    # Conversion factor: (Photons/Joule) * (µmol/Photon) * PAR_Fraction
    # (1 / E_photon_par) gives Photons/Joule
    # (1 / N_A) gives mol/Photon => (1e6 / N_A) gives µmol/Photon
    conversion_factor = (1.0 / E_photon_par) * (1e6 / N_A) * PAR_fraction

    print(f"[INFO] SPD Loaded: PAR fraction={PAR_fraction:.3f}, Effective PAR λ={lambda_eff_par*1e9:.1f} nm")
    print(f"[INFO] PPFD Conversion Factor: {conversion_factor:.5f} µmol/J (µmol/s per W)")
    return conversion_factor

CONVERSION_FACTOR = compute_conversion_factor(SPD_FILE)


def calculate_cob_normalization_integral(angles_deg, shape_factors):
    """ Calculates the normalization integral for the COB angular distribution. """
    if len(angles_deg) != len(shape_factors) or len(angles_deg) < 2:
        print("[Error] Invalid COB angle data for normalization.")
        return np.pi # Fallback to isotropic normalization factor

    # Ensure 0 and 90 degrees are included for proper integration range
    angles_rad = np.radians(angles_deg)
    shape = shape_factors

    if angles_deg[0] > EPSILON: # Add 0 degree point if missing
        angles_rad = np.insert(angles_rad, 0, 0.0)
        shape = np.insert(shape, 0, shape[0]) # Assume same intensity as first point
    if abs(angles_deg[-1] - 90.0) > EPSILON: # Add 90 degree point if missing
        angles_rad = np.append(angles_rad, np.pi/2.0)
        shape = np.append(shape, 0.0) # Assume zero intensity at 90 deg

    # Integrand: shape(theta) * 2 * pi * sin(theta)
    integrand = shape * 2.0 * np.pi * np.sin(angles_rad)

    # Integrate using trapezoidal rule
    norm_integral = np.trapz(integrand, angles_rad)

    if norm_integral < EPSILON:
        print("[Warning] COB normalization integral is near zero. Check COB data. Using fallback.")
        return np.pi # Avoid division by zero, fallback to isotropic

    print(f"[INFO] COB Normalization Integral: {norm_integral:.4f}")
    return norm_integral

COB_NORMALIZATION_INTEGRAL = calculate_cob_normalization_integral(COB_angles_deg, COB_shape)

# ------------------------------------
# 4) Numba-Compatible Lambertian Intensity Function
# ------------------------------------
@njit(cache=True)
def interp_1d_lambertian(x, xp, fp):
    """ Safely interpolates 1D COB shape factor, clamping boundaries (0-90 deg). """
    # Assumes xp is sorted (COB_angles_deg)
    x_clamped = max(0.0, min(x, 90.0)) # Clamp angle to 0-90 range
    idx = np.searchsorted(xp, x_clamped, side='right')

    if idx == 0:
        return fp[0]  # Value at angle 0
    if idx == len(xp):
        # Should be handled by clamp, but safety check
        # If angle is exactly 90 and 90 is last point
        if abs(x_clamped - xp[-1]) < EPSILON: return fp[-1]
        # If angle > last defined angle (but <= 90) -> interpolate towards 90 (where intensity is likely 0)
        # Or just return last value if that's more appropriate? Let's interpolate towards 0 at 90
        x0, x1 = xp[-1], 90.0
        y0, y1 = fp[-1], 0.0 # Assume intensity drops to 0 at 90 if not defined
        delta_x = x1 - x0
        if delta_x <= EPSILON: return y0
        weight = (x_clamped - x0) / delta_x
        return y0 * (1.0 - weight) + y1 * weight


    # Standard linear interpolation
    x0, x1 = xp[idx-1], xp[idx]
    y0, y1 = fp[idx-1], fp[idx]
    delta_x = x1 - x0
    # Handle potential duplicate points in angle data
    if delta_x <= EPSILON:
        return y0 # or (y0+y1)/2 ? Let's take the lower index value
    weight = (x_clamped - x0) / delta_x
    return y0 * (1.0 - weight) + y1 * weight

@njit(cache=True)
def calculate_lambertian_intensity(dx, dy, dz, dist, total_emitter_lumens,
                                   cob_angles_deg, cob_shape, cob_norm_integral):
    """ Calculates luminous intensity (cd) based on angle and normalized COB shape. """
    if dist < EPSILON:
        # At zero distance, assume peak intensity (angle = 0)
        shape_factor = cob_shape[0] # Assumes first entry is angle 0
    else:
        # Cosine of angle between emitter's downward axis (0,0,-1) and vector to point (dx,dy,dz)
        cos_theta_nadir = max(-1.0, min(1.0, -dz / dist))
        # Angle in degrees from nadir (0 = straight down)
        angle_deg = math.degrees(math.acos(cos_theta_nadir))
        # Interpolate the relative shape factor at this angle
        shape_factor = interp_1d_lambertian(angle_deg, cob_angles_deg, cob_shape)

    # Calculate peak intensity (I_peak) needed to achieve total_emitter_lumens
    # L_total = I_peak * norm_integral => I_peak = L_total / norm_integral
    I_peak = total_emitter_lumens / max(cob_norm_integral, EPSILON)

    # Intensity = Peak Intensity * Shape Factor
    candela = I_peak * shape_factor

    return max(0.0, candela) # Ensure non-negative

# ------------------------------------
# 5) Geometry Building (Minor changes)
# ------------------------------------
# Functions cached_build_floor_grid, build_floor_grid, cached_build_patches, build_patches
# remain IDENTICAL to version 3.0.

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
         return {k: np.empty((0, 3) if k in ['center', 'normal', 't1', 't2'] else 0, dtype=np.float64) for k in patches_data}

    return {k: np.array(v, dtype=np.float64) for k, v in patches_data.items()}

def build_patches(W, L, H): return cached_build_patches(W, L, H)


# --- COB Placement (Simplified) ---
def _get_cob_abstract_coords_and_transform(W, L, H, num_total_layers):
    """ Generates abstract hexagonal grid coordinates and transformation parameters. """
    n = num_total_layers - 1
    abstract_positions = []
    # Center point (layer 0)
    abstract_positions.append({'x': 0, 'y': 0, 'h': H, 'layer': 0, 'is_vertex': True})
    # Subsequent layers forming hexagons
    for i in range(1, n + 1):
        # Iterate around the hexagon perimeter for layer i
        for x in range(i, 0, -1): abstract_positions.append({'x': x, 'y': i - x, 'h': H, 'layer': i, 'is_vertex': (x == i or x == 0)})
        for x in range(0, -i, -1): abstract_positions.append({'x': x, 'y': i + x, 'h': H, 'layer': i, 'is_vertex': (x == -i or x == 0)})
        for x in range(-i, 0, 1): abstract_positions.append({'x': x, 'y': -i - x, 'h': H, 'layer': i, 'is_vertex': (x == -i or x == 0)})
        for x in range(0, i + 1, 1): abstract_positions.append({'x': x, 'y': -i + x, 'h': H, 'layer': i, 'is_vertex': (x == i or x == 0)})

    # Ensure uniqueness and sort for consistent ordering
    unique_positions = []
    seen_coords = set()
    for pos in abstract_positions:
        coord_tuple = (pos['x'], pos['y'], pos['layer'])
        if coord_tuple not in seen_coords:
            unique_positions.append(pos)
            seen_coords.add(coord_tuple)
    # Sort by layer, then angle for consistency
    unique_positions.sort(key=lambda p: (p['layer'], math.atan2(p['y'], p['x']) if p['layer'] > 0 else 0))

    # Transformation parameters
    theta = math.radians(45) # Rotation angle
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    center_x, center_y = W / 2, L / 2
    # Scale to fit within room dimensions based on outermost layer
    scale_x = max(W / (n * math.sqrt(2)) if n > 0 else W / 2, EPSILON)
    scale_y = max(L / (n * math.sqrt(2)) if n > 0 else L / 2, EPSILON)
    # Adjust Z position slightly below ceiling height H
    transform_params = {'center_x': center_x, 'center_y': center_y,
                        'scale_x': scale_x, 'scale_y': scale_y,
                        'cos_t': cos_t, 'sin_t': sin_t,
                        'Z_pos': H * 0.95 } # Place COBs slightly below ceiling
    return unique_positions, transform_params

def _apply_transform(abstract_pos, transform_params):
    """ Applies scaling, rotation, and translation to abstract coordinates. """
    ax, ay = abstract_pos['x'], abstract_pos['y']
    # Rotate
    rx = ax * transform_params['cos_t'] - ay * transform_params['sin_t']
    ry = ax * transform_params['sin_t'] + ay * transform_params['cos_t']
    # Scale and translate
    px = transform_params['center_x'] + rx * transform_params['scale_x']
    py = transform_params['center_y'] + ry * transform_params['scale_y']
    pz = transform_params['Z_pos']
    return px, py, pz

def build_all_light_sources(W, L, H, cob_lumen_params):
    """ Builds only COB light sources based on hexagonal layout. """
    num_total_layers = len(cob_lumen_params)
    if num_total_layers == 0 :
        print("[Warning] No COB lumen parameters provided. No lights generated.")
        return np.empty((0,3)), np.empty((0,)), np.empty((0,4)) # Position, Lumens, Position+Layer

    abstract_cob_coords, transform_params = _get_cob_abstract_coords_and_transform(W, L, H, num_total_layers)

    all_positions_list = []
    all_lumens_list = []
    cob_positions_only_list = [] # For CSV/Plotting: Stores [x, y, z, layer]

    print("[Geometry] Placing COB emitters...")
    for i, abs_cob in enumerate(abstract_cob_coords):
        layer = abs_cob['layer']
        if not (0 <= layer < num_total_layers):
            print(f"[Warning] Skipping abstract COB with invalid layer {layer}")
            continue

        lumens = cob_lumen_params[layer]
        if lumens > EPSILON:
            px, py, pz = _apply_transform(abs_cob, transform_params)
            all_positions_list.append([px, py, pz])
            all_lumens_list.append(lumens)
            cob_positions_only_list.append([px, py, pz, layer])
        # else: # Optional: print if skipping zero-lumen COB
            # print(f"[Info] Skipping COB at layer {layer} due to zero lumens.")

    # Convert lists to numpy arrays
    try:
        if not all_positions_list:
            print("[Warning] No valid COB positions generated (all lumens might be zero).")
            light_positions, light_lumens = np.empty((0, 3)), np.empty(0)
            cob_positions_only = np.empty((0, 4))
        else:
            light_positions = np.array(all_positions_list)
            light_lumens = np.array(all_lumens_list)
            cob_positions_only = np.array(cob_positions_only_list)

    except Exception as e:
        print(f"[ERROR] Failed converting COB geometry lists to NumPy arrays: {e}")
        raise # Re-raise the exception to halt execution

    num_cobs_placed = cob_positions_only.shape[0]
    print(f"[INFO] Generated {num_cobs_placed} COB emitters.")

    # Return positions, lumens, and the detailed COB positions (with layer info)
    return light_positions, light_lumens, cob_positions_only

def prepare_geometry(W, L, H, cob_lumen_params):
    """Prepares all geometry: COB lights, floor grid, patches (with basis vectors)."""
    print("[Geometry] Building COB light sources...")
    light_positions, light_lumens, cob_positions_only = build_all_light_sources(
        W, L, H, cob_lumen_params
    ) # Removed strip params and returns

    print("[Geometry] Building floor grid...")
    X, Y = build_floor_grid(W, L)

    print("[Geometry] Building room patches (incl. basis vectors)...")
    patches_dict = build_patches(W, L, H) # Returns dictionary of arrays

    # Return only relevant geometry
    return light_positions, light_lumens, cob_positions_only, X, Y, patches_dict


# ------------------------------------
# 6) Numba-JIT Computations (UPDATED Direct Illumination, FF/Radiosity Unchanged)
# ------------------------------------

# --- Direct Illumination Kernels (Simplified) ---
@njit(parallel=True, cache=True)
def compute_direct_floor(light_positions, light_lumens, X, Y,
                         cob_angles_deg, cob_shape, cob_norm_integral):
    """ Computes direct floor illuminance using the Lambertian COB model. """
    min_dist2 = (FLOOR_GRID_RES / 4.0) ** 2
    rows, cols = X.shape
    out = np.zeros_like(X)
    num_lights = light_positions.shape[0]

    # Pre-fetch COB data (already done globally, but ensures Numba sees them)
    local_cob_angles = cob_angles_deg
    local_cob_shape = cob_shape
    local_cob_norm_int = cob_norm_integral

    for r in prange(rows):
        for c in range(cols):
            fx, fy, fz = X[r, c], Y[r, c], 0.0 # Floor point
            val = 0.0 # Accumulator for illuminance at (fx, fy)

            for k in range(num_lights):
                lx, ly, lz = light_positions[k, 0:3] # Light source k position
                lumens_k = light_lumens[k] # Light source k total lumens

                if lumens_k < EPSILON: continue # Skip zero-lumen sources

                # Vector from light source to floor point
                dx, dy, dz = fx - lx, fy - ly, fz - lz
                d2 = max(dx*dx + dy*dy + dz*dz, min_dist2) # Squared distance, clamped
                dist = math.sqrt(d2)

                # Cosine of angle at the floor surface (normal = 0,0,1)
                cos_in_floor = -dz / dist # = dot((0,0,1), (-dx,-dy,-dz)/dist)
                if cos_in_floor < EPSILON: continue # Light hits from below or parallel

                cos_in_floor = min(cos_in_floor, 1.0)

                # Calculate intensity (candela) from the light source towards the floor point
                I_val = calculate_lambertian_intensity(dx, dy, dz, dist, lumens_k,
                                                       local_cob_angles, local_cob_shape,
                                                       local_cob_norm_int)

                # Add contribution to illuminance: E = (I / d^2) * cos(angle_at_surface)
                if I_val > EPSILON:
                    val += (I_val / d2) * cos_in_floor

            out[r, c] = val
    return out

@njit(parallel=True, cache=True) # Parallel might be overkill here depending on Np
def compute_patch_direct(light_positions, light_lumens,
                         patch_centers, patch_normals, patch_areas,
                         cob_angles_deg, cob_shape, cob_norm_integral):
    """ Computes direct patch illuminance using the Lambertian COB model. """
    min_dist2 = (FLOOR_GRID_RES / 4.0) ** 2 # Use a small clamp distance
    Np = patch_centers.shape[0]
    out = np.zeros(Np)
    num_lights = light_positions.shape[0]

    # Pre-fetch COB data
    local_cob_angles = cob_angles_deg
    local_cob_shape = cob_shape
    local_cob_norm_int = cob_norm_integral

    for ip in prange(Np): # Parallelize over patches
        pc = patch_centers[ip] # Patch center
        n_patch = patch_normals[ip] # Patch normal vector
        norm_n_patch_val = np.linalg.norm(n_patch)

        if norm_n_patch_val < EPSILON: continue # Skip degenerate patches
        n_patch_unit = n_patch / norm_n_patch_val

        accum_E = 0.0 # Accumulator for illuminance on patch ip

        for k in range(num_lights):
            lx, ly, lz = light_positions[k, 0:3] # Light source k position
            lumens_k = light_lumens[k] # Light source k total lumens

            if lumens_k < EPSILON: continue # Skip zero-lumen sources

            # Vector from light source to patch center
            dx, dy, dz = pc[0] - lx, pc[1] - ly, pc[2] - lz
            d2 = max(dx*dx + dy*dy + dz*dz, min_dist2) # Squared distance, clamped
            dist = math.sqrt(d2)

            # Vector from patch center towards light source (unit vector)
            vec_to_light_unit = np.array((-dx/dist, -dy/dist, -dz/dist))

            # Cosine of angle at the patch surface
            cos_in_patch = np.dot(n_patch_unit, vec_to_light_unit)

            if cos_in_patch < EPSILON: continue # Light doesn't hit the front face

            cos_in_patch = min(cos_in_patch, 1.0)

            # Calculate intensity (candela) from the light source towards the patch center
            I_val = calculate_lambertian_intensity(dx, dy, dz, dist, lumens_k,
                                                   local_cob_angles, local_cob_shape,
                                                   local_cob_norm_int)

            # Add contribution to illuminance: E = (I / d^2) * cos(angle_at_patch)
            if I_val > EPSILON:
                accum_E += (I_val / d2) * cos_in_patch

        out[ip] = accum_E
    return out


# --- Monte Carlo Form Factor Calculation ---
# Functions compute_form_factor_mc_pair, compute_form_factor_matrix
# remain IDENTICAL to version 3.0. They depend on patch geometry, not light source type.

@njit(cache=True)
def compute_form_factor_mc_pair(
    pj, Aj, nj_unit, t1j, t2j, hs1j, hs2j, # Source Patch j
    pi, Ai, ni_unit, t1i, t2i, hs1i, hs2i, # Destination Patch i
    num_samples):
    """ Computes form factor F_ji using Monte Carlo between two patches. """
    # Check for valid normals
    if abs(np.linalg.norm(nj_unit) - 1.0) > EPSILON or abs(np.linalg.norm(ni_unit) - 1.0) > EPSILON:
        # print("Warning: Normal vector not normalized in FF calculation.") # Debug only
        return 0.0 # Should not happen if normals are pre-normalized

    sum_kernel = 0.0
    valid_samples = max(1, num_samples) # Ensure at least one sample

    for _ in range(valid_samples):
        # Random point on source patch j (using tangents and half-sizes)
        u1j = np.random.uniform(-hs1j, hs1j)
        u2j = np.random.uniform(-hs2j, hs2j)
        xj = pj + u1j * t1j + u2j * t2j # Sample point on j

        # Random point on destination patch i
        u1i = np.random.uniform(-hs1i, hs1i)
        u2i = np.random.uniform(-hs2i, hs2i)
        xi = pi + u1i * t1i + u2i * t2i # Sample point on i

        # Vector between points
        vij = xi - xj
        dist2 = np.dot(vij, vij)

        # Avoid division by zero / instability at very close points
        if dist2 < EPSILON * EPSILON: # Use squared epsilon for distance check
            continue

        dist = math.sqrt(dist2)
        vij_unit = vij / dist

        # Cosines of angles with normals
        cos_j = np.dot(nj_unit, vij_unit)    # Angle on source patch j
        cos_i = np.dot(ni_unit, -vij_unit)   # Angle on destination patch i

        # Check if points can "see" each other (positive cosines)
        if cos_j > EPSILON and cos_i > EPSILON:
            # Form factor kernel
            kernel = (cos_j * cos_i) / (math.pi * dist2)
            sum_kernel += kernel

    # Average kernel over samples
    avg_kernel = sum_kernel / valid_samples

    # Form factor F_ji = integral over Ai, Aj of [ (cos_j * cos_i) / (pi * dist^2) ] dAi dAj / Aj
    # MC approximation: F_ji approx avg_kernel * Ai
    form_factor_ji = avg_kernel * Ai # Ai is the area of the *receiving* patch i

    return max(0.0, form_factor_ji) # Ensure non-negative

@njit(parallel=True, cache=True)
def compute_form_factor_matrix(
    patch_centers, patch_areas, patch_normals, patch_t1, patch_t2, patch_hs1, patch_hs2,
    num_samples):
    """ Computes the full Np x Np form factor matrix F[j, i] = F_ji using MC. """
    Np = patch_centers.shape[0]
    form_factor_matrix = np.zeros((Np, Np), dtype=np.float64)

    # Pre-normalize all normals for efficiency
    normals_unit = np.empty_like(patch_normals)
    for k in range(Np):
        norm_n = np.linalg.norm(patch_normals[k])
        if norm_n > EPSILON:
            normals_unit[k] = patch_normals[k] / norm_n
        else:
            # Handle zero-norm vector case (e.g., degenerate patch)
            normals_unit[k] = np.array((0.0, 0.0, 0.0))

    # Parallel loop over source patches (j)
    for j in prange(Np):
        # Extract source patch j parameters
        pj = patch_centers[j]; Aj = patch_areas[j]
        nj_unit = normals_unit[j]
        t1j = patch_t1[j]; t2j = patch_t2[j]
        hs1j = patch_hs1[j]; hs2j = patch_hs2[j]

        # Skip if source patch has no area or invalid normal
        if Aj <= EPSILON or np.linalg.norm(nj_unit) < EPSILON:
            continue

        # Inner loop over destination patches (i)
        for i in range(Np):
            if i == j: continue # Form factor F_ii is zero for planar/convex patches

            # Extract destination patch i parameters
            pi = patch_centers[i]; Ai = patch_areas[i]
            ni_unit = normals_unit[i]
            t1i = patch_t1[i]; t2i = patch_t2[i]
            hs1i = patch_hs1[i]; hs2i = patch_hs2[i]

            # Skip if destination patch has no area or invalid normal
            if Ai <= EPSILON or np.linalg.norm(ni_unit) < EPSILON:
                continue

            # Compute F_ji using the pair function
            F_ji = compute_form_factor_mc_pair(
                pj, Aj, nj_unit, t1j, t2j, hs1j, hs2j, # Source j
                pi, Ai, ni_unit, t1i, t2i, hs1i, hs2i, # Dest i
                num_samples
            )
            form_factor_matrix[j, i] = F_ji

    return form_factor_matrix


# --- Radiosity Loop (Using Form Factors) ---
# Function iterative_radiosity_loop_ff remains IDENTICAL to version 3.0.
# It uses the precomputed FF matrix, patch areas, reflectances, and direct illumination.

@njit(cache=True)
def iterative_radiosity_loop_ff(patch_direct, patch_areas, patch_refl, form_factor_matrix,
                                max_bounces, convergence_threshold):
    """ Calculates total incident irradiance using precomputed Form Factors F_ji. """
    Np = patch_direct.shape[0]
    if Np == 0: return np.empty(0, dtype=np.float64)

    # Initialize: Total incident = Direct incident (before any bounces)
    patch_incident_total = patch_direct.copy()
    # Initial exitance (radiosity) B = rho * E_inc_total (assuming patches aren't emitters themselves)
    patch_exitance = patch_incident_total * patch_refl

    # Iteratively calculate bounces
    for bounce in range(max_bounces):
        newly_incident_flux_indirect = np.zeros(Np, dtype=np.float64)
        prev_patch_incident_total = patch_incident_total.copy() # Store for convergence check

        # Calculate indirect flux transfer using the FF matrix
        # For each destination patch i, sum flux coming from all source patches j
        for i in range(Np):
            flux_sum_to_i = 0.0
            # Sum contributions from all source patches j
            for j in range(Np):
                if i == j: continue # F_ii = 0

                # Flux arriving at i from j = B_j * A_j * F_ji
                # Where B_j is exitance of patch j (W/m^2 or lm/m^2)
                # A_j is area of patch j (m^2)
                # F_ji is form factor from j to i (dimensionless)
                # Result flux_ji is in (W or lm)
                flux_ji = patch_exitance[j] * patch_areas[j] * form_factor_matrix[j, i]

                # Accumulate positive flux contributions
                if flux_ji > EPSILON:
                    flux_sum_to_i += flux_ji

            newly_incident_flux_indirect[i] = flux_sum_to_i # Total indirect flux (W or lm) arriving at patch i

        # --- Update Radiosity and Check Convergence ---
        max_rel_change = 0.0
        converged = True # Assume convergence until proven otherwise

        for i in range(Np):
            # New indirect incident irradiance (E_indirect) on patch i
            incident_irradiance_indirect_i = 0.0
            if patch_areas[i] > EPSILON:
                # E_indirect = Total indirect flux / Area_i (W/m^2 or lm/m^2)
                incident_irradiance_indirect_i = newly_incident_flux_indirect[i] / patch_areas[i]

            # Update total incident irradiance: E_inc_total(new) = E_direct + E_indirect(calculated this bounce)
            # Note: E_direct is constant throughout
            patch_incident_total[i] = patch_direct[i] + incident_irradiance_indirect_i

            # Update exitance for the *next* bounce calculation: B(new) = rho * E_inc_total(new)
            patch_exitance[i] = patch_incident_total[i] * patch_refl[i]

            # Check convergence based on relative change in total incident irradiance
            change = abs(patch_incident_total[i] - prev_patch_incident_total[i])
            # Use the previous total as the denominator for relative change, add EPSILON for stability
            denominator = abs(prev_patch_incident_total[i]) + EPSILON
            rel_change = change / denominator

            if rel_change > max_rel_change:
                max_rel_change = rel_change
            if rel_change > convergence_threshold:
                 converged = False # Not converged if any patch exceeds threshold

        # Check for convergence after iterating through all patches
        if converged:
            # print(f"Radiosity converged after {bounce+1} bounces. Max rel change: {max_rel_change:.2e}")
            break # Exit bounce loop
    # else: # Optional: print if max bounces reached without convergence
        # print(f"Radiosity loop reached max {max_bounces} bounces. Max rel change: {max_rel_change:.2e}")

    # Return the final total incident irradiance on each patch
    return patch_incident_total


# --- Monte Carlo Reflection onto Floor ---
# Functions compute_row_reflection, compute_reflection_on_floor remain IDENTICAL to version 3.0.
# They depend on patch exitance (calculated by radiosity), not the initial light source model.

@njit(cache=True)
def compute_row_reflection(r, X, Y, patch_centers, patch_normals, patch_areas, patch_exitance, mc_samples):
    """ Computes indirect illuminance for a single floor row using MC sampling from patches. """
    row_vals = np.empty(X.shape[1]) # Array to store results for this row
    Np = patch_centers.shape[0] # Number of patches
    valid_mc_samples = max(1, mc_samples) # Ensure at least one sample

    # Define minimum distance squared locally (can be passed if needed)
    min_dist2_indirect = (FLOOR_GRID_RES / 8.0) ** 2 # Smaller clamp for indirect?

    for c in range(X.shape[1]): # Iterate through columns (points) in the row
        fx, fy, fz = X[r, c], Y[r, c], 0.0 # Floor point coordinates
        total_indirect_E = 0.0 # Accumulator for indirect illuminance at this point

        # Iterate through all patches acting as secondary sources
        for p in range(Np):
            exitance_p = patch_exitance[p] # Exitance (lm/m^2 or W/m^2) of patch p
            area_p = patch_areas[p]       # Area (m^2) of patch p

            # Skip patch if it emits negligible flux (B*A)
            if exitance_p * area_p <= EPSILON: continue

            pc = patch_centers[p] # Center of patch p
            n = patch_normals[p]  # Normal of patch p
            norm_n_val = np.linalg.norm(n)
            if norm_n_val < EPSILON: continue # Skip degenerate patches
            n_unit = n / norm_n_val

            # Estimate patch dimensions for sampling (simple square root approx)
            # More accurate: use hs1, hs2 and t1, t2 if passed
            half_side = math.sqrt(area_p) * 0.5
            # Simple basis vectors generation (assuming not axis-aligned)
            if abs(n_unit[0]) > 0.9: v_tmp = np.array((0.0, 1.0, 0.0))
            else: v_tmp = np.array((1.0, 0.0, 0.0))
            tangent1 = np.cross(n_unit, v_tmp); t1_norm = np.linalg.norm(tangent1)
            if t1_norm < EPSILON: tangent1 = np.cross(n_unit, np.array((0.0,0.0,1.0))); t1_norm = np.linalg.norm(tangent1) # Try another axis
            if t1_norm < EPSILON: tangent1 = np.array((1.0,0.0,0.0)) # Failsafe
            else: tangent1 /= t1_norm
            tangent2 = np.cross(n_unit, tangent1) # Should be normalized

            # Monte Carlo sampling from patch p to floor point (fx, fy, fz)
            sample_sum_integrand = 0.0
            for _ in range(valid_mc_samples):
                # Generate random offset within the approximate patch area
                off1 = np.random.uniform(-half_side, half_side)
                off2 = np.random.uniform(-half_side, half_side)
                # Sample point on the patch surface
                sample_point = pc + off1 * tangent1 + off2 * tangent2

                # Vector from sample point on patch to floor point
                vec_sp_to_fp = np.array((fx - sample_point[0],
                                         fy - sample_point[1],
                                         fz - sample_point[2]))
                dist2 = np.dot(vec_sp_to_fp, vec_sp_to_fp)

                # Clamp distance squared
                dist2 = max(dist2, min_dist2_indirect)

                dist = math.sqrt(dist2)
                vec_sp_to_fp_unit = vec_sp_to_fp / dist

                # Cosine angle at floor point (normal 0,0,1)
                cos_f = -vec_sp_to_fp_unit[2] # = dot((0,0,1), -vec_unit)
                # Cosine angle at patch sample point (normal n_unit)
                cos_p = np.dot(n_unit, vec_sp_to_fp_unit)

                # Check if points can "see" each other
                if cos_f > EPSILON and cos_p > EPSILON:
                    # Integrand term for the reflection equation (excluding constant factors)
                    # dE = B * dA * (cos_p * cos_f) / (pi * dist^2)
                    # We sum (cos_p * cos_f) / dist^2 and average later
                    if dist2 > EPSILON: # Check dist2 again after clamping
                       integrand_term = (cos_p * cos_f) / dist2
                       sample_sum_integrand += integrand_term

            # Average the integrand term over all samples
            avg_integrand = sample_sum_integrand / valid_mc_samples

            # Contribution of patch p to indirect illuminance at floor point
            # dE = (B_p / pi) * avg_integrand * Area_p (from MC approx)
            total_indirect_E += (area_p * exitance_p / math.pi) * avg_integrand

        row_vals[c] = total_indirect_E # Store final indirect E for this point

    return row_vals

def compute_reflection_on_floor(X, Y, patch_centers, patch_normals, patch_areas, patch_incident_total, patch_refl, mc_samples=MC_SAMPLES):
    """ Calculates indirect floor illuminance using MC via parallel rows. """
    rows, cols = X.shape
    if rows == 0 or cols == 0: return np.zeros((rows, cols))

    # Calculate exitance B = rho * E_total_incident
    patch_exitance = patch_incident_total * patch_refl

    # Use joblib for parallel processing over rows
    print(f"  > Starting parallel indirect floor calculation ({rows} rows, {mc_samples} samples/patch)...")
    start_time = time.time()
    results = Parallel(n_jobs=-1)(delayed(compute_row_reflection)(
        r, X, Y, patch_centers, patch_normals, patch_areas, patch_exitance, mc_samples
    ) for r in range(rows))
    end_time = time.time()
    print(f"  > Parallel row computation finished in {end_time - start_time:.2f}s")

    # Assemble results into the output array
    out = np.zeros((rows, cols))
    if results and len(results) == rows:
        for r, row_vals in enumerate(results):
            if row_vals is not None and len(row_vals) == cols:
                out[r, :] = row_vals
            else:
                print(f"[Warning] Invalid results returned for indirect reflection row {r}.")
                # Fill with zeros or handle error as needed
    else:
        print("[Error] Parallel indirect reflection computation returned unexpected or incomplete results.")
        # Return zeros or raise an error

    return out

# ------------------------------------
# 7) Heatmap Plotting Function (Simplified)
# ------------------------------------
def plot_heatmap(floor_ppfd, X, Y, cob_marker_positions, W, L, annotation_step=10):
    """ Plots the floor PPFD heatmap with COB markers only. """
    fig, ax = plt.subplots(figsize=(10, 8))
    extent = [0, W, 0, L] # Plot boundaries [xmin, xmax, ymin, ymax]

    # Determine color scale limits
    vmin = np.min(floor_ppfd)
    vmax = np.max(floor_ppfd)
    if abs(vmin - vmax) < EPSILON: vmax = vmin + 1.0 # Avoid zero range for colorbar

    # Plot the heatmap image
    im = ax.imshow(floor_ppfd, cmap='viridis', interpolation='nearest', origin='lower',
                   extent=extent, aspect='equal', vmin=vmin, vmax=vmax)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('PPFD (µmol/m²/s)')

    # Add annotations (optional)
    rows, cols = floor_ppfd.shape
    if annotation_step > 0 and rows > 0 and cols > 0 and X.shape == floor_ppfd.shape and Y.shape == floor_ppfd.shape:
        step_r = max(1, rows // annotation_step)
        step_c = max(1, cols // annotation_step)
        for r in range(0, rows, step_r):
             for c in range(0, cols, step_c):
                text_x = X[r, c]
                text_y = Y[r, c]
                # Add text with background for better visibility
                ax.text(text_x, text_y, f"{floor_ppfd[r, c]:.1f}",
                        ha="center", va="center", color="white", fontsize=6,
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.5))
    elif annotation_step > 0:
        print("[Plot Warning] Skipping annotations due to grid mismatch or zero size.")

    # Add COB markers
    if cob_marker_positions is not None and cob_marker_positions.shape[0] > 0:
        # Ensure we have at least x, y coordinates
        if cob_marker_positions.shape[1] >= 2:
            ax.scatter(cob_marker_positions[:, 0], cob_marker_positions[:, 1], marker='o',
                       color='red', edgecolors='black', s=50, label="COB Positions", alpha=0.8, zorder=3)
        else:
             print("[Plot Warning] COB marker positions have unexpected shape.")

    # Set plot labels and limits
    ax.set_title("Floor PPFD (COB Emitters - MC Form Factors)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # Add legend (only for COBs now)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles=handles, fontsize='small', loc='center left', bbox_to_anchor=(1.02, 0.5))

    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

# ------------------------------------
# 8) CSV Output Function (Unchanged conceptually, uses cob_positions_only)
# ------------------------------------
# Function write_ppfd_to_csv remains IDENTICAL to version 3.0.
# It already used cob_positions_only (derived from light sources) to define layers.

def write_ppfd_to_csv(filename, floor_ppfd, X, Y, cob_positions, W, L):
    """ Writes PPFD data grouped by layers defined by COB radial distance. """
    if floor_ppfd is None or floor_ppfd.size == 0:
        print("[CSV Warning] Empty or invalid PPFD data. Skipping CSV output.")
        return

    # --- Define Layers based on COB radial distances ---
    layer_data = {i: [] for i in range(FIXED_NUM_LAYERS)} # Dict to hold PPFD values per layer
    center_x, center_y = W / 2.0, L / 2.0
    layer_radii_outer = np.zeros(FIXED_NUM_LAYERS) # Outer radius for each layer

    if cob_positions is not None and cob_positions.shape[0] > 0 and cob_positions.shape[1] >= 4:
        # We have COB positions with layer info [x, y, z, layer]
        cob_layers = cob_positions[:, 3].astype(int) # Extract layer index
        # Calculate radial distance from center for each COB
        distances = np.sqrt((cob_positions[:, 0] - center_x)**2 + (cob_positions[:, 1] - center_y)**2)

        # Find the max distance for COBs in each layer
        for i in range(FIXED_NUM_LAYERS):
            mask = (cob_layers == i)
            if np.any(mask):
                layer_radii_outer[i] = np.max(distances[mask])
            elif i > 0:
                # If a layer has no COBs, use the radius of the previous layer (creates concentric rings)
                layer_radii_outer[i] = layer_radii_outer[i-1]
            # else: layer_radii_outer[0] remains 0 if layer 0 has no COBs (handled later)

        # Ensure radii are monotonically increasing and handle missing inner layers
        layer_radii_outer = np.maximum.accumulate(layer_radii_outer)

    else:
        # Fallback: If no COB positions available, create arbitrary concentric layers
        print("[CSV Warning] No COB positions provided for layer definition. Creating default layers.")
        max_room_dist = np.sqrt((W/2)**2 + (L/2)**2)
        if FIXED_NUM_LAYERS > 0:
            layer_radii_outer = np.linspace(max_room_dist / FIXED_NUM_LAYERS, max_room_dist, FIXED_NUM_LAYERS)
        else:
            layer_radii_outer = np.array([])

    # Refine layer radii: Ensure minimum radius for layer 0, handle duplicates, add buffer
    if len(layer_radii_outer) > 0:
        # Ensure layer 0 has a small non-zero radius if needed
        if abs(layer_radii_outer[0]) < EPSILON:
             min_radius = FLOOR_GRID_RES / 2.0
             if FIXED_NUM_LAYERS > 1: min_radius = min(min_radius, layer_radii_outer[1] / 2.0 if abs(layer_radii_outer[1])>EPSILON else min_radius)
             layer_radii_outer[0] = max(min_radius, EPSILON)

        # Ensure radii are strictly increasing (remove duplicates by adding small offset)
        for i in range(1, FIXED_NUM_LAYERS):
            if layer_radii_outer[i] <= layer_radii_outer[i-1] + EPSILON:
                layer_radii_outer[i] = layer_radii_outer[i-1] + FLOOR_GRID_RES * 0.1

        # Add a small buffer to the outermost radius to catch points exactly on the edge
        layer_radii_outer[-1] += FLOOR_GRID_RES * 0.1
    else:
        print("[CSV Warning] Could not define valid layer radii. Skipping point assignment.")
        return # Cannot proceed without radii

    # --- Assign Floor Points to Layers ---
    rows, cols = floor_ppfd.shape
    if X.shape == floor_ppfd.shape and Y.shape == floor_ppfd.shape:
        for r in range(rows):
            for c in range(cols):
                fx, fy = X[r, c], Y[r, c]
                dist = np.sqrt((fx - center_x)**2 + (fy - center_y)**2)

                # Find which layer the point falls into
                assigned_layer = -1
                for i in range(FIXED_NUM_LAYERS):
                    outer = layer_radii_outer[i]
                    inner = layer_radii_outer[i-1] if i > 0 else 0.0
                    # Check if distance is within the annulus of layer i
                    if (i == 0 and dist <= outer + EPSILON) or \
                       (i > 0 and inner < dist <= outer + EPSILON):
                        assigned_layer = i
                        break # Stop searching once layer is found

                # Add point's PPFD value to the corresponding layer's list
                if assigned_layer != -1:
                    layer_data[assigned_layer].append(floor_ppfd[r, c])
                # else: # Optional: Log points outside all defined layers
                    # print(f"[CSV Debug] Point ({fx:.2f}, {fy:.2f}) dist {dist:.2f} outside defined layers (max radius {layer_radii_outer[-1]:.2f})")

    else:
        print("[CSV Warning] Floor grid shape mismatch. Cannot assign points to layers.")
        return # Cannot proceed

    # --- Write to CSV File ---
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Layer', 'PPFD']) # Header row
            # Write data row by row
            for layer_index in sorted(layer_data.keys()): # Iterate through layers 0 to N-1
                if layer_data[layer_index]: # Check if layer has data
                    for ppfd_value in layer_data[layer_index]:
                        writer.writerow([layer_index, ppfd_value])
                # else: # Optional: Write a placeholder or skip empty layers
                    # writer.writerow([layer_index, None]) # Example placeholder
    except IOError as e:
        print(f"[Error] Could not write to CSV file '{filename}': {e}")
    except Exception as e:
        print(f"[Error] Unexpected error during CSV writing: {e}")


# ------------------------------------
# 9) Main Simulation Function (Simplified Calls)
# ------------------------------------
def simulate_lighting(light_positions, light_lumens, X, Y, patches_dict):
    """ Runs the full lighting simulation pipeline (COBs only, MC Form Factors). """

    # --- Setup & Input Validation ---
    if X is None or Y is None or X.size == 0 or Y.size == 0:
        print("[Error] Invalid floor grid (X or Y). Cannot simulate.")
        return None # Indicate failure

    if not isinstance(patches_dict, dict) or not patches_dict.get('center', np.array([])).shape[0]:
        print("[Error] No valid patches provided. Cannot simulate.")
        return np.zeros_like(X, dtype=np.float64) if X is not None else None

    p_centers = patches_dict.get('center', np.empty((0,3)))
    p_areas   = patches_dict.get('area',   np.empty(0))
    p_normals = patches_dict.get('normal', np.empty((0,3)))
    p_refl    = patches_dict.get('refl',   np.empty(0))
    p_t1      = patches_dict.get('t1',     np.empty((0,3)))
    p_t2      = patches_dict.get('t2',     np.empty((0,3)))
    p_hs1     = patches_dict.get('hs1',    np.empty(0))
    p_hs2     = patches_dict.get('hs2',    np.empty(0))

    if p_centers.shape[0] == 0 or p_areas.shape[0] == 0 or p_normals.shape[0] == 0:
         print("[Error] Essential patch arrays (centers, areas, normals) are empty. Cannot simulate.")
         return np.zeros_like(X, dtype=np.float64) if X is not None else None

    num_patches = p_centers.shape[0]

    # --- Direct Illumination Calculation ---
    if light_positions.shape[0] == 0:
         print("[Warning] No light sources defined. Direct illumination will be zero.")
         floor_lux_direct = np.zeros_like(X, dtype=np.float64)
         patch_direct_lux = np.zeros(num_patches, dtype=np.float64)
    else:
        print("[Simulation] Calculating direct floor illuminance (Lambertian COB)...")
        start_direct_floor_time = time.time()
        floor_lux_direct = compute_direct_floor(
            light_positions, light_lumens, X, Y,
            COB_angles_deg, COB_shape, COB_NORMALIZATION_INTEGRAL # Pass COB model data
        )
        end_direct_floor_time = time.time()
        print(f"  > Direct floor finished in {end_direct_floor_time - start_direct_floor_time:.2f} seconds.")

        print("[Simulation] Calculating direct patch illuminance (Lambertian COB)...")
        start_direct_patch_time = time.time()
        patch_direct_lux = compute_patch_direct(
            light_positions, light_lumens,
            p_centers, p_normals, p_areas, # Pass patch arrays
            COB_angles_deg, COB_shape, COB_NORMALIZATION_INTEGRAL # Pass COB model data
        )
        end_direct_patch_time = time.time()
        print(f"  > Direct patch finished in {end_direct_patch_time - start_direct_patch_time:.2f} seconds.")

    # Handle potential NaNs from direct calculations
    floor_lux_direct = np.nan_to_num(floor_lux_direct, nan=0.0)
    patch_direct_lux = np.nan_to_num(patch_direct_lux, nan=0.0)


    # --- Form Factor Calculation (Unchanged Call) ---
    print(f"[Simulation] Pre-calculating Form Factor matrix ({num_patches}x{num_patches}) using MC ({N_FF_SAMPLES} samples)...")
    start_ff_time = time.time()
    form_factor_matrix = compute_form_factor_matrix(
        p_centers, p_areas, p_normals, p_t1, p_t2, p_hs1, p_hs2,
        N_FF_SAMPLES
    )
    end_ff_time = time.time()
    print(f"  > Form Factor matrix calculation finished in {end_ff_time - start_ff_time:.2f} seconds.")

    if form_factor_matrix is None or np.any(np.isnan(form_factor_matrix)):
        print("[ERROR] NaN or None detected in Form Factor matrix! Check MC sampling or geometry. Aborting indirect.")
        # Proceed with direct only? Or return error? Let's return direct for now.
        print("[Warning] Proceeding with DIRECT illumination only due to Form Factor error.")
        total_floor_lux = floor_lux_direct
        # Skip Radiosity and Indirect steps
    else:
        # --- Radiosity Calculation (Unchanged Call) ---
        print("[Simulation] Running radiosity using precomputed Form Factors...")
        start_rad_time = time.time()
        patch_total_incident_lux = iterative_radiosity_loop_ff(
            patch_direct_lux, p_areas, p_refl, form_factor_matrix,
            MAX_RADIOSITY_BOUNCES, RADIOSITY_CONVERGENCE_THRESHOLD
        )
        end_rad_time = time.time()
        print(f"  > Radiosity loop finished in {end_rad_time - start_rad_time:.2f} seconds.")

        patch_total_incident_lux = np.nan_to_num(patch_total_incident_lux, nan=0.0)

        # --- Indirect Floor Illumination Calculation (Unchanged Call) ---
        print("[Simulation] Calculating indirect floor illuminance (Monte Carlo)...")
        start_indirect_time = time.time()
        floor_lux_indirect = compute_reflection_on_floor(
            X, Y, p_centers, p_normals, p_areas, patch_total_incident_lux, p_refl, MC_SAMPLES
        )
        end_indirect_time = time.time()
        # Timing is printed inside compute_reflection_on_floor

        floor_lux_indirect = np.nan_to_num(floor_lux_indirect, nan=0.0)

        # --- Combine Results ---
        total_floor_lux = floor_lux_direct + floor_lux_indirect


    # --- Convert to PPFD ---
    effic = max(LUMINOUS_EFFICACY, EPSILON)
    total_radiant_Wm2 = total_floor_lux / effic # Convert Lux (lm/m^2) to W/m^2
    floor_ppfd = total_radiant_Wm2 * CONVERSION_FACTOR # Convert W/m^2 to PPFD (µmol/m²/s)

    # Final check for non-finite values
    if np.any(~np.isfinite(floor_ppfd)):
        print("[Warning] Non-finite values (NaN/Inf) detected in final floor_ppfd. Replacing with 0.")
        floor_ppfd = np.nan_to_num(floor_ppfd, nan=0.0, posinf=0.0, neginf=0.0)

    return floor_ppfd


# ------------------------------------
# 10) Execution Block (Simplified)
# ------------------------------------
def main():
    # --- Simulation-Specific Parameters ---
    W = 3.6576; L = 3.6576; H = 0.9144 # Room dimensions (meters)
    global FIXED_NUM_LAYERS
    FIXED_NUM_LAYERS = 6 # Must match param array length below

    # Define COB lumens per layer (index corresponds to layer number)
    # Example: 10 layers, constant 10000 lumens for each layer's COBs
    #cob_lumen_params = np.array([10000.0] * FIXED_NUM_LAYERS)
    # Example 2: Decreasing lumens per layer
    #cob_lumen_params = np.linspace(15000, 5000, FIXED_NUM_LAYERS)
    cob_lumen_params = np.array([
        10000,
        10000,
        9000,
        11000,
        13000,
        16000
    ])
    # --- Runtime Parameter Adjustments ---
    global N_FF_SAMPLES, MC_SAMPLES
    parser = argparse.ArgumentParser(description="Simplified Lighting Simulation (COBs, MC Form Factors)")
    parser.add_argument('--no-plot', action='store_true', help='Disable the heatmap plot')
    parser.add_argument('--anno', type=int, default=15, help='Annotation density step for plot (0 disables)')
    parser.add_argument('--ffsamples', type=int, default=N_FF_SAMPLES, help=f'MC samples for Form Factor calculation (default: {N_FF_SAMPLES})')
    parser.add_argument('--mcsamples', type=int, default=MC_SAMPLES, help=f'MC samples for indirect floor illumination (default: {MC_SAMPLES})')
    args = parser.parse_args()

    # Override global samples if provided via command line
    N_FF_SAMPLES = args.ffsamples
    MC_SAMPLES = args.mcsamples
    print(f"[Config] Using N_FF_SAMPLES = {N_FF_SAMPLES}, MC_SAMPLES = {MC_SAMPLES}")

    # --- Geometry ---
    print("Preparing geometry...")
    start_geom_time = time.time()
    # Note: prepare_geometry now returns fewer items
    light_positions, light_lumens, cob_positions_only, X, Y, patches_dict = prepare_geometry(
        W, L, H, cob_lumen_params
    )
    end_geom_time = time.time()
    print(f"Geometry preparation finished in {end_geom_time - start_geom_time:.2f}s")

    total_lights = light_positions.shape[0]
    num_patches = patches_dict.get('center', np.array([])).shape[0]

    if num_patches == 0:
        print("[Error] No patches were generated. Cannot run simulation.")
        return
    if total_lights == 0:
        print("[Warning] No light sources were generated (check lumen params). Simulation will only show reflections if ambient light added.")

    # --- Simulation ---
    print(f"\nStarting simulation: {total_lights} COB emitters, {num_patches} patches...")
    start_sim_time = time.time()
    floor_ppfd = simulate_lighting(light_positions, light_lumens, X, Y, patches_dict)
    end_sim_time = time.time()
    print(f"\nTotal simulation time: {end_sim_time - start_sim_time:.2f} seconds.")

    if floor_ppfd is None or floor_ppfd.size == 0:
        print("[Error] Simulation failed or returned no data.")
        return

    # --- Statistics & Output ---
    print("\nCalculating statistics...")
    mean_ppfd = np.mean(floor_ppfd); std_dev = np.std(floor_ppfd)
    min_ppfd = np.min(floor_ppfd); max_ppfd = np.max(floor_ppfd)
    mad = rmse = cv_percent = min_max_ratio = min_avg_ratio = cu_percent = dou_percent = 0.0
    if mean_ppfd > EPSILON:
        mad = np.mean(np.abs(floor_ppfd - mean_ppfd))
        rmse = np.sqrt(np.mean((floor_ppfd - mean_ppfd)**2))
        cv_percent = (std_dev / mean_ppfd) * 100
        min_max_ratio = min_ppfd / max_ppfd if max_ppfd > 0 else 0
        min_avg_ratio = min_ppfd / mean_ppfd
        # Uniformity Metrics based on deviations
        cu_percent = (1 - mad / mean_ppfd) * 100 # Christiansen Uniformity (using MAD)
        dou_percent = (1 - rmse / mean_ppfd) * 100 # Distributional Uniformity (using RMSE) - custom metric
    else:
        print("[Warning] Mean PPFD is near zero. Statistics might be misleading.")


    print(f"\n--- Results ---")
    print(f"Room: {W:.2f} x {L:.2f} x {H:.2f} m")
    print(f"Floor Grid: {X.shape[1]} x {X.shape[0]} points (Resolution ~{FLOOR_GRID_RES:.3f} m)")
    print(f"Emitters: {total_lights} COBs")
    print(f"Avg PPFD: {mean_ppfd:.2f} µmol/m²/s")
    print(f"Std Dev: {std_dev:.2f} | Min: {min_ppfd:.2f} | Max: {max_ppfd:.2f}")
    print(f"RMSE: {rmse:.2f} | MAD: {mad:.2f}")
    print(f"CV (StdDev/Mean): {cv_percent:.2f}%")
    print(f"Min/Max Ratio: {min_max_ratio:.3f} | Min/Avg Ratio: {min_avg_ratio:.3f}")
    print(f"Uniformity CU (1-MAD/Mean): {cu_percent:.2f}%")
    # print(f"Uniformity DOU (1-RMSE/Mean): {dou_percent:.2f}%") # Optional alternative metric

    # --- CSV ---
    csv_filename = "ppfd_layer_data.csv"
    print(f"\nWriting layer data to {csv_filename}...")
    # Pass cob_positions_only which contains [x,y,z,layer] for CSV layer definition
    write_ppfd_to_csv(csv_filename, floor_ppfd, X, Y, cob_positions_only, W, L)
    print("CSV write complete.")

    # --- Plotting ---
    if not args.no_plot:
        print("\nGenerating heatmap plot...")
        if np.all(np.isfinite(floor_ppfd)):
            # Pass cob_positions_only for plotting markers
            plot_heatmap(floor_ppfd, X, Y, cob_positions_only, W, L, annotation_step=args.anno)
            print("Plot window opened. Close plot window to exit.")
            plt.show()
        else:
            print("[Plot Error] Cannot plot non-finite PPFD data. Check simulation results.")
    else:
        print("\nPlotting disabled via --no-plot argument.")


if __name__ == "__main__":
    # Global constants (like COB data, conversion factor) are processed before main()
    main()

# --- END OF REVISED FILE lighting-simulation-simplified.py ---
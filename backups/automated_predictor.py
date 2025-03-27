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

# --- Simulation Geometry ---
SIM_W = 10.9728  # 40 ft in meters (Corrected from 30ft in description)
SIM_L = 10.9728  # 40 ft in meters (Corrected from 30ft in description)
SIM_H = 0.9144  # 3 ft in meters

# --- Predictive Model Parameters ---
known_configs = {
    10: np.array([4036.684, 2018.342, 8074.0665, 5045.855, 6055.0856, 12110.052, 15137.565, 2018.342, 20183.42, 24220.104], dtype=np.float64),
    11: np.array([3343.1405, 4298.3235, 4298.9822, 2865.549, 11462.2685, 17193.294, 7641.464, 2387.9575, 9074.2385, 22406.9105, 22924.392], dtype=np.float64),
    12: np.array([3393.159, 3393.159, 3393.8276, 2908.422, 9694.8, 17450.532, 11633.688, 2423.685, 9210.003, 9694.74, 19389.48, 23267.376], dtype=np.float64),
    13: np.array([3431.5085, 3431.5085, 6373.4516, 2941.293, 6863.0557, 12745.603, 12745.603, 13235.8185, 9314.0945, 3921.724, 3921.724, 23530.344, 23530.344], dtype=np.float64),
    14: np.array([3399.7962, 3399.7962, 5344.9613, 5344.2873, 6799.6487, 7771.0271, 11656.5406, 15053.6553, 15053.6553, 3885.5135, 3885.5135, 9713.7838, 19427.5677, 23313.0812], dtype=np.float64),
    15: np.array([1932.1547, 2415.1934, 4344.4082, 8212.8061, 7728.6775, 7728.6188, 6762.5415, 12559.0156, 16423.3299, 9660.7733, 3864.3093, 5796.4640, 7728.6188, 23185.8563, 23185.8563], dtype=np.float64),
    16: np.array([3801.6215, 3162.5110, 4132.8450, 7000.3495, 9000.9811, 9000.4963, 7500.8053, 7000.4393, 13000.3296, 16500.3501, 11000.8911, 4571.2174, 5127.4023, 9924.7413, 16685.5291, 25000.0000], dtype=np.float64),
    17: np.array([4332.1143, 3657.6542, 4017.4842, 5156.7357, 6708.1373, 8319.1898, 9815.3845, 11114.2396, 12022.3055, 12091.5716, 10714.1765, 8237.7581, 6442.0864, 6687.0190, 10137.5776, 16513.6125, 25000.0000], dtype=np.float64),
    18: np.array([4372.8801, 3801.7895, 5000.3816, 6500.3919, 8000.5562, 7691.6100, 8000.2891, 8000.7902, 10000.7478, 13000.3536, 12700.1186, 9909.0286, 7743.6182, 6354.5384, 6986.5800, 10529.9883, 16699.2440, 25000.0000], dtype=np.float64),
}
SPLINE_DEGREE = 3
SMOOTHING_FACTOR_MODE = "multiplier"
SMOOTHING_MULTIPLIER = 1.5
RATIO_FIT_DEGREE = 2
CLAMP_OUTER_LAYER = True
OUTER_LAYER_MAX_FLUX = 25000.0
# Start with no correction factor; it will be learned
EMPIRICAL_PPFD_CORRECTION = {} # This will be updated dynamically

# --- Refinement Parameters ---
REFINEMENT_LEARNING_RATE = 0.35 # Increase significantly to push DOU
MULTIPLIER_MIN = 0.85          # Keep tight bounds for stability
MULTIPLIER_MAX = 1.15          # Keep tight bounds for stability

# --- Simulation Parameters (from lighting-simulation-data.py) ---
REFL_WALL = 0.8
REFL_CEIL = 0.8
REFL_FLOOR = 0.1
LUMINOUS_EFFICACY = 182.0  # lumens/W
SPD_FILE = "spd_data.csv"  # Make sure this file exists or adjust path
MAX_RADIOSITY_BOUNCES = 10
RADIOSITY_CONVERGENCE_THRESHOLD = 1e-3
WALL_SUBDIVS_X = 10
WALL_SUBDIVS_Y = 5
CEIL_SUBDIVS_X = 10
CEIL_SUBDIVS_Y = 10
FLOOR_GRID_RES = 0.08  # m
# FIXED_NUM_LAYERS is now NUM_LAYERS_TARGET
MC_SAMPLES = 16 # Monte Carlo samples for reflection calc

COB_ANGLE_DATA = np.array([
    [0, 1.00], [10, 0.98], [20, 0.95], [30, 0.88], [40, 0.78],
    [50, 0.65], [60, 0.50], [70, 0.30], [80, 0.10], [90, 0.00],
], dtype=np.float64)
COB_angles_deg = COB_ANGLE_DATA[:, 0]
COB_shape = COB_ANGLE_DATA[:, 1]

# --- Optional Plotting ---
SHOW_FINAL_HEATMAP = True
ANNOTATION_STEP = 10 # Granularity of PPFD values shown on heatmap

# ==============================================================================
# Simulation Core Functions (Integrated from lighting-simulation-data.py)
# ==============================================================================

def compute_conversion_factor(spd_file):
    # ... (same code as in lighting-simulation-data.py) ...
    try:
        spd = np.loadtxt(spd_file, delimiter=' ', skiprows=1)
    except Exception as e:
        print(f"[Warning] Error loading SPD data from '{spd_file}': {e}. Using default 0.0138 µmol/J.")
        return 0.0138 # Default value if file fails

    wl = spd[:, 0]
    intens = spd[:, 1]
    mask_par = (wl >= 400) & (wl <= 700)
    tot = np.trapz(intens, wl)
    tot_par = np.trapz(intens[mask_par], wl[mask_par])
    PAR_fraction = tot_par / tot if tot > 0 else 1.0

    wl_m = wl * 1e-9
    h, c, N_A = 6.626e-34, 3.0e8, 6.022e23
    # Use only PAR wavelengths for effective energy calculation
    if np.any(intens[mask_par] > 0):
        numerator = np.trapz(wl_m[mask_par] * intens[mask_par], wl_m[mask_par])
        denominator = np.trapz(intens[mask_par], wl_m[mask_par])
        if denominator > 1e-15:
            lambda_eff = numerator / denominator
            E_photon = (h * c / lambda_eff)
        else:
            warnings.warn("Zero intensity in PAR range for E_photon calculation. Using fallback.")
            E_photon = h*c / (550e-9) # Fallback to ~green peak
    else:
         warnings.warn("No PAR intensity found. Using fallback energy.")
         E_photon = h*c / (550e-9) # Fallback

    conversion_factor = (1.0 / E_photon) * (1e6 / N_A) * PAR_fraction if E_photon > 0 else 0
    print(f"[INFO] SPD: PAR fraction={PAR_fraction:.3f}, conv_factor={conversion_factor:.5f} µmol/J")
    return conversion_factor

CONVERSION_FACTOR = compute_conversion_factor(SPD_FILE)

def integrate_shape_for_flux(angles_deg, shape):
    # ... (same code as in lighting-simulation-data.py) ...
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
    # ... (same code as in lighting-simulation-data.py) ...
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

def build_cob_positions(W, L, H, num_total_layers):
    # ... (same code as in lighting-simulation-data.py, using num_total_layers) ...
    # Note: layer index is 0 to num_total_layers-1
    n = num_total_layers - 1 # Max layer index
    positions = []
    positions.append((0, 0, H, 0)) # Center COB, layer 0
    for i in range(1, n + 1): # Layers 1 to n
        for x in range(-i, i + 1):
            y_abs = i - abs(x)
            # Add point if it's on the diamond perimeter for layer i
            if abs(x) + y_abs == i:
                 if y_abs == 0:
                     positions.append((x, 0, H, i))
                 else:
                     positions.append((x, y_abs, H, i))
                     positions.append((x, -y_abs, H, i))

    # Transformation based on room dimensions and rotation
    theta = math.radians(45)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    centerX, centerY = W / 2, L / 2
    # Scale factor to fit the largest diamond (layer n) within the room boundaries (approx)
    # The furthest point is at (n, 0) in diamond coords, becomes (n*cos_t, n*sin_t) rotated
    # We want this rotated point scaled to fit within roughly W/2, L/2 from center
    # max_rotated_coord = n * max(abs(cos_t), abs(sin_t)) # This depends on angle
    # Let's use the diagonal distance n*sqrt(2) mapped to the room diagonal W*sqrt(2)/2 ? No, simpler:
    # Map the extent 'n' in diamond coords to roughly half the room dimension.
    # The scaling needs to ensure the n-th layer's corners are roughly mapped near the room edges after rotation.
    # A point like (n, 0) rotates to (n*cos(45), n*sin(45)) = (n*sqrt(2)/2, n*sqrt(2)/2).
    # We want scale*n*sqrt(2)/2 to be around W/2 or L/2.
    # scale = (W/2) / (n*sqrt(2)/2) = W / (n*sqrt(2)) if W=L
    scale_x = W / (n * math.sqrt(2)) if n > 0 else W/2 # Avoid div by zero if n=0 (only 1 layer)
    scale_y = L / (n * math.sqrt(2)) if n > 0 else L/2

    transformed = []
    for (xx, yy, hh, layer) in positions:
        rx = xx * cos_t - yy * sin_t
        ry = xx * sin_t + yy * cos_t
        px = centerX + rx * scale_x
        py = centerY + ry * scale_y
        # Ensure COBs are slightly below ceiling height for geometry consistency
        transformed.append((px, py, H * 0.98, layer))

    # Verify number of COBs matches expectation for N layers: 1 + sum(4*i for i=1 to N-1) = 1 + 4*N*(N-1)/2 = 1 + 2N(N-1) ? No, check calculation.
    # N=1 -> 1
    # N=2 -> 1 + 4 = 5
    # N=3 -> 1 + 4 + 8 = 13
    # N=4 -> 1 + 4 + 8 + 12 = 25
    # Formula is 1 + sum_{i=1}^{N-1} 4i = 1 + 4 * (N-1)N / 2 = 1 + 2N(N-1) = 2N^2 - 2N + 1. Let's double check build logic.
    # Example N=3: layer 0: (0,0). layer 1: (1,0),(-1,0),(0,1),(0,-1). layer 2: (2,0),(-2,0),(0,2),(0,-2),(1,1),(1,-1),(-1,1),(-1,-1). Total=1+4+8=13. Correct.
    # 2*3^2 - 2*3 + 1 = 18 - 6 + 1 = 13. Formula correct.
    expected_cobs = 2*num_total_layers**2 - 2*num_total_layers + 1 if num_total_layers > 0 else 0
    if len(transformed) != expected_cobs:
         warnings.warn(f"Unexpected number of COBs generated for N={num_total_layers}. Expected {expected_cobs}, got {len(transformed)}.", UserWarning)


    return np.array(transformed, dtype=np.float64)


def pack_luminous_flux_dynamic(flux_params_per_layer, cob_positions):
    # ... (same code as in lighting-simulation-data.py) ...
    # flux_params_per_layer is the array of N fluxes
    led_intensities = []
    num_layers = len(flux_params_per_layer)
    if num_layers == 0: return np.array([], dtype=np.float64)

    for pos in cob_positions:
        layer = int(pos[3])
        if 0 <= layer < num_layers:
            intensity = flux_params_per_layer[layer]
            led_intensities.append(intensity)
        else:
            warnings.warn(f"COB position has invalid layer index {layer}. Assigning 0 flux.", UserWarning)
            led_intensities.append(0.0)
    return np.array(led_intensities, dtype=np.float64)


# LRU Cache might need adjustment if W, L, H change frequently, but okay for fixed geometry
@lru_cache(maxsize=4)
def cached_build_floor_grid(W: float, L: float, grid_res: float):
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
    patch_centers = []
    patch_areas = []
    patch_normals = []
    patch_refl = []

    # Floor (single patch)
    patch_centers.append((W/2, L/2, 0.0))
    patch_areas.append(W * L)
    patch_normals.append((0.0, 0.0, 1.0)) # Normal points UP
    patch_refl.append(refl_f)

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
    return cached_build_patches(W, L, H,
                                WALL_SUBDIVS_X, WALL_SUBDIVS_Y, CEIL_SUBDIVS_X, CEIL_SUBDIVS_Y,
                                REFL_FLOOR, REFL_CEIL, REFL_WALL)

def prepare_geometry(W, L, H, num_total_layers):
    """Prepares all static geometry needed for the simulation."""
    cob_positions = build_cob_positions(W, L, H, num_total_layers)
    X, Y = build_floor_grid(W, L)
    patches = build_patches(W, L, H)
    return (cob_positions, X, Y, patches)

# --- Numba JIT Compiled Calculation Functions ---
@njit(parallel=True)
def compute_direct_floor(cob_positions, cob_lumens, X, Y):
    # ... (same code as in lighting-simulation-data.py) ...
    # Ensure inputs are typed correctly for numba if necessary
    min_dist2 = (FLOOR_GRID_RES / 2.0)**2
    rows, cols = X.shape
    out = np.zeros((rows, cols), dtype=np.float64) # Explicit dtype

    for r in prange(rows): # Use prange for parallel loop
        for c in range(cols):
            fx, fy = X[r, c], Y[r, c]
            val = 0.0
            for k in range(cob_positions.shape[0]):
                lx, ly, lz = cob_positions[k, 0], cob_positions[k, 1], cob_positions[k, 2]
                lumens_k = cob_lumens[k]
                if lumens_k <= 0: continue # Skip zero-flux COBs

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


@njit # Re-enabled for performance
def compute_patch_direct(cob_positions, cob_lumens, patch_centers, patch_normals, patch_areas):
    """
    Computes the direct illuminance (E) on each patch center from all COBs.
    Uses corrected incidence angle calculation. Optimized for Numba.
    """
    min_dist2 = (FLOOR_GRID_RES / 2.0)**2
    Np = patch_centers.shape[0]
    out = np.zeros(Np, dtype=np.float64)

    for ip in range(Np):
        pc = patch_centers[ip]
        n = patch_normals[ip]
        # Assuming normals are pre-normalized to unit vectors
        # If not, calculate norm_n = math.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])
        # For performance with Numba, it's better if they are unit vectors already
        norm_n = 1.0 # Assume unit normal

        accum = 0.0
        for k in range(cob_positions.shape[0]): # Loop through COBs
            lx, ly, lz = cob_positions[k, 0], cob_positions[k, 1], cob_positions[k, 2]
            lumens_k = cob_lumens[k]
            if lumens_k <= 0: continue

            # Vector L -> P
            dx, dy, dz = pc[0] - lx, pc[1] - ly, pc[2] - lz
            d2 = dx*dx + dy*dy + dz*dz
            if d2 < min_dist2: d2 = min_dist2
            dist = math.sqrt(d2)

            # Angle relative to LED Z-axis (downward)
            cos_th_led = -dz / dist # Positive if patch is below LED Z-plane

            if cos_th_led <= 1e-6: continue # Light doesn't go towards patch

            # Calculate intensity based on angle from LED normal
            # Manual clip for Numba compatibility
            clipped_cos_th = max(-1.0, min(cos_th_led, 1.0))
            angle_deg = math.degrees(math.acos(clipped_cos_th))
            I_theta = luminous_intensity(angle_deg, lumens_k) # Assumes luminous_intensity is @njit or Numba-compatible

            # Corrected incidence angle calculation
            # Vector P->L = (-dx, -dy, -dz)
            # Dot product: (P->L) dot n
            dot_patch_correct = (-dx)*n[0] + (-dy)*n[1] + (-dz)*n[2]
            # Cosine of incidence angle (angle between P->L and n)
            # Assumes norm_n = 1.0
            cos_in_patch = dot_patch_correct / dist

            # Check if light hits the front face of the patch
            if cos_in_patch <= 1e-6:
                 continue # Light hits back face or edge-on

            # Calculate illuminance contribution
            # E = (Intensity / distance^2) * cos(incidence angle)
            E_local = (I_theta / d2) * cos_in_patch
            accum += E_local

        out[ip] = accum

    return out

@njit
def iterative_radiosity_loop(patch_centers, patch_normals, patch_direct_E, patch_areas, patch_refl,
                             max_bounces, convergence_threshold):
    """
    Calculates the total illuminance (E) on each patch including indirect
    reflections using an iterative radiosity method. Numba-compiled.

    Args:
        patch_centers (np.array): (Np, 3) array of patch center coordinates.
        patch_normals (np.array): (Np, 3) array of patch normal vectors.
        patch_direct_E (np.array): (Np,) array of direct illuminance on each patch.
        patch_areas (np.array): (Np,) array of patch areas.
        patch_refl (np.array): (Np,) array of patch reflectances.
        max_bounces (int): Maximum number of bounces to simulate.
        convergence_threshold (float): Relative change threshold for convergence.

    Returns:
        np.array: (Np,) array of final total illuminance (E_direct + E_indirect) on each patch.
    """
    Np = patch_direct_E.shape[0]

    # Store B (Radiosity = flux leaving per unit area) for each patch
    patch_B = patch_refl * patch_direct_E # Initial radiosity B_0 = rho * E_direct
    patch_B_new = np.zeros_like(patch_B)
    patch_E_indirect = np.zeros_like(patch_direct_E) # To store indirect illuminance accumulated *during* a bounce

    epsilon = 1e-9 # For checking convergence denominator

    # --- Removed print_interval calculation as the print statement is removed ---
    # print_interval = max(1, max_bounces // 5)

    for bounce in range(max_bounces):
        # --- Removed start_bounce_time as the timing print is removed ---
        # start_bounce_time = time.time()
        patch_E_indirect.fill(0.0) # Reset indirect illuminance for this bounce calculation

        # Calculate Form Factors implicitly and transfer flux
        # Loop through each patch j acting as a source
        for j in range(Np):
            # If patch j doesn't reflect or has no radiosity from previous bounce, skip it
            if patch_refl[j] <= 1e-6 or patch_B[j] <= 1e-6:
                 continue

            # Total Flux leaving patch j = B_j * Area_j (Not needed directly if calculating E)
            # Use Radiosity B_j directly

            pj = patch_centers[j]
            nj = patch_normals[j]
            # Assume normals are unit vectors

            # Loop through each patch i acting as a receiver
            for i in range(Np):
                if i == j: continue # Patches do not illuminate themselves

                pi = patch_centers[i]
                ni = patch_normals[i]
                # Assume normals are unit vectors

                vij = pi - pj # Vector from center of j to center of i
                dist2 = vij[0]*vij[0] + vij[1]*vij[1] + vij[2]*vij[2]
                if dist2 < 1e-15: continue # Avoid issues with coincident patches
                dist = math.sqrt(dist2)

                # Cosine of angle between normal of j and vector j->i
                cos_j = np.dot(nj, vij) / dist
                # Cosine of angle between normal of i and vector i->j (-vij)
                cos_i = np.dot(ni, -vij) / dist

                # Check visibility and orientation (light must leave front of j and hit front of i)
                if cos_j <= 1e-6 or cos_i <= 1e-6:
                    continue

                # Approximate indirect illuminance on patch i from patch j's radiosity
                # E_indirect_on_i_from_j = B_j * (cos_j * cos_i) / (pi * dist^2) * Area_j
                E_indirect_on_i_from_j = patch_B[j] * cos_j * cos_i / (math.pi * dist2) * patch_areas[j]
                # Ensure non-negative contribution (should be handled by cos checks, but safety)
                E_indirect_on_i_from_j = max(0.0, E_indirect_on_i_from_j)

                patch_E_indirect[i] += E_indirect_on_i_from_j

        # Update Radiosity B for the *next* bounce based on total illuminance E = E_direct + E_indirect
        max_rel_change = 0.0
        for i in range(Np):
            # Total illuminance incident on patch i in this step
            total_E_i = patch_direct_E[i] + patch_E_indirect[i]
            # Radiosity for the start of the next bounce calculation
            patch_B_new[i] = patch_refl[i] * total_E_i # B = rho * E_total

            # Check convergence based on change in Radiosity B
            change = abs(patch_B_new[i] - patch_B[i])
            denom = abs(patch_B[i]) + epsilon # Avoid division by zero
            rel_change = change / denom
            if rel_change > max_rel_change:
                max_rel_change = rel_change

        # Update patch_B for the next iteration
        patch_B[:] = patch_B_new[:] # Use slicing for in-place update required by Numba arrays

        # --- Removed timing calculation ---
        # end_bounce_time = time.time()

        # --- Removed the print statement causing the f-string error ---
        # if bounce % print_interval == 0 or bounce == max_bounces - 1:
        #      print(f"  Radiosity Bounce {bounce+1}/{max_bounces}: Max Rel Change = {max_rel_change:.4e}, Time = {end_bounce_time - start_bounce_time:.2f}s")


        # Check for convergence
        if max_rel_change < convergence_threshold:
            # This print is outside the main computation, should be OK even with njit
            # but let's keep it simple just in case.
            # print(f"  Radiosity converged after {bounce+1} bounces.") # Original version
            print("  Radiosity converged.") # Simplified version
            break # Exit bounce loop

    # This else block executes if the loop completes without breaking (no convergence)
    # This else block executes if the loop completes without breaking (no convergence)
    else:
        # --- Remove or comment out this line: ---
        # warnings.warn("Radiosity did not converge within max bounces. Check results.", UserWarning)
        # --- Optional: Add a simple print statement *outside* the @njit function if you need notification ---
        pass # Or just do nothing inside the njit function

    # Return the final *total* illuminance E = E_direct + E_indirect
    # Note: patch_E_indirect at this point holds the indirect illuminance from the *last* completed bounce.
    # We need E_direct + sum of all E_indirect contributions.
    # The radiosity B already incorporates all bounces. E_total = B / rho
    # However, returning E_total makes more sense for the subsequent MC step.
    # Let's recalculate final E based on final B. This assumes rho > 0 for patches that received light.
    final_E = np.zeros_like(patch_direct_E)
    for i in range(Np):
         if patch_refl[i] > 1e-6:
             final_E[i] = patch_B[i] / patch_refl[i]
         else:
             # If reflectance is zero, total illuminance is just the direct part
             final_E[i] = patch_direct_E[i]
         # Safety check for negative values if something went wrong
         final_E[i] = max(0.0, final_E[i])

    # Alternatively, and perhaps more correctly, accumulate E_indirect over bounces?
    # The current implementation using B should implicitly account for all bounces.
    # E_total(n) = E_direct + E_indirect(n)
    # B(n) = rho * E_total(n)
    # E_indirect(n) = Func( B(n-1) )
    # So the final B holds the result after max_bounces (or convergence).
    # The corresponding E_total is B_final / rho.

    return final_E # Return total illuminance on each patch


# --- Reflection Calculation using Monte Carlo (Joblib parallelized) ---
# Helper for parallel processing
@njit
def compute_row_reflection_mc(r, X, Y, patch_centers, patch_normals, patch_areas, patch_total_E, patch_refl, mc_samples):
    row_vals = np.zeros(X.shape[1], dtype=np.float64)
    # Pre-calculate leaving flux density (Radiosity B = rho * E)
    patch_B = patch_refl * patch_total_E

    # Optimization: Pre-calculate tangents for patches if possible (can't easily do in njit if complex)
    # Basic tangent calculation inside loop for now

    for c in range(X.shape[1]):
        fx, fy, fz = X[r, c], Y[r, c], 0.0 # Floor point
        val = 0.0

        for p in range(patch_centers.shape[0]):
            # Skip floor patch itself (index 0) and non-reflective patches
            if p == 0 or patch_refl[p] <= 1e-6 or patch_B[p] <= 1e-6:
                continue

            radiosity_p = patch_B[p] # Flux leaving per unit area (lm/m^2 or W/m^2)
            pc = patch_centers[p]
            n = patch_normals[p]
            area_p = patch_areas[p]

            # Simple tangent generation (robustness depends on normal orientation)
            if abs(n[2]) < 0.999: # If normal is not purely vertical
                tangent1 = np.array([-n[1], n[0], 0.0])
            else: # Normal is vertical (ceiling)
                tangent1 = np.array([1.0, 0.0, 0.0])

            norm_t1 = np.linalg.norm(tangent1)
            if norm_t1 > 1e-9:
                tangent1 /= norm_t1
            else: # Fallback if normal was tricky (e.g., [0,0,1]) -> t1=[0,1,0]? Let's stick to X-axis.
                 tangent1 = np.array([1.0, 0.0, 0.0]) if abs(n[2]) > 0.999 else np.array([-n[1], n[0], 0.0]) # Re-evaluate simple case


            tangent2 = np.cross(n, tangent1)
            norm_t2 = np.linalg.norm(tangent2)
            if norm_t2 > 1e-9:
                 tangent2 /= norm_t2
            else: # Fallback if cross product is zero (n parallel to t1?)
                 # This case shouldn't happen if t1 is chosen correctly perpendicular to n
                 # If n = [1,0,0], t1 = [0,1,0], t2 = [0,0,-1]
                 tangent2 = np.cross(n, np.array([0.0, 1.0, 0.0])) # Try crossing with Y axis
                 if np.linalg.norm(tangent2) < 1e-9: tangent2 = np.cross(n, np.array([0.0, 0.0, 1.0])) # Try Z axis
                 if np.linalg.norm(tangent2) > 1e-9: tangent2 /= np.linalg.norm(tangent2)
                 else: tangent2 = np.array([0., 1., 0.]) # Absolute fallback

            half_side = math.sqrt(area_p) / 2.0 # Approximate patch as square for sampling

            sample_sum_ff = 0.0
            for _ in range(mc_samples):
                # Generate random point on the patch surface
                offset1 = np.random.uniform(-half_side, half_side)
                offset2 = np.random.uniform(-half_side, half_side)
                sample_point = pc + offset1*tangent1 + offset2*tangent2

                # Vector from sample point on patch p to floor point f
                v_pf = np.array([fx - sample_point[0], fy - sample_point[1], fz - sample_point[2]])
                dist2 = v_pf[0]**2 + v_pf[1]**2 + v_pf[2]**2
                if dist2 < 1e-15: continue
                dist = math.sqrt(dist2)

                # Cosine angle at patch p (normal n)
                cos_p = np.dot(n, v_pf) / dist
                # Cosine angle at floor f (normal nf = [0,0,1])
                nf = np.array([0.0, 0.0, 1.0])
                cos_f = np.dot(nf, -v_pf) / dist

                # Check visibility / orientation
                if cos_p <= 1e-6 or cos_f <= 1e-6:
                    continue

                # Differential Form Factor term (geometry term)
                geom_term = (cos_p * cos_f) / (math.pi * dist2)
                sample_sum_ff += max(0.0, geom_term)

            # Average geometric term over MC samples
            avg_geom_term = sample_sum_ff / mc_samples

            # Contribution to floor illuminance from patch p's radiosity
            # dE_f = B_p * geom_term * dArea_p
            # E_f = sum ( B_p * avg_geom_term * Area_p )
            val += radiosity_p * avg_geom_term * area_p

        row_vals[c] = val
    return row_vals

def compute_reflection_on_floor(X, Y, patch_centers, patch_normals, patch_areas, patch_total_E, patch_refl,
                                mc_samples=MC_SAMPLES):
    """Calculates the indirect illuminance on the floor grid via Monte Carlo."""
    rows, cols = X.shape
    print(f"[INFO] Computing indirect floor illuminance via MC ({mc_samples} samples/patch-point pair)...")
    start_time = time.time()
    # Use joblib for parallel execution of rows
    results = Parallel(n_jobs=-1)(delayed(compute_row_reflection_mc)(
        r, X, Y, patch_centers, patch_normals, patch_areas, patch_total_E, patch_refl, mc_samples
    ) for r in range(rows))
    end_time = time.time()
    print(f"[INFO] MC reflection calculation finished in {end_time - start_time:.2f}s.")

    # Combine results into the output array
    out = np.zeros((rows, cols), dtype=np.float64)
    for r, row_vals in enumerate(results):
        out[r, :] = row_vals
    return out

# --- Main Simulation Function ---
def simulate_lighting(flux_params_per_layer, geometry_data):
    """
    Runs the full lighting simulation for a given set of flux parameters.

    Args:
        flux_params_per_layer (np.array): Array of luminous flux values for each layer (N elements).
        geometry_data (tuple): Pre-calculated geometry: (cob_positions, X, Y, patches).
                                Patches = (p_centers, p_areas, p_normals, p_refl)

    Returns:
        tuple: (floor_ppfd, X, Y, cob_positions)
               floor_ppfd: 2D array of PPFD values on the floor grid.
               X, Y: Meshgrid coordinates for the floor grid.
               cob_positions: Positions of the COBs used in the simulation.
    """
    cob_positions, X, Y, (p_centers, p_areas, p_normals, p_refl) = geometry_data
    num_layers = len(flux_params_per_layer)
    print(f"\n--- Running Simulation for N={num_layers} ---")

    # 1. Assign flux to individual COBs based on layer
    lumens_per_cob = pack_luminous_flux_dynamic(flux_params_per_layer, cob_positions)
    total_sim_flux = np.sum(lumens_per_cob)
    print(f"[Sim] Total Luminous Flux in Simulation: {total_sim_flux:.2f} lm")
    if abs(total_sim_flux) < 1e-6:
         warnings.warn("[Sim] Total simulation flux is near zero. Results will be zero.", UserWarning)
         return (np.zeros_like(X), X, Y, cob_positions) # Return zero PPFD

    # 2. Compute Direct Illuminance on Floor and Patches
    print("[Sim] Calculating direct illuminance...")
    start_direct = time.time()
    floor_lux_direct = compute_direct_floor(cob_positions, lumens_per_cob, X, Y)
    patch_E_direct = compute_patch_direct(cob_positions, lumens_per_cob, p_centers, p_normals, p_areas)
    print(f"[Sim] Direct calculation time: {time.time() - start_direct:.2f}s")
    print(f"[Sim] Max Direct Floor Lux: {np.max(floor_lux_direct):.2f}")
    print(f"[Sim] Max Direct Patch E: {np.max(patch_E_direct):.2f}")


    # 3. Compute Radiosity (Total Illuminance on Patches)
    print("[Sim] Calculating radiosity...")
    start_radio = time.time()
    patch_E_total = iterative_radiosity_loop(p_centers, p_normals, patch_E_direct, p_areas, p_refl,
                                            MAX_RADIOSITY_BOUNCES, RADIOSITY_CONVERGENCE_THRESHOLD)
    print(f"[Sim] Radiosity calculation time: {time.time() - start_radio:.2f}s")
    print(f"[Sim] Max Total Patch E: {np.max(patch_E_total):.2f}")

    # 4. Compute Indirect Illuminance on Floor from Patches
    print("[Sim] Calculating indirect illuminance on floor...")
    floor_lux_indirect = compute_reflection_on_floor(X, Y, p_centers, p_normals, p_areas, patch_E_total, p_refl, MC_SAMPLES)
    print(f"[Sim] Max Indirect Floor Lux: {np.max(floor_lux_indirect):.2f}")

    # 5. Calculate Total Illuminance (Lux) and Convert to PPFD
    total_luminous_lux = floor_lux_direct + floor_lux_indirect

    # Convert Lux (lm/m^2) to Radiance (W/m^2) then to PPFD (µmol/m^2/s)
    total_radiant_Wm2 = total_luminous_lux / LUMINOUS_EFFICACY
    floor_ppfd = total_radiant_Wm2 * CONVERSION_FACTOR

    print(f"[Sim] Simulation Complete. Max PPFD: {np.max(floor_ppfd):.2f} µmol/m²/s")
    return floor_ppfd, X, Y, cob_positions


# --- Simulation Results Analysis ---
def calculate_metrics(floor_ppfd):
    """Calculates average PPFD and DOU from the floor grid data."""
    if floor_ppfd is None or floor_ppfd.size == 0:
        return 0.0, 0.0

    mean_ppfd = np.mean(floor_ppfd)
    if mean_ppfd < 1e-6: # Avoid division by zero if avg PPFD is effectively zero
        return 0.0, 0.0

    rmse = np.sqrt(np.mean((floor_ppfd - mean_ppfd)**2))
    # DOU = 100 * (PPFD_min / PPFD_avg) -- standard definition
    # Let's use the CV-based approximation if PPFD_min is hard to get / noisy
    # Or use the 1 - RMSE/Mean version from the original script
    dou = 100 * (1 - rmse / mean_ppfd) # As used before
    # dou_alt = 100 * (np.min(floor_ppfd) / mean_ppfd) # Standard DOU

    return mean_ppfd, dou # Using the RMSE-based DOU


def calculate_per_layer_ppfd(floor_ppfd, X, Y, cob_positions, W, L, num_layers):
    """
    Calculates the average PPFD for points corresponding to each COB layer,
    based on distance rings defined by midpoints between layer radii.

    Returns:
        np.array: Average PPFD per layer (size num_layers).
                  Uses overall average PPFD as fallback for layers with no points.
    """
    if floor_ppfd is None or floor_ppfd.size == 0 or cob_positions.size == 0:
        return np.zeros(num_layers, dtype=np.float64)

    layer_data = {i: [] for i in range(num_layers)}
    cob_layers = cob_positions[:, 3].astype(int)
    center_x, center_y = W / 2.0, L / 2.0

    # Determine the characteristic radius for each layer (max distance of COBs in layer)
    layer_radii = np.zeros(num_layers, dtype=np.float64)
    non_zero_radii = []
    for i in range(num_layers):
        layer_cob_indices = np.where(cob_layers == i)[0]
        if len(layer_cob_indices) > 0:
            layer_cob_coords = cob_positions[layer_cob_indices, :2]
            distances = np.sqrt((layer_cob_coords[:, 0] - center_x)**2 +
                                (layer_cob_coords[:, 1] - center_y)**2)
            if distances.size > 0:
                layer_radii[i] = np.max(distances)
                # Store non-zero radii for boundary calculations
                if layer_radii[i] > 1e-6:
                     non_zero_radii.append(layer_radii[i])
            else: # Should not happen if indices exist
                layer_radii[i] = layer_radii[i-1] if i > 0 else 0.0
        elif i > 0:
            layer_radii[i] = layer_radii[i-1] # If layer empty, use previous radius
        # else layer_radii[0] remains 0.0 initially

    # Ensure layer_radii are sorted (handles empty layers potentially messing order)
    layer_radii = np.sort(layer_radii)

    # Define ring boundaries using midpoints between characteristic radii.
    ring_boundaries = np.zeros(num_layers, dtype=np.float64)
    if num_layers == 1:
        ring_boundaries[0] = max(W, L) * 1.5 # Whole area if only 1 layer
    else:
        # Find the radius of the first layer with COBs significantly away from center
        first_ring_radius = layer_radii[1] if layer_radii.size > 1 else max(W,L)/2.0
        if first_ring_radius < 1e-5: # If layer 1 is also at center, find first non-zero
            first_ring_radius = min(non_zero_radii) if non_zero_radii else max(W, L) / 2.0

        # Boundary 0: Defines the edge of the central disk (layer 0)
        ring_boundaries[0] = first_ring_radius / 2.0 # Place boundary halfway to first ring

        # Boundaries 1 to N-2: Midpoint between layer i and i+1 radii
        for i in range(1, num_layers - 1):
            r_i = layer_radii[i]
            r_next = layer_radii[i+1]
            # If radii are distinct, use midpoint
            if r_next > r_i + 1e-6:
                 ring_boundaries[i] = (r_i + r_next) / 2.0
            # If radii are same/close (e.g., empty layers), keep boundary same as previous effective one
            else:
                 ring_boundaries[i] = ring_boundaries[i-1] # Avoid creating zero-width rings

        # Last boundary: Use a large extent to capture all points
        ring_boundaries[num_layers - 1] = max(W, L) * 1.5

    # Ensure boundaries are monotonically increasing and the first is positive
    # (Sorting might be needed if midpoints calculation gets weird with identical radii)
    for i in range(1, num_layers):
        if ring_boundaries[i] < ring_boundaries[i-1] + 1e-6 :
             ring_boundaries[i] = ring_boundaries[i-1] + 1e-6 # Enforce small separation
    ring_boundaries[0] = max(1e-6, ring_boundaries[0]) # Ensure layer 0 gets a small area

    # --- Keep print statements for debugging this ---
    print(f"[Feedback] Layer Radii (Max Dist): {np.array2string(layer_radii, precision=3)}")
    print(f"[Feedback] Ring Boundaries: {np.array2string(ring_boundaries, precision=3)}")
    # --- End Debug Prints ---

    # Assign each floor grid point PPFD to a layer based on distance and *boundaries*
    rows, cols = floor_ppfd.shape
    points_assigned = [0] * num_layers
    for r in range(rows):
        for c in range(cols):
            fx, fy = X[r, c], Y[r, c]
            dist_to_center = math.sqrt((fx - center_x)**2 + (fy - center_y)**2)

            assigned_layer = -1
            # Check layer 0
            # Ensure points exactly at boundary 0 go to layer 0
            if dist_to_center <= ring_boundaries[0] + 1e-9: # Add epsilon for float comparison
                assigned_layer = 0
            # Check layers 1 to N-1
            else:
                for i in range(1, num_layers):
                    # Point belongs to layer i if: boundary[i-1] < dist <= boundary[i]
                    if dist_to_center > ring_boundaries[i-1] + 1e-9 and dist_to_center <= ring_boundaries[i] + 1e-9:
                        assigned_layer = i
                        break

            # Assign points beyond the last calculated boundary to the outermost layer
            if assigned_layer == -1 and dist_to_center > ring_boundaries[num_layers-1] + 1e-9:
                 assigned_layer = num_layers - 1

            if assigned_layer != -1:
                layer_data[assigned_layer].append(floor_ppfd[r, c])
                points_assigned[assigned_layer] += 1
            # else: point wasn't assigned (this should ideally not happen)

    # Calculate average PPFD for each layer
    avg_ppfd_per_layer = np.zeros(num_layers, dtype=np.float64)
    valid_layers = 0
    overall_avg_ppfd = np.mean(floor_ppfd) if floor_ppfd.size > 0 else 0 # Calculate once

    for i in range(num_layers):
        if layer_data[i]: # Check if list is not empty
            avg_ppfd_per_layer[i] = np.mean(layer_data[i])
            valid_layers += 1
        else:
            # Fallback: Use the overall average PPFD for this layer's feedback
            avg_ppfd_per_layer[i] = overall_avg_ppfd
            # Issue a warning
            warnings.warn(f"Layer {i} had no PPFD points assigned ({points_assigned[i]} points). Using overall average PPFD ({overall_avg_ppfd:.2f}) as feedback.", UserWarning)

    if valid_layers < num_layers:
         print(f"[Feedback] Warning: Only {valid_layers}/{num_layers} layers received PPFD points.")
    else:
         print(f"[Feedback] All {num_layers} layers received PPFD points based on new boundaries.")

    return avg_ppfd_per_layer

# --- Optional Plotting Function ---
def plot_heatmap(floor_ppfd, X, Y, cob_positions, title="Floor PPFD Heatmap", annotation_step=5):
    """Displays a heatmap of the floor PPFD."""
    fig, ax = plt.subplots(figsize=(10, 8))
    if floor_ppfd is None or floor_ppfd.size == 0:
        ax.set_title(f"{title} (No Data)")
        return

    extent = [X.min(), X.max(), Y.min(), Y.max()]
    im = ax.imshow(floor_ppfd, cmap='hot', interpolation='nearest', origin='lower', extent=extent, aspect='equal')
    fig.colorbar(im, ax=ax, label="PPFD (µmol/m²/s)")

    # Annotate PPFD values sparsely
    rows, cols = floor_ppfd.shape
    for r in range(0, rows, annotation_step):
        for c in range(0, cols, annotation_step):
            try:
                ax.text(X[r, c], Y[r, c], f"{floor_ppfd[r, c]:.0f}",
                        ha="center", va="center", color="white", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.5))
            except IndexError:
                continue # Avoid errors if annotation step is too large

    # Plot COB positions with layer info (optional, can be slow for large N)
    layers = cob_positions[:, 3].astype(int)
    unique_layers = sorted(np.unique(layers))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_layers)))
    for i, layer in enumerate(unique_layers):
        idx = np.where(layers == layer)[0]
        ax.scatter(cob_positions[idx, 0], cob_positions[idx, 1], marker='o',
                   color=colors[i], edgecolors='black', s=30, label=f"Layer {layer}")

    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show(block=False) # Show plot without blocking execution
    plt.pause(0.1) # Pause briefly to allow plot to render

# ==============================================================================
# Predictive Model & Refinement Functions (from predictive_tool.py)
# ==============================================================================

def prepare_spline_data(known_configs):
    # ... (same code as in predictive_tool.py) ...
    n_list, x_norm_list, flux_list = [], [], []
    min_n, max_n = float('inf'), float('-inf')

    sorted_keys = sorted(known_configs.keys())
    print("[Spline] Preparing data from N:", sorted_keys)
    for n in sorted_keys:
        fluxes = known_configs[n]
        num_layers_in_config = len(fluxes)
        if num_layers_in_config != n:
             warnings.warn(f"[Spline] Mismatch for N={n}. Expected {n} fluxes, got {num_layers_in_config}. Using actual count {num_layers_in_config}.", UserWarning)
             n_actual = num_layers_in_config
        else:
             n_actual = n

        if n_actual < 2: continue # Need at least 2 layers for spline

        min_n, max_n = min(min_n, n_actual), max(max_n, n_actual)
        norm_positions = np.linspace(0, 1, n_actual)
        for i, flux in enumerate(fluxes):
            n_list.append(n_actual)
            x_norm_list.append(norm_positions[i])
            flux_list.append(flux)

    if not n_list:
        raise ValueError("[Spline] No valid configuration data found.")

    num_data_points = len(flux_list)
    print(f"[Spline] Fitting using data from N={min_n} to N={max_n}. Total points: {num_data_points}")
    return np.array(n_list), np.array(x_norm_list), np.array(flux_list), min_n, max_n, num_data_points

def fit_and_predict_ratio(n_target, known_configs, actual_ppfds, target_ppfd, plot_fit=False):
    """
    Calculates the PPFD/Lumen ratio for known configs, fits multiple trend models (Linear, Quadratic, Exponential),
    plots the results, and predicts the ratio for n_target using the exponential model by default.

    Args:
        n_target (int): The target number of layers N.
        known_configs (dict): Dictionary of known successful flux configurations {N: flux_array}.
        actual_ppfds (dict): Dictionary of actual average PPFD achieved for some N {N: ppfd_value}.
        target_ppfd (float): The target average PPFD used for N values without actual results.
        plot_fit (bool): If True, displays a plot of the data and fitted curves.

    Returns:
        float: The predicted PPFD/Lumen ratio for n_target based on the exponential fit.
    """
    n_values_sorted = sorted(known_configs.keys())
    total_fluxes = [np.sum(known_configs[n]) for n in n_values_sorted]

    ratios = []
    valid_n = []

    print("\n[Ratio] Calculating PPFD/Lumen ratio and fitting trend vs N:")
    for i, n in enumerate(n_values_sorted):
        if total_fluxes[i] > 1e-6:
            ppfd_to_use = actual_ppfds.get(n, target_ppfd)
            ratio = ppfd_to_use / total_fluxes[i]
            ratios.append(ratio)
            valid_n.append(n)
            print(f"  N={n}: Total Flux={total_fluxes[i]:.1f}, PPFD={ppfd_to_use:.2f} (Source: {'Actual' if n in actual_ppfds else 'Target'}), Ratio={ratio:.6e}")
        else:
            print(f"  N={n}: Total Flux is zero, skipping.")

    if not ratios:
        raise ValueError("[Ratio] Could not calculate PPFD/Lumen ratio from any configuration.")

    valid_n_array = np.array(valid_n)
    ratios_array = np.array(ratios)

    # --- Fit Polynomial Models ---
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

    # --- Fit Exponential Decay Model ---
    # Define the exponential function: a * exp(-b * N) + c
    def exp_decay_func(n_vals, a, b, c):
        # Add constraint: b should be positive for decay
        b_constrained = max(1e-9, b) # Prevent b from being zero or negative
        return a * np.exp(-b_constrained * n_vals) + c

    coeffs_exp = None
    predicted_ratio_exp = None
    try:
        # Provide initial guesses (p0) - this can be important!
        # Guess 'c' as the minimum ratio (asymptotic value)
        # Guess 'a' as the difference between max and min ratio
        # Guess 'b' as small positive value (e.g., 0.1)
        initial_guess = [np.max(ratios_array) - np.min(ratios_array), 0.1, np.min(ratios_array)]
        coeffs_exp, _ = curve_fit(exp_decay_func, valid_n_array, ratios_array, p0=initial_guess, maxfev=5000)
        predicted_ratio_exp = exp_decay_func(n_target, *coeffs_exp)
        print(f"\n[Ratio] Exponential Fit (a*exp(-b*N)+c): a={coeffs_exp[0]:.4e}, b={coeffs_exp[1]:.4e}, c={coeffs_exp[2]:.4e}")
        print(f"[Ratio] Exponential Prediction for N={n_target}: {predicted_ratio_exp:.6e}")
    except Exception as e:
        print(f"[Ratio] Could not fit exponential model: {e}")
        predicted_ratio_exp = predicted_ratio_quad if predicted_ratio_quad is not None else (predicted_ratio_lin if predicted_ratio_lin is not None else np.mean(ratios_array)) # Fallback

    # --- Select Model to Return ---
    # For now, let's prioritize Exponential > Quadratic > Linear > Mean
    final_predicted_ratio = predicted_ratio_exp if predicted_ratio_exp is not None else \
                           (predicted_ratio_quad if predicted_ratio_quad is not None else \
                           (predicted_ratio_lin if predicted_ratio_lin is not None else np.mean(ratios_array)))

    print(f"\n[Ratio] Using prediction from Exponential model: {final_predicted_ratio:.6e}")


    # --- Plotting ---
    if plot_fit:
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_n_array, ratios_array, label='Data Points', color='red', zorder=5)

        # Generate points for plotting curves
        n_plot = np.linspace(min(valid_n_array), n_target + 1 , 100) # Extend slightly beyond target N

        if poly_func_quad:
            plt.plot(n_plot, poly_func_quad(n_plot), label=f'Quadratic Fit (Pred={predicted_ratio_quad:.4e})', linestyle='--')
        if poly_func_lin:
            plt.plot(n_plot, poly_func_lin(n_plot), label=f'Linear Fit (Pred={predicted_ratio_lin:.4e})', linestyle=':')
        if coeffs_exp is not None:
             plt.plot(n_plot, exp_decay_func(n_plot, *coeffs_exp), label=f'Exponential Fit (Pred={predicted_ratio_exp:.4e})', linestyle='-.')

        # Highlight the prediction point
        plt.scatter([n_target], [final_predicted_ratio], color='blue', marker='x', s=100, zorder=6, label=f'Selected Pred. N={n_target}')

        plt.xlabel("Number of Layers (N)")
        plt.ylabel("PPFD / Total Luminous Flux Ratio")
        plt.title("PPFD/Lumen Ratio vs. N and Fitted Models")
        plt.legend()
        plt.grid(True)
        plt.show(block=False) # Show plot without blocking
        plt.pause(0.1) # Allow time to render

    # --- Manual Override Section ---
    # manual_override_ratio = None
    # if n_target == 20:
    #     manual_override_ratio = 6.45e-3 # Based on Iteration 0 results (1180 PPFD / ~182k Lumens)
    #     print(f"[Ratio] MANUAL OVERRIDE for N={n_target}: Using ratio = {manual_override_ratio:.6e}")
    #
    # if manual_override_ratio is not None:
    #     final_predicted_ratio = manual_override_ratio
    # --- End Manual Override ---

    # --- End Manual Override ---
    # Sanity check the final predicted ratio
    if final_predicted_ratio <= 1e-9:
         warnings.warn(f"[Ratio] Final predicted ratio ({final_predicted_ratio:.4e}) is non-positive for N={n_target}. Using ratio from nearest known N.", UserWarning)
         try:
             nearest_n_idx = np.abs(valid_n_array - n_target).argmin()
             final_predicted_ratio = ratios_array[nearest_n_idx]
             print(f"  Using ratio from nearest N ({valid_n_array[nearest_n_idx]}): {final_predicted_ratio:.6e}")
         except IndexError:
              print("  Error finding nearest N ratio, using mean.")
              final_predicted_ratio = np.mean(ratios) if ratios else 1e-5 # Fallback non-zero


    return final_predicted_ratio

# Renamed function to reflect its role in initial prediction
def generate_initial_flux_prediction(num_layers_target, known_configs, actual_ppfds,
                                     target_ppfd=TARGET_PPFD, k=SPLINE_DEGREE,
                                     smoothing_mode=SMOOTHING_FACTOR_MODE, smoothing_mult=SMOOTHING_MULTIPLIER,
                                     clamp_outer=CLAMP_OUTER_LAYER, outer_max=OUTER_LAYER_MAX_FLUX,
                                     initial_ppfd_correction_factor=1.0):
    """
    Generates the initial (Iteration 0) flux prediction using spline, ratio fit, clamping,
    and an initial empirical correction factor.
    """
    print(f"\n--- Generating Initial Prediction for N={num_layers_target} ---")
    if num_layers_target < 2:
        raise ValueError("Number of layers must be at least 2.")

    # --- Step 1: Fit Spline to Known Data ---
    n_coords, x_norm_coords, flux_values, min_n_data, max_n_data, num_data_points = prepare_spline_data(known_configs)

    s_value = None
    if smoothing_mode == "num_points": s_value = float(num_data_points)
    elif smoothing_mode == "multiplier": s_value = float(num_data_points) * smoothing_mult
    elif isinstance(smoothing_mode, (int, float)): s_value = float(smoothing_mode)

    if s_value is not None: print(f"[Spline] Using smoothing factor s = {s_value:.2f}")
    else: print("[Spline] Using default smoothing factor s (estimated by FITPACK)")

    # Adjust spline degree k if insufficient data
    unique_n = len(np.unique(n_coords))
    unique_x = len(np.unique(x_norm_coords))
    if unique_n <= k or unique_x <= k:
         k_orig = k
         k = max(1, min(unique_n - 1, unique_x - 1, k))
         warnings.warn(f"[Spline] Insufficient unique coordinates ({unique_n} N, {unique_x} x) for degree {k_orig}. Reducing degree to {k}.", UserWarning)

    try:
        with warnings.catch_warnings():
             # Filter known warnings from SmoothBivariateSpline
             warnings.filterwarnings("ignore", message="The required storage space", category=UserWarning)
             warnings.filterwarnings("ignore", message="The number of knots required", category=UserWarning)
             warnings.filterwarnings("ignore", message="Warning: s=", category=UserWarning)
             warnings.filterwarnings("ignore", message="Warning: ier=.*", category=UserWarning) # Catch various ier flags
             spline = SmoothBivariateSpline(n_coords, x_norm_coords, flux_values, kx=k, ky=k, s=s_value)
        print(f"[Spline] Successfully fitted spline (requested degree k={k}).")
    except Exception as e:
         print(f"\n[Spline] Error fitting spline: {e}")
         raise

    # --- Step 2: Evaluate Spline for Target N to get Shape ---
    x_target_norm = np.linspace(0, 1, num_layers_target)
    n_target_array = np.full_like(x_target_norm, num_layers_target)
    interpolated_fluxes = spline(n_target_array, x_target_norm, grid=False)

    interpolated_fluxes = np.maximum(interpolated_fluxes, 0)

    # --- Apply Shape Modifier ---
    shape_modifier_power = 0.6 # *** Experiment with this value (e.g., 0.2, 0.3, 0.4) ***
    print(f"[Predict] Applying shape modifier: x^{shape_modifier_power}")
    x_target_norm = np.linspace(0, 1, num_layers_target)
    # Avoid 0^p issues if p is fractional, add small epsilon or handle x=0 separately
    modifier = np.power(np.maximum(x_target_norm, 1e-9), shape_modifier_power)
    # Optional: Normalize modifier so it doesn't drastically change total shape sum? Not necessary if rescaling later.
    interpolated_fluxes = interpolated_fluxes * modifier
    # Ensure non-negativity again
    interpolated_fluxes = np.maximum(interpolated_fluxes, 0)
    # --- End Shape Modifier ---

    spline_shape_total_flux = np.sum(interpolated_fluxes) # Recalculate total after modification
    print(f"[Spline] Modified shape total flux for N={num_layers_target}: {spline_shape_total_flux:.2f}")

    if spline_shape_total_flux <= 1e-6:
         warnings.warn("[Predict] Modified shape resulted in near-zero total flux. Cannot scale reliably.", UserWarning)
         # Maybe return the *unmodified* spline shape scaled arbitrarily?
         # Let's return the modified shape for now, likely all zeros.
         return interpolated_fluxes, 0.0
    
    if num_layers_target < min_n_data or num_layers_target > max_n_data:
        warnings.warn(f"[Spline] Extrapolating flux profile for N={num_layers_target} (data range {min_n_data}-{max_n_data}). Results less reliable.", UserWarning)

    # Ensure non-negativity
    interpolated_fluxes = np.maximum(interpolated_fluxes, 0)
    spline_shape_total_flux = np.sum(interpolated_fluxes)
    print(f"[Spline] Raw shape total flux for N={num_layers_target}: {spline_shape_total_flux:.2f}")

    if spline_shape_total_flux <= 1e-6:
         warnings.warn("[Predict] Spline evaluation resulted in near-zero total flux. Cannot scale reliably. Returning raw shape.", UserWarning)
         return interpolated_fluxes, 0.0 # Return shape and zero target flux

    # --- Step 3: Predict Target Total Flux ---
    # Get actual PPFDs for known N=16, 17, 18 (update this map as needed)
    # actual_ppfds = { 16: 1248.63, 17: 1246.87, 18: 1247.32 } # Provided in prompt, passed as argument

    predicted_ppfd_per_lumen = fit_and_predict_ratio(num_layers_target, known_configs, actual_ppfds, target_ppfd)
    if predicted_ppfd_per_lumen <= 1e-9:
         warnings.warn("[Predict] Predicted PPFD/Lumen ratio is non-positive. Cannot determine target flux.", UserWarning)
         # Return the raw shape, maybe scaled arbitrarily? Or just the shape.
         scale_factor = 100000.0 / spline_shape_total_flux # Arbitrary scaling
         return interpolated_fluxes * scale_factor, 100000.0

    target_total_flux_uncorrected = target_ppfd / predicted_ppfd_per_lumen
    # Apply the initial empirical correction factor passed to the function
    target_total_flux = target_total_flux_uncorrected * initial_ppfd_correction_factor
    print(f"[Predict] Target Total Flux (Uncorrected): {target_total_flux_uncorrected:.2f} lm")
    if abs(initial_ppfd_correction_factor - 1.0) > 1e-4:
        print(f"[Predict] Applying Initial Correction Factor: {initial_ppfd_correction_factor:.4f}")
    print(f"[Predict] Final Initial Target Total Flux: {target_total_flux:.2f} lm")


    # --- Step 4: Scale Spline Shape, Apply Clamping ---
    flux_initial = np.zeros_like(interpolated_fluxes)
    outer_clamp_value_actual = 0.0

    if clamp_outer:
        print(f"[Predict] Applying outer layer clamp: Layer {num_layers_target - 1} = {outer_max:.1f} lm")
        if outer_max >= target_total_flux:
             warnings.warn("[Predict] Outer layer clamp value >= target total flux. Setting inner layers to zero.", UserWarning)
             flux_initial[-1] = target_total_flux # Assign whole budget to outer
             outer_clamp_value_actual = target_total_flux
             # Inner layers remain zero
        else:
             # Assign clamp value to outer layer
             flux_initial[-1] = outer_max
             outer_clamp_value_actual = outer_max
             # Determine remaining flux budget for inner layers
             inner_flux_budget = target_total_flux - outer_clamp_value_actual
             if inner_flux_budget < 0: inner_flux_budget = 0 # Sanity check

             # Get sum of the spline shape for inner layers
             inner_shape_sum = np.sum(interpolated_fluxes[:-1])
             if inner_shape_sum <= 1e-6:
                  warnings.warn("[Predict] Sum of inner layer shapes from spline is near zero. Cannot scale inner layers.", UserWarning)
                  # Inner layers remain zero
             else:
                  # Scale inner layers based on their shape proportion and remaining budget
                  S_inner = inner_flux_budget / inner_shape_sum
                  print(f"[Predict] Scaling inner layers (0 to {num_layers_target - 2}) by factor: {S_inner:.6f}")
                  flux_initial[:-1] = S_inner * interpolated_fluxes[:-1]
    else:
        # No clamping - Apply global scaling
        print("[Predict] Applying global scaling factor (no outer clamp).")
        global_scale_factor = target_total_flux / spline_shape_total_flux
        print(f"[Predict] Global scaling factor: {global_scale_factor:.6f}")
        flux_initial = interpolated_fluxes * global_scale_factor
        outer_clamp_value_actual = flux_initial[-1] # Store the resulting outer value

    # Ensure non-negativity again after scaling
    flux_initial = np.maximum(flux_initial, 0)
    final_total_flux = np.sum(flux_initial)

    # Check if total flux matches target reasonably well
    if not np.isclose(final_total_flux, target_total_flux, rtol=1e-3):
         warnings.warn(f"[Predict] Initial predicted total flux ({final_total_flux:.2f}) differs slightly from target ({target_total_flux:.2f}).", UserWarning)

    print(f"[Predict] Initial Flux Profile Total: {final_total_flux:.2f}")
    return flux_initial, target_total_flux # Return fluxes and the target total flux used

def apply_inner_refinement_step(flux_input, ppfd_feedback_per_layer, target_ppfd, target_total_flux, outer_clamp_value, iteration, learn_rate=REFINEMENT_LEARNING_RATE, mult_min=MULTIPLIER_MIN, mult_max=MULTIPLIER_MAX):
    """
    Refines inner layer fluxes based on per-layer PPFD feedback, maintaining outer clamp and total flux.
    Assumes ppfd_feedback_per_layer is an array of size N (same as flux_input).
    """
    num_layers = len(flux_input)
    if ppfd_feedback_per_layer is None or len(ppfd_feedback_per_layer) != num_layers:
        warnings.warn(f"[Refine] PPFD feedback invalid for INNER Iteration {iteration}. Skipping refinement step.", UserWarning)
        return flux_input # Return unchanged input

    print(f"\n--- Applying INNER PPFD Refinement Iteration {iteration} ---")
    print(f"[Refine] Target Total Flux for this step: {target_total_flux:.2f} lm")
    print(f"[Refine] Outer Layer {num_layers-1} clamped to: {outer_clamp_value:.2f} lm")
    print(f"[Refine] Using Learning Rate: {learn_rate:.2f}")

    # Calculate errors relative to the target PPFD for feedback
    errors = target_ppfd - ppfd_feedback_per_layer
    # Normalize errors relative to target PPFD for consistent learning rate effect
    relative_errors = np.divide(errors, target_ppfd, out=np.zeros_like(errors), where=abs(target_ppfd)>1e-9)

    # Calculate multipliers based on relative errors and learning rate
    multipliers = 1.0 + learn_rate * relative_errors
    # Clip multipliers
    multipliers = np.clip(multipliers, mult_min, mult_max)

    print(f"[Refine] PPFD Feedback (Avg per Layer):\n{np.array2string(ppfd_feedback_per_layer, precision=2, suppress_small=True)}")
    print(f"[Refine] Refinement Multipliers (min={mult_min:.2f}, max={mult_max:.2f}):\n{np.array2string(multipliers, precision=4, suppress_small=True)}")

    # --- Apply multipliers ONLY to inner layers ---
    flux_inner_refined = flux_input[:-1] * multipliers[:-1] # Element-wise multiplication
    # Ensure non-negativity
    flux_inner_refined = np.maximum(flux_inner_refined, 0)

    # --- Rescale inner layers to meet the budget ---
    # Budget for inner layers = Target Total Flux - Outer Clamp Value
    inner_flux_budget = target_total_flux - outer_clamp_value
    if inner_flux_budget < 0:
         warnings.warn(f"[Refine] Target total flux ({target_total_flux:.2f}) < outer clamp ({outer_clamp_value:.2f}) in iter {iteration}. Setting inner flux budget to zero.", UserWarning)
         inner_flux_budget = 0

    refined_inner_total = np.sum(flux_inner_refined)
    print(f"[Refine] Refined Inner Flux Total (Before Rescale): {refined_inner_total:.2f}")
    print(f"[Refine] Inner Flux Budget: {inner_flux_budget:.2f}")


    # Calculate rescaling factor for inner layers
    if refined_inner_total <= 1e-6:
         warnings.warn(f"[Refine] Refined inner flux total near zero in iteration {iteration}. Cannot rescale inner layers.", UserWarning)
         # Keep inner fluxes as they are (likely near zero)
         final_fluxes_inner = flux_inner_refined
    else:
         rescale_inner_factor = inner_flux_budget / refined_inner_total
         print(f"[Refine] Inner rescale factor: {rescale_inner_factor:.6f}")
         final_fluxes_inner = flux_inner_refined * rescale_inner_factor
         # Ensure non-negativity after rescaling
         final_fluxes_inner = np.maximum(final_fluxes_inner, 0)

    # --- Combine inner and outer layers ---
    final_fluxes = np.zeros_like(flux_input)
    final_fluxes[:-1] = final_fluxes_inner
    final_fluxes[-1] = outer_clamp_value # Set the clamped outer layer value

    # Final check on total flux
    actual_total_flux_iter = np.sum(final_fluxes)
    if not np.isclose(actual_total_flux_iter, target_total_flux, rtol=1e-3):
         warnings.warn(f"[Refine] Total flux ({actual_total_flux_iter:.2f}) after inner refinement iter {iteration} differs slightly from target ({target_total_flux:.2f}). Check calculations.", UserWarning)
    print(f"[Refine] Refined Flux Total (Iter {iteration}): {actual_total_flux_iter:.2f}")

    return final_fluxes

# ==============================================================================
# Main Automation Workflow
# ==============================================================================

def run_automated_prediction(num_layers_target, target_ppfd, target_dou,
                             max_iterations, ppfd_tolerance,
                             sim_w, sim_l, sim_h):
    """
    Automates the Predict -> Simulate -> Refine cycle. Includes smoothed PPFD correction.

    Returns:
        tuple: (success_flag, final_fluxes, final_ppfd, final_dou, iterations_run)
    """
    print(f"===== Starting Automated Prediction for N={num_layers_target} =====")
    print(f"Target PPFD: {target_ppfd:.1f} +/- {ppfd_tolerance*100:.1f}%")
    print(f"Target DOU: > {target_dou:.1f}%")
    print(f"Max Iterations: {max_iterations}")

    # --- Pre-calculate Geometry (only once) ---
    print("\n[Setup] Preparing simulation geometry...")
    try:
        geometry_data = prepare_geometry(sim_w, sim_l, sim_h, num_layers_target)
        cob_positions, X_grid, Y_grid, _ = geometry_data # Unpack for later use
    except Exception as e:
        print(f"[Error] Failed to prepare geometry: {e}")
        return False, None, 0.0, 0.0, 0

    # --- Initialize PPFD Correction History ---
    correction_history = deque(maxlen=3) # Store last 3 correction factors
    # Get initial correction factor if it exists for N_target, else 1.0
    current_ppfd_correction = EMPIRICAL_PPFD_CORRECTION.get(num_layers_target, 1.0)
    correction_history.append(current_ppfd_correction) # Initialize history with starting factor
    print(f"[Setup] Initial PPFD Correction Factor: {current_ppfd_correction:.5f}")


    # --- Initial Prediction (Iteration 0) ---
    # Use actual PPFDs for known N=16, 17, 18 from the start for ratio fit
    actual_ppfds_known = { 16: 1248.63, 17: 1246.87, 18: 1247.32 } # From prompt

    try:
        # Pass the initial (potentially smoothed if history existed, but likely just the starting value) correction factor
        initial_smoothed_correction = np.mean(list(correction_history))
        current_fluxes, target_total_flux = generate_initial_flux_prediction(
            num_layers_target,
            known_configs,
            actual_ppfds_known,
            target_ppfd=target_ppfd,
            initial_ppfd_correction_factor=initial_smoothed_correction # Use current average
        )
    except Exception as e:
        print(f"[Error] Failed during initial flux prediction: {e}")
        import traceback
        traceback.print_exc()
        return False, None, 0.0, 0.0, 0

    # Store the outer clamp value used (or resulting value if not clamped)
    # If clamping is ON, generate_initial_flux_prediction enforces it.
    # The refinement step requires the clamp value explicitly.
    outer_clamp_value = OUTER_LAYER_MAX_FLUX if CLAMP_OUTER_LAYER else current_fluxes[-1]
    if CLAMP_OUTER_LAYER and target_total_flux > 0 and outer_clamp_value >= target_total_flux : # Check target_total_flux > 0
         outer_clamp_value = target_total_flux # Adjust if budget was limiting
         print(f"[Setup] Outer clamp value adjusted to target total flux: {outer_clamp_value:.2f}")


    final_fluxes = current_fluxes # Initialize final fluxes
    success = False
    floor_ppfd = None # Initialize floor_ppfd outside loop scope

    # --- Iteration Loop ---
    for i in range(max_iterations + 1): # Run iter 0 (initial) + max_iterations refinements
        print(f"\n======= Iteration {i} / {max_iterations} =======")
        # Ensure target_total_flux is reasonable before printing/using
        if target_total_flux < 0:
             print("[Warn] Target total flux became negative, capping at 0 for this iteration.")
             target_total_flux = 0
             # May need to adjust current_fluxes if target is zero?
             # For now, let the simulation run with current fluxes and refinement handle it.

        print(f"[Fluxes] Current Flux Profile (N={num_layers_target}):")
        # Only print first few and last few layers if N is large? For brevity.
        max_print_layers = 5
        if num_layers_target > 2 * max_print_layers:
             for l, flux in enumerate(current_fluxes[:max_print_layers]): print(f"    Layer {l}: {flux:.4f}")
             print("    ...")
             for l, flux in enumerate(current_fluxes[-max_print_layers:], start=num_layers_target-max_print_layers): print(f"    Layer {l}: {flux:.4f}")
        else:
             for l, flux in enumerate(current_fluxes): print(f"    Layer {l}: {flux:.4f}")

        print(f"    Total Flux: {np.sum(current_fluxes):.2f} (Target: {target_total_flux:.2f})")


        # --- Simulate ---
        start_sim_time = time.time()
        try:
            # Ensure fluxes are non-negative before passing to simulation
            current_fluxes = np.maximum(0, current_fluxes)
            floor_ppfd, _, _, _ = simulate_lighting(current_fluxes, geometry_data)
        except Exception as e:
            print(f"[Error] Simulation failed in Iteration {i}: {e}")
            import traceback
            traceback.print_exc()
            # Abort on simulation failure
            return False, current_fluxes, 0.0, 0.0, i
        end_sim_time = time.time()
        print(f"[Sim] Iteration {i} simulation time: {end_sim_time - start_sim_time:.2f}s")


        # --- Analyze Results ---
        avg_ppfd, dou = calculate_metrics(floor_ppfd)
        print(f"\n[Results] Iteration {i}:")
        print(f"  Average PPFD = {avg_ppfd:.2f} µmol/m²/s")
        print(f"  DOU (RMSE based) = {dou:.2f}%")

        # --- Check Criteria ---
        # Check avg_ppfd first to avoid issues if it's zero
        if avg_ppfd <= 1e-3: # If PPFD is effectively zero, criteria cannot be met
             ppfd_met = False
        else:
             ppfd_met = abs(avg_ppfd - target_ppfd) <= ppfd_tolerance * target_ppfd

        dou_met = dou >= target_dou

        if ppfd_met and dou_met:
            print(f"\n[Success] Criteria met after {i} iterations!")
            success = True
            final_fluxes = current_fluxes
            if SHOW_FINAL_HEATMAP:
                 plot_heatmap(floor_ppfd, X_grid, Y_grid, cob_positions, title=f"Final PPFD N={num_layers_target} (Iter {i})", annotation_step=ANNOTATION_STEP)
            break # Exit loop on success
        elif i == max_iterations:
            print(f"\n[Failure] Maximum iterations ({max_iterations}) reached. Criteria not met.")
            final_fluxes = current_fluxes # Keep the last result
            if SHOW_FINAL_HEATMAP:
                 plot_heatmap(floor_ppfd, X_grid, Y_grid, cob_positions, title=f"Final PPFD N={num_layers_target} (Iter {i} - Failed)", annotation_step=ANNOTATION_STEP)
            break # Exit loop on max iterations


        # --- Refine (if not last iteration and criteria not met) ---
        print("\n[Refine] Criteria not met, proceeding to refinement...")

        # Calculate per-layer PPFD feedback
        ppfd_feedback = calculate_per_layer_ppfd(floor_ppfd, X_grid, Y_grid, cob_positions, sim_w, sim_l, num_layers_target)
        if ppfd_feedback is None or np.any(np.isnan(ppfd_feedback)):
             print("[Error] Failed to get valid per-layer PPFD feedback (contains NaN). Aborting.")
             # Use metrics from current state as final, mark as failed
             final_avg_ppfd, final_dou = calculate_metrics(floor_ppfd)
             return False, current_fluxes, final_avg_ppfd, final_dou, i

        # --- Update Correction Factor (Smoothed) ---
        if avg_ppfd > 1e-6: # Avoid division by zero
             latest_correction = target_ppfd / avg_ppfd
             correction_history.append(latest_correction) # Add latest to history (deque handles maxlen)
             current_ppfd_correction = np.mean(list(correction_history)) # Use MEAN of history
             EMPIRICAL_PPFD_CORRECTION[num_layers_target] = latest_correction # Store the LATEST raw factor
             print(f"[Refine] Latest Correction Factor: {latest_correction:.5f}")
             print(f"[Refine] Smoothed PPFD Correction Factor (Avg of last {len(correction_history)}): {current_ppfd_correction:.5f}")
        else:
             warnings.warn("[Refine] Average PPFD is near zero, cannot update correction factor. Using previous smoothed value.", UserWarning)
             # current_ppfd_correction remains the mean of the existing history


        # --- Recalculate the target total flux for the *next* iteration ---
        # --- using the *smoothed* correction factor ---
        # Optimization: Get ratio once if it doesn't change based on known_configs
        predicted_ppfd_per_lumen = fit_and_predict_ratio(num_layers_target, known_configs, actual_ppfds_known, target_ppfd)
        if predicted_ppfd_per_lumen > 1e-9:
             target_total_flux_uncorrected = target_ppfd / predicted_ppfd_per_lumen
             target_total_flux = target_total_flux_uncorrected * current_ppfd_correction # Apply **SMOOTHED** correction
             # Sanity check: Ensure target total flux is non-negative
             if target_total_flux < 0:
                  warnings.warn(f"[Refine] Calculated target total flux ({target_total_flux:.2f}) is negative. Setting to 0.", UserWarning)
                  target_total_flux = 0
        else:
             warnings.warn("[Refine] PPFD/Lumen ratio is non-positive. Cannot recalculate target total flux, using previous value.", UserWarning)
             # target_total_flux remains unchanged, potentially problematic


        # --- Apply the refinement step ---
        try:
            # Ensure outer_clamp_value is not greater than the potentially reduced target_total_flux
            current_outer_clamp = min(outer_clamp_value, target_total_flux) if CLAMP_OUTER_LAYER else current_fluxes[-1] # Re-evaluate clamp for safety

            refined_fluxes = apply_inner_refinement_step(
                current_fluxes,
                ppfd_feedback,       # Use the calculated per-layer averages
                target_ppfd,         # Target PPFD for error calculation in refinement
                target_total_flux,   # Use the updated target total flux
                current_outer_clamp, # Pass the clamp value
                i + 1                # Iteration number for logging
            )
            # --- Remove or comment out the global call ---
            # refined_fluxes = apply_global_refinement_step(...)

            current_fluxes = refined_fluxes

        except Exception as e:
            print(f"[Error] Refinement failed in Iteration {i}: {e}")
            import traceback
            traceback.print_exc()
            # Use metrics from current state as final, mark as failed
            final_avg_ppfd, final_dou = calculate_metrics(floor_ppfd)
            return False, current_fluxes, final_avg_ppfd, final_dou, i

    # --- End of Loop ---
    # Recalculate metrics for the very last state reached (whether success or failure)
    final_avg_ppfd, final_dou = calculate_metrics(floor_ppfd)

    print("\n===== Automated Prediction Finished =====")
    print(f"Success: {success}")
    print(f"Iterations Run: {i}") # 'i' will be the last iteration number run (or max_iterations)
    print(f"Final Average PPFD: {final_avg_ppfd:.2f}")
    print(f"Final DOU: {final_dou:.2f}")
    print("Final Flux Assignments:")
    if num_layers_target > 2 * max_print_layers:
        for l, flux in enumerate(final_fluxes[:max_print_layers]): print(f"    {flux:.4f}, # Layer {l}")
        print("    ...")
        for l, flux in enumerate(final_fluxes[-max_print_layers:], start=num_layers_target-max_print_layers): print(f"    {flux:.4f}, # Layer {l}")
    else:
        for l, flux in enumerate(final_fluxes): print(f"    {flux:.4f}, # Layer {l}")
    print(f"    Total: {np.sum(final_fluxes):.2f}")

    return success, final_fluxes, final_avg_ppfd, final_dou, i

# ==============================================================================
# Script Execution
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()

    success, final_fluxes, final_ppfd, final_dou, iters = run_automated_prediction(
        NUM_LAYERS_TARGET,
        TARGET_PPFD,
        TARGET_DOU,
        MAX_ITERATIONS,
        PPFD_TOLERANCE,
        SIM_W, SIM_L, SIM_H
    )

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")

    # Optional: Save final successful fluxes to a file or known_configs
    # if success:
    #     # Example: Add to known_configs (in memory for this run)
    #     # known_configs[NUM_LAYERS_TARGET] = final_fluxes
    #     # print(f"Added successful N={NUM_LAYERS_TARGET} config to known_configs for this session.")
    #     # Example: Save to CSV
    #     try:
    #         with open(f"successful_flux_N{NUM_LAYERS_TARGET}.csv", 'w', newline='') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(['Layer', 'Flux'])
    #             for i, flux in enumerate(final_fluxes):
    #                 writer.writerow([i, flux])
    #         print(f"Saved successful fluxes to successful_flux_N{NUM_LAYERS_TARGET}.csv")
    #     except Exception as e:
    #         print(f"Error saving final fluxes to CSV: {e}")

    # Keep plots open at the end if shown
    if SHOW_FINAL_HEATMAP and plt.get_fignums():
        print("\nClose plot window(s) to exit.")
        plt.show(block=True)
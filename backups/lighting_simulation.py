#!/usr/bin/env python3
"""
Lighting Simulation Script

This stand-alone script simulates lighting in a room using pre-built geometry.
All machine learning, surrogate modeling, progress callbacks, and visualization
functions have been removed.

Usage: Run the script directly to see summary PPFD results.
"""

import math
import numpy as np
from numba import njit
from functools import lru_cache

# ------------------------------
# Configuration & Constants
# ------------------------------
REFL_WALL = 0.8
REFL_CEIL = 0.8
REFL_FLOOR = 0.1

LUMINOUS_EFFICACY = 182.0         # lm/W
SPD_FILE = "./backups/spd_data.csv"

# Adaptive Radiosity parameters
MAX_RADIOSITY_BOUNCES = 10
RADIOSITY_CONVERGENCE_THRESHOLD = 1e-3

WALL_SUBDIVS_X = 10
WALL_SUBDIVS_Y = 5
CEIL_SUBDIVS_X = 10
CEIL_SUBDIVS_Y = 10

# Floor grid resolution (meters)
FLOOR_GRID_RES = 0.08

# Default candidate layers count (determines COB arrangement)
FIXED_NUM_LAYERS = 6

MC_SAMPLES = 16  # Number of Monte Carlo samples for indirect contributions

# ------------------------------
# SPD Conversion Factor
# ------------------------------
def compute_conversion_factor(spd_file):
    try:
        spd = np.loadtxt(spd_file, delimiter=' ', skiprows=1)
    except Exception as e:
        print("Error loading SPD data:", e)
        return 0.0138
    wl = spd[:, 0]
    intens = spd[:, 1]
    mask_par = (wl >= 400) & (wl <= 700)
    tot = np.trapz(intens, wl)
    tot_par = np.trapz(intens[mask_par], wl[mask_par])
    PAR_fraction = tot_par / tot if tot > 0 else 1.0

    wl_m = wl * 1e-9
    h, c, N_A = 6.626e-34, 3.0e8, 6.022e23
    numerator = np.trapz(wl_m[mask_par] * intens[mask_par], wl_m[mask_par])
    denominator = np.trapz(intens[mask_par], wl_m[mask_par]) or 1e-15
    lambda_eff = numerator / denominator
    E_photon = (h * c / lambda_eff) if lambda_eff > 0 else 1.0
    conversion_factor = (1.0 / E_photon) * (1e6 / N_A) * PAR_fraction
    print(f"[INFO] SPD: PAR fraction={PAR_fraction:.3f}, conv_factor={conversion_factor:.5f} µmol/J")
    return conversion_factor

CONVERSION_FACTOR = compute_conversion_factor(SPD_FILE)

# ------------------------------
# Geometry Preparation Functions
# ------------------------------
def prepare_geometry(W, L, H):
    """
    Build and cache geometry used in the simulation:
      - COB positions
      - Floor grid (X,Y)
      - Patches (centers, areas, normals, reflectances)
    Returns a tuple: (cob_positions, X, Y, (patch_centers, patch_areas, patch_normals, patch_refl)).
    """
    cob_positions = build_cob_positions(W, L, H)
    X, Y = build_floor_grid(W, L)
    patches = build_patches(W, L, H)
    return (cob_positions, X, Y, patches)

def build_cob_positions(W, L, H):
    """
    Builds COB positions arranged in a diamond pattern (rotated 45°)
    such that the overall array tightly fills the room dimensions.
    
    Instead of converting to feet, we now use FIXED_NUM_LAYERS to determine
    the number of layers (with layer 0 at the center). The outermost layer 
    (n = FIXED_NUM_LAYERS - 1) is scaled so that, after a 45° rotation,
    the COB array spans nearly the full room (from 0 to W in x and 0 to L in y).
    """
    n = FIXED_NUM_LAYERS - 1  # outermost layer index; total layers = FIXED_NUM_LAYERS
    positions = []
    positions.append((0, 0, H, 0))
    for i in range(1, n + 1):
        for x in range(-i, i + 1):
            y_abs = i - abs(x)
            if y_abs == 0:
                positions.append((x, 0, H, i))
            else:
                positions.append((x, y_abs, H, i))
                positions.append((x, -y_abs, H, i))
    
    theta = math.radians(45)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    centerX = W / 2
    centerY = L / 2
    # Scale factors chosen so that the farthest point (n,0) maps to the room edge:
    # For (n,0) after rotation: rx = n/sqrt2, so we set scale_x such that centerX + (n/sqrt2)*scale_x = W.
    scale_x = (W/2 * math.sqrt(2)) / n
    scale_y = (L/2 * math.sqrt(2)) / n
    
    transformed = []
    for (x, y, h, layer) in positions:
        rx = x * cos_t - y * sin_t
        ry = x * sin_t + y * cos_t
        px = centerX + rx * scale_x
        py = centerY + ry * scale_y
        # COBs are placed slightly below the ceiling (95% of H)
        transformed.append((px, py, H * 0.95, layer))
    
    return np.array(transformed, dtype=np.float64)

def pack_luminous_flux_dynamic(params, cob_positions):
    led_intensities = []
    for pos in cob_positions:
        layer = int(pos[3])
        intensity = params[layer]
        led_intensities.append(intensity)
    return np.array(led_intensities, dtype=np.float64)

@lru_cache(maxsize=32)
def cached_build_floor_grid(W: float, L: float):
    num_x = int(round(W / FLOOR_GRID_RES)) + 1
    num_y = int(round(L / FLOOR_GRID_RES)) + 1
    xs = np.linspace(0, W, num_x)
    ys = np.linspace(0, L, num_y)
    X, Y = np.meshgrid(xs, ys)
    return X, Y

def build_floor_grid(W, L):
    return cached_build_floor_grid(W, L)

@lru_cache(maxsize=32)
def cached_build_patches(W: float, L: float, H: float):
    patch_centers = []
    patch_areas = []
    patch_normals = []
    patch_refl = []
    
    # Floor
    patch_centers.append((W/2, L/2, 0.0))
    patch_areas.append(W * L)
    patch_normals.append((0.0, 0.0, 1.0))
    patch_refl.append(REFL_FLOOR)
    
    # Ceiling
    xs_ceiling = np.linspace(0, W, CEIL_SUBDIVS_X + 1)
    ys_ceiling = np.linspace(0, L, CEIL_SUBDIVS_Y + 1)
    for i in range(CEIL_SUBDIVS_X):
        for j in range(CEIL_SUBDIVS_Y):
            cx = (xs_ceiling[i] + xs_ceiling[i+1]) / 2
            cy = (ys_ceiling[j] + ys_ceiling[j+1]) / 2
            area = (xs_ceiling[i+1] - xs_ceiling[i]) * (ys_ceiling[j+1] - ys_ceiling[j])
            patch_centers.append((cx, cy, H + 0.01))
            patch_areas.append(area)
            patch_normals.append((0.0, 0.0, -1.0))
            patch_refl.append(REFL_CEIL)
    
    # Walls
    xs_wall = np.linspace(0, W, WALL_SUBDIVS_X + 1)
    zs_wall = np.linspace(0, H, WALL_SUBDIVS_Y + 1)
    
    # Front wall (y=0)
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cx = (xs_wall[i] + xs_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (xs_wall[i+1]-xs_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((cx, 0.0, cz))
            patch_areas.append(area)
            patch_normals.append((0.0, -1.0, 0.0))
            patch_refl.append(REFL_WALL)
    
    # Back wall (y=L)
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cx = (xs_wall[i] + xs_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (xs_wall[i+1]-xs_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((cx, L, cz))
            patch_areas.append(area)
            patch_normals.append((0.0, 1.0, 0.0))
            patch_refl.append(REFL_WALL)
    
    ys_wall = np.linspace(0, L, WALL_SUBDIVS_X + 1)
    # Left wall (x=0)
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cy = (ys_wall[i] + ys_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (ys_wall[i+1]-ys_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((0.0, cy, cz))
            patch_areas.append(area)
            patch_normals.append((-1.0, 0.0, 0.0))
            patch_refl.append(REFL_WALL)
    
    # Right wall (x=W)
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cy = (ys_wall[i] + ys_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (ys_wall[i+1]-ys_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((W, cy, cz))
            patch_areas.append(area)
            patch_normals.append((1.0, 0.0, 0.0))
            patch_refl.append(REFL_WALL)
    
    return (np.array(patch_centers, dtype=np.float64),
            np.array(patch_areas, dtype=np.float64),
            np.array(patch_normals, dtype=np.float64),
            np.array(patch_refl, dtype=np.float64))

def build_patches(W, L, H):
    return cached_build_patches(W, L, H)

# ------------------------------
# Lighting Simulation Functions
# ------------------------------
@njit
def compute_direct_floor(light_positions, light_fluxes, X, Y):
    min_dist2 = (FLOOR_GRID_RES / 2.0) ** 2
    out = np.zeros_like(X, dtype=np.float64)
    rows, cols = X.shape
    for r in range(rows):
        for c in range(cols):
            fx = X[r, c]
            fy = Y[r, c]
            val = 0.0
            for k in range(light_positions.shape[0]):
                lx = light_positions[k, 0]
                ly = light_positions[k, 1]
                lz = light_positions[k, 2]
                dx = fx - lx
                dy = fy - ly
                dz = -lz
                d2 = dx*dx + dy*dy + dz*dz
                if d2 < min_dist2:
                    d2 = min_dist2
                dist = math.sqrt(d2)
                cos_th = -dz / dist
                if cos_th < 0:
                    cos_th = 0.0
                val += (light_fluxes[k] / math.pi) * (cos_th / d2)
            out[r, c] = val
    return out

def compute_patch_direct(light_positions, light_fluxes, patch_centers, patch_normals, patch_areas):
    min_dist2 = (FLOOR_GRID_RES / 2.0) ** 2
    Np = patch_centers.shape[0]
    out = np.zeros(Np, dtype=np.float64)
    for ip in range(Np):
        pc = patch_centers[ip]
        n = patch_normals[ip]
        norm_n = math.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])
        accum = 0.0
        for j in range(light_positions.shape[0]):
            lx, ly, lz = light_positions[j, :3]
            dx = pc[0] - lx
            dy = pc[1] - ly
            dz = pc[2] - lz
            d2 = dx*dx + dy*dy + dz*dz
            if d2 < min_dist2:
                d2 = min_dist2
            dist = math.sqrt(d2)
            cos_th_led = abs(dz) / dist
            E_led = (light_fluxes[j] / math.pi) * (cos_th_led / d2)
            dot_patch = -(dx*n[0] + dy*n[1] + dz*n[2])
            cos_in_patch = max(dot_patch/(dist*norm_n), 0)
            accum += E_led * cos_in_patch
        out[ip] = accum
    return out

@njit
def iterative_radiosity_loop(patch_centers, patch_normals, patch_direct, patch_areas, patch_refl, max_bounces, convergence_threshold):
    Np = patch_direct.shape[0]
    patch_rad = patch_direct.copy()
    epsilon = 1e-6
    for bounce in range(max_bounces):
        new_flux = np.zeros(Np, dtype=np.float64)
        for j in range(Np):
            if patch_refl[j] <= 0:
                continue
            outF = patch_rad[j] * patch_areas[j] * patch_refl[j]
            pj = patch_centers[j]
            nj = patch_normals[j]
            norm_nj = math.sqrt(nj[0]*nj[0] + nj[1]*nj[1] + nj[2]*nj[2])
            for i in range(Np):
                if i == j:
                    continue
                pi = patch_centers[i]
                ni = patch_normals[i]
                norm_ni = math.sqrt(ni[0]*ni[0] + ni[1]*ni[1] + ni[2]*ni[2])
                dx = pi[0] - pj[0]
                dy = pi[1] - pj[1]
                dz = pi[2] - pj[2]
                dist2 = dx*dx + dy*dy + dz*dz
                if dist2 < 1e-15:
                    continue
                dist = math.sqrt(dist2)
                dot_j = (nj[0]*dx + nj[1]*dy + nj[2]*dz)
                cos_j = dot_j/(dist*norm_nj)
                if cos_j < 0:
                    continue
                dot_i = -(ni[0]*dx + ni[1]*dy + ni[2]*dz)
                cos_i = dot_i/(dist*norm_ni)
                if cos_i < 0:
                    continue
                ff = (cos_j * cos_i) / (math.pi * dist2)
                new_flux[i] += outF * ff
        new_patch_rad = np.empty_like(patch_rad)
        max_rel_change = 0.0
        for i in range(Np):
            new_patch_rad[i] = patch_direct[i] + new_flux[i] / patch_areas[i]
            change = abs(new_patch_rad[i] - patch_rad[i])
            denom = abs(patch_rad[i]) + epsilon
            rel_change = change / denom
            if rel_change > max_rel_change:
                max_rel_change = rel_change
        patch_rad = new_patch_rad.copy()
        if max_rel_change < convergence_threshold:
            break
    return patch_rad

def compute_reflection_on_floor(X, Y, patch_centers, patch_normals, patch_areas, patch_rad, patch_refl, mc_samples=MC_SAMPLES):
    """
    Computes indirect irradiance on the floor using Monte Carlo integration
    to estimate the form factors from each patch to each floor grid point.
    """
    rows, cols = X.shape
    out = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            fx = X[r, c]
            fy = Y[r, c]
            val = 0.0
            for p in range(patch_centers.shape[0]):
                if patch_refl[p] <= 0:
                    continue
                outF = patch_rad[p] * patch_areas[p] * patch_refl[p]
                pc = patch_centers[p]
                n = np.array(patch_normals[p])
                # Determine tangent vectors for the patch plane
                if abs(n[2]) < 0.99:
                    tangent1 = np.array([-n[1], n[0], 0.0])
                else:
                    tangent1 = np.array([1.0, 0.0, 0.0])
                tangent1_norm = np.linalg.norm(tangent1)
                if tangent1_norm > 0:
                    tangent1 = tangent1 / tangent1_norm
                else:
                    tangent1 = np.array([1.0, 0.0, 0.0])
                tangent2 = np.cross(n, tangent1)
                tangent2_norm = np.linalg.norm(tangent2)
                if tangent2_norm > 0:
                    tangent2 = tangent2 / tangent2_norm
                else:
                    tangent2 = np.array([0.0, 1.0, 0.0])
                half_side = math.sqrt(patch_areas[p]) / 2.0
                sample_sum = 0.0
                for _ in range(mc_samples):
                    offset1 = np.random.uniform(-half_side, half_side)
                    offset2 = np.random.uniform(-half_side, half_side)
                    sample_point = np.array(pc) + offset1 * tangent1 + offset2 * tangent2
                    dx = fx - sample_point[0]
                    dy = fy - sample_point[1]
                    dz = -sample_point[2]
                    dist2 = dx*dx + dy*dy + dz*dz
                    if dist2 < 1e-15:
                        continue
                    dist = math.sqrt(dist2)
                    cos_f = -dz/dist if (-dz/dist) > 0 else 0.0
                    dot_p = dx*n[0] + dy*n[1] + dz*n[2]
                    cos_p = dot_p/(dist* np.linalg.norm(n))
                    if cos_p < 0:
                        cos_p = 0.0
                    ff = (cos_p * cos_f) / (math.pi * dist2)
                    sample_sum += ff
                avg_ff = sample_sum / mc_samples
                val += outF * avg_ff
            out[r, c] = val
    return out

def simulate_lighting(params, geo):
    """
    Simulates the lighting using pre-built geometry and given lighting parameters.
    Returns a 2D floor PPFD array.
    """
    cob_positions, X, Y, (p_centers, p_areas, p_normals, p_refl) = geo

    # Compute LED intensities (lumens per layer)
    led_intensities = pack_luminous_flux_dynamic(params, cob_positions)
    power_arr = led_intensities / LUMINOUS_EFFICACY

    # Direct irradiance on floor
    direct_irr = compute_direct_floor(cob_positions, power_arr, X, Y)
    # Calculate irradiance from patches (direct + radiosity)
    patch_direct = compute_patch_direct(cob_positions, power_arr, p_centers, p_normals, p_areas)
    patch_rad = iterative_radiosity_loop(p_centers, p_normals, patch_direct, p_areas, p_refl,
                                         MAX_RADIOSITY_BOUNCES, RADIOSITY_CONVERGENCE_THRESHOLD)
    reflect_irr = compute_reflection_on_floor(X, Y, p_centers, p_normals, p_areas, patch_rad, p_refl)

    return (direct_irr + reflect_irr) * CONVERSION_FACTOR

# ------------------------------
# Main Simulation Entry Point
# ------------------------------
def main():
    # Room dimensions (meters)
    W = 3.6576   # Room width
    L = 3.6576   # Room length
    H = 0.9144      # Room height

    # Default lighting parameters for each layer (in lumens)
    #params = np.array([8000.0] * FIXED_NUM_LAYERS, dtype=np.float64)

    # With FIXED_NUM_LAYERS = 6, we define luminous flux per layer as follows:
    #  - Center COB (Layer 1, index 0):  1000.00 lumens
    #  - Layer 2 (index 1):              1000.00 lumens
    #  - Layer 3 (index 2):              5016.69 lumens
    #  - Layer 4 (index 3):              1000.00 lumens
    #  - Layer 5 (index 4):              5564.06 lumens
    #  - Layer 6 (index 5):              24000.00 lumens
    params = np.array([1000.00, 1000.00, 5016.69, 1000.00, 5564.06, 24000.00], dtype=np.float64)

    # Prepare geometry for the room
    geo = prepare_geometry(W, L, H)

    # Run the lighting simulation
    floor_ppfd = simulate_lighting(params, geo)

    mean_ppfd = np.mean(floor_ppfd)
    mad = np.mean(np.abs(floor_ppfd - mean_ppfd))
    rmse = np.sqrt(np.mean((floor_ppfd - mean_ppfd)**2))
    dou = 100 * (1 - rmse / mean_ppfd)
    cv = 100 * (np.std(floor_ppfd) / mean_ppfd)

    # Compute and print summary statistics
    mean_ppfd = np.mean(floor_ppfd)
    print(f"Average PPFD: {mean_ppfd:.2f} µmol/m²/s")
    print("Floor PPFD distribution:")
    print(floor_ppfd)
    print(f"MAD: {mad:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"DOU (%): {dou:.2f}")
    print(f"CV (%): {cv:.2f}")

if __name__ == "__main__":
    main()

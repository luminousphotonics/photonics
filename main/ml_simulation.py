#!/usr/bin/env python3
"""
Lighting Optimization Script (ml_simulation.py)
Now supports dynamic COB positioning and layer‐specific luminous flux parameters.
"""

import math
import numpy as np
from scipy.optimize import minimize
import sys
from numba import njit
from functools import lru_cache
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D  # just for side-effects


# ------------------------------
# Configuration & Constants
# ------------------------------

REFL_WALL = 0.08
REFL_CEIL = 0.1
REFL_FLOOR = 0.0

LUMINOUS_EFFICACY = 182.0         # lm/W
SPD_FILE = "./backups/spd_data.csv"

NUM_RADIOSITY_BOUNCES = 2
WALL_SUBDIVS_X = 10
WALL_SUBDIVS_Y = 5
CEIL_SUBDIVS_X = 10
CEIL_SUBDIVS_Y = 10

# Coarser floor grid resolution (meters)
FLOOR_GRID_RES = 0.05

MIN_LUMENS = 2000.0
MAX_LUMENS_MAIN = 24000.0
MAX_LUMENS_CORNER = 36000.0

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
# Dynamic COB Positioning Functions
# ------------------------------
def build_cob_positions(W, L, H):
    """
    Generate COB positions dynamically based on floor dimensions.
    W, L: floor width and length in meters.
    H: COB plane height in meters.
    
    First, compute n = max(1, floor(floor_width_ft/2) - 1),
    where floor_width_ft = W in feet.
    Then, for each layer i from 0 to n:
      - layer 0 (center): count = 1
      - for i >= 1, count = 4*i (all integer (x,y) with |x|+|y|=i)
    Rotate the grid by 45° and scale so that the outer COBs nearly reach the floor edge.
    
    Returns an array of shape (num_COBs, 4) with columns [x, y, H, layer].
    """
    ft2m = 3.28084
    floor_width_ft = W * ft2m
    n = max(1, int(floor_width_ft / 2) - 1)
    
    positions = []
    # Center (layer 0)
    positions.append((0, 0, H, 0))
    
    for i in range(1, n+1):
        for x in range(-i, i+1):
            y_abs = i - abs(x)
            if y_abs == 0:
                positions.append((x, 0, H, i))
            else:
                positions.append((x, y_abs, H, i))
                positions.append((x, -y_abs, H, i))
    
    # Rotate by 45°.
    theta = math.radians(45)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    # Scale: outermost point (n,0) rotates to (n/√2, n/√2); want its distance = ~95% of half the floor width.
    desired_max = (W / 2) * 0.95
    scale = (desired_max * math.sqrt(2)) / n
    centerX = W / 2
    centerY = L / 2
    transformed = []
    for (x, y, h, layer) in positions:
        rx = x * cos_t - y * sin_t
        ry = x * sin_t + y * cos_t
        px = centerX + rx * scale
        py = centerY + ry * scale
        transformed.append((px, py, h, layer))
    return np.array(transformed, dtype=np.float64)

def pack_luminous_flux_dynamic(params, cob_positions):
    """
    Given optimization parameters (one per layer) and the COB positions array (with column 3 as layer),
    assign each COB the intensity corresponding to its layer.
    If a COB's layer index exceeds the parameter array length, use the last parameter.
    """
    led_intensities = []
    for pos in cob_positions:
        layer = int(pos[3])
        if layer < len(params):
            intensity = params[layer]
        else:
            intensity = params[-1]
        led_intensities.append(intensity)
    return np.array(led_intensities, dtype=np.float64)

# ------------------------------
# Floor Grid & Direct Irradiance
# ------------------------------
@lru_cache(maxsize=32)
def cached_build_floor_grid(W: float, L: float):
    xs = np.arange(0, W, FLOOR_GRID_RES)
    ys = np.arange(0, L, FLOOR_GRID_RES)
    X, Y = np.meshgrid(xs, ys)
    return X, Y

def build_floor_grid(W, L):
    return cached_build_floor_grid(W, L)

# ------------------------------
# Floor Grid & Direct Irradiance
# ------------------------------
@njit
def compute_direct_floor(light_positions, light_fluxes, X, Y):
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
                dist2 = dx*dx + dy*dy + dz*dz
                if dist2 < 1e-15:
                    continue
                dist = math.sqrt(dist2)
                cos_th = -dz / dist
                if cos_th < 0:
                    cos_th = 0.0
                val += (light_fluxes[k] / math.pi) * (cos_th / dist2)
            out[r, c] = val
    return out

# ------------------------------
# Surface Patch Functions
# ------------------------------
@lru_cache(maxsize=32)
def cached_build_patches(W: float, L: float, H: float):
    # (As before – unchanged.)
    patch_centers = []
    patch_areas = []
    patch_normals = []
    patch_refl = []
    patch_centers.append((W/2, L/2, 0.0))
    patch_areas.append(W * L)
    patch_normals.append((0.0, 0.0, 1.0))
    patch_refl.append(REFL_FLOOR)
    dx_c = W / CEIL_SUBDIVS_X
    dy_c = L / CEIL_SUBDIVS_Y
    for i in range(CEIL_SUBDIVS_X):
        for j in range(CEIL_SUBDIVS_Y):
            cx = (i + 0.5) * dx_c
            cy = (j + 0.5) * dy_c
            patch_centers.append((cx, cy, H))
            patch_areas.append(dx_c * dy_c)
            patch_normals.append((0.0, 0.0, -1.0))
            patch_refl.append(REFL_CEIL)
    dx = W / WALL_SUBDIVS_X
    dz = H / WALL_SUBDIVS_Y
    for ix in range(WALL_SUBDIVS_X):
        for iz in range(WALL_SUBDIVS_Y):
            px = (ix + 0.5) * dx
            py = 0.0
            pz = (iz + 0.5) * dz
            patch_centers.append((px, py, pz))
            patch_areas.append(dx * dz)
            patch_normals.append((0.0, -1.0, 0.0))
            patch_refl.append(REFL_WALL)
    for ix in range(WALL_SUBDIVS_X):
        for iz in range(WALL_SUBDIVS_Y):
            px = (ix + 0.5) * dx
            py = L
            pz = (iz + 0.5) * dz
            patch_centers.append((px, py, pz))
            patch_areas.append(dx * dz)
            patch_normals.append((0.0, 1.0, 0.0))
            patch_refl.append(REFL_WALL)
    dy = L / WALL_SUBDIVS_X
    for iy in range(WALL_SUBDIVS_X):
        for iz in range(WALL_SUBDIVS_Y):
            px = 0.0
            py = (iy + 0.5) * dy
            pz = (iz + 0.5) * dz
            patch_centers.append((px, py, pz))
            patch_areas.append(dy * dz)
            patch_normals.append((-1.0, 0.0, 0.0))
            patch_refl.append(REFL_WALL)
    for iy in range(WALL_SUBDIVS_X):
        for iz in range(WALL_SUBDIVS_Y):
            px = W
            py = (iy + 0.5) * dy
            pz = (iz + 0.5) * dz
            patch_centers.append((px, py, pz))
            patch_areas.append(dy * dz)
            patch_normals.append((1.0, 0.0, 0.0))
            patch_refl.append(REFL_WALL)
    return (np.array(patch_centers, dtype=np.float64),
            np.array(patch_areas, dtype=np.float64),
            np.array(patch_normals, dtype=np.float64),
            np.array(patch_refl, dtype=np.float64))

def build_patches(W, L, H):
    return cached_build_patches(W, L, H)

def compute_patch_direct(light_positions, light_fluxes, patch_centers, patch_normals, patch_areas):
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
            dist2 = dx*dx + dy*dy + dz*dz
            if dist2 < 1e-15:
                continue
            dist = math.sqrt(dist2)
            cos_th_led = -dz/dist
            if cos_th_led < 0:
                continue
            E_led = (light_fluxes[j]/math.pi)*(cos_th_led/dist2)
            dot_patch = -(dx*n[0] + dy*n[1] + dz*n[2])
            cos_in_patch = dot_patch/(dist*norm_n)
            if cos_in_patch < 0:
                cos_in_patch = 0
            accum += E_led*cos_in_patch
        out[ip] = accum
    return out


@njit
def iterative_radiosity_loop(patch_centers, patch_normals, patch_direct, patch_areas, patch_refl, num_bounces):
    Np = patch_direct.shape[0]
    patch_rad = patch_direct.copy()
    for _ in range(num_bounces):
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
        patch_rad = patch_direct + new_flux/patch_areas
    return patch_rad

@njit
def compute_reflection_on_floor(X, Y, patch_centers, patch_normals, patch_areas, patch_rad, patch_refl):
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
                dx = fx - pc[0]
                dy = fy - pc[1]
                dz = -pc[2]
                dist2 = dx*dx + dy*dy + dz*dz
                if dist2 < 1e-15:
                    continue
                dist = math.sqrt(dist2)
                n = patch_normals[p]
                norm_n = math.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])
                dot_p = dx*n[0] + dy*n[1] + dz*n[2]
                cos_p = dot_p/(dist*norm_n)
                if cos_p < 0:
                    cos_p = 0
                cos_f = -dz/dist
                if cos_f < 0:
                    cos_f = 0
                ff = (cos_p * cos_f) / (math.pi * dist2)
                val += outF * ff
            out[r, c] = val
    return out

# ------------------------------
# Simulation & Optimization
# ------------------------------
def simulate_lighting(params, W, L, H):
    cob_positions = build_cob_positions(W, L, H)
    led_intensities = pack_luminous_flux_dynamic(params, cob_positions)
    power_arr = led_intensities / LUMINOUS_EFFICACY
    X, Y = build_floor_grid(W, L)
    direct_irr = compute_direct_floor(cob_positions, power_arr, X, Y)
    patch_centers, patch_areas, patch_normals, patch_refl = build_patches(W, L, H)
    patch_direct = compute_patch_direct(cob_positions, power_arr, patch_centers, patch_normals, patch_areas)
    patch_rad = iterative_radiosity_loop(patch_centers, patch_normals, patch_direct, patch_areas, patch_refl, NUM_RADIOSITY_BOUNCES)
    reflect_irr = compute_reflection_on_floor(X, Y, patch_centers, patch_normals, patch_areas, patch_rad, patch_refl)
    floor_irr = direct_irr + reflect_irr
    floor_ppfd = floor_irr * CONVERSION_FACTOR
    return floor_ppfd

def objective_function(params, W, L, H, target_ppfd):
    floor_ppfd = simulate_lighting(params, W, L, H)
    mean_ppfd = np.mean(floor_ppfd)
    mad = np.mean(np.abs(floor_ppfd - mean_ppfd))
    ppfd_penalty = (mean_ppfd - target_ppfd) ** 2
    return mad + 2.0 * ppfd_penalty

def optimize_lighting(W, L, H, target_ppfd, progress_callback=None):
    cob_positions = build_cob_positions(W, L, H)
    n = int(np.max(cob_positions[:, 3]))  # maximum layer index
    # x0: one value per layer; use 8000 for inner layers and 18000 for the outermost.
    x0 = np.array([8000] * n + [18000], dtype=np.float64)
    bounds = [(MIN_LUMENS, MAX_LUMENS_MAIN)] * n + [(MIN_LUMENS, MAX_LUMENS_CORNER)]
    total_iterations_estimate = 500
    iteration = 0

    def wrapped_obj(p):
        nonlocal iteration
        iteration += 1
        val = objective_function(p, W, L, H, target_ppfd)
        floor_ppfd = simulate_lighting(p, W, L, H)
        mp = np.mean(floor_ppfd)
        msg = f"[DEBUG] param={p}, mean_ppfd={mp:.1f}, obj={val:.3f}"
        if progress_callback:
            progress_pct = min(100, (iteration / total_iterations_estimate) * 100)
            progress_callback(f"PROGRESS:{progress_pct}")
            progress_callback(msg)
        return val

    if progress_callback:
        progress_callback("[INFO] Starting SLSQP optimization...")
    res = minimize(
        wrapped_obj, x0, method='SLSQP', bounds=bounds,
        options={'maxiter': total_iterations_estimate, 'disp': True}
    )
    if not res.success and progress_callback:
        progress_callback(f"[WARN] Optimization did not converge: {res.message}")
    return res.x


# --- New functions to generate graphs ---

def generate_surface_graph(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, cmap="jet") -> str:
    """
    Generates a 3D surface plot where each grid point's PPFD is its height above z=0.
    We 'shift' the entire dataset so that the global min PPFD becomes z=0.
    
    X, Y, Z must be the same shape (rows x cols).
    Returns a base64-encoded PNG string.
    """
    # Shift so the minimum PPFD is at z=0
    z_min = np.min(Z)
    Z_shifted = Z - z_min
    # Ensure no negative values after shifting
    Z_shifted[Z_shifted < 0] = 0

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # Create the surface. Each point in X, Y has a height Z_shifted[r,c].
    surf = ax.plot_surface(X, Y, Z_shifted, cmap=cmap, edgecolor='none')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("PPFD (µmol/m²/s)")
    ax.set_title("Light Intensity Surface Graph")

    # The highest point is now (max(Z) - min(Z)) above zero.
    ax.set_zlim(0, np.max(Z_shifted))

    # Add color bar
    fig.colorbar(surf, fraction=0.032, pad=0.04)

    # Save figure to in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_heatmap(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, cmap="jet") -> str:
    """
    2D heatmap with edges pinned to zero.
    Returns base64-encoded PNG string.
    """
    Z_copy = Z.copy()
    Z_copy[0, :] = 0
    Z_copy[-1, :] = 0
    Z_copy[:, 0] = 0
    Z_copy[:, -1] = 0

    fig, ax = plt.subplots(figsize=(8,6))
    extent = [X.min(), X.max(), Y.min(), Y.max()]

    # Calculate a suitable colorbar maximum (20% above the max intensity)
    colorbar_max = Z.max() * 1.2

    # Use vmin and vmax to control the color scaling
    im = ax.imshow(Z_copy, cmap=cmap, extent=extent, origin="lower", aspect="auto", vmin=0, vmax=colorbar_max)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Light Intensity Heatmap")

    # Create the colorbar and set its limits and ticks
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks(np.linspace(0, colorbar_max, num=6))  # 6 ticks
    cbar.set_ticklabels([f"{int(x)}" for x in np.linspace(0, colorbar_max, num=6)]) # Integer labels


    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# --- Updated run_ml_simulation to include graph images ---
def run_ml_simulation(floor_width_ft, floor_length_ft, target_ppfd, progress_callback=None):
    ft2m = 3.28084
    W_m = floor_width_ft / ft2m
    L_m = floor_length_ft / ft2m
    H_m = 3.0 / ft2m  # Constant LED height
    best_params = optimize_lighting(W_m, L_m, H_m, target_ppfd, progress_callback=progress_callback)
    final_ppfd = simulate_lighting(best_params, W_m, L_m, H_m)
    mean_ppfd = np.mean(final_ppfd)
    mad = np.mean(np.abs(final_ppfd - mean_ppfd))
    
    # Generate surface graph and heatmap based on the floor grid.
    X, Y = build_floor_grid(W_m, L_m)  # same shape as final_ppfd
    surface_graph_b64 = generate_surface_graph(X, Y, final_ppfd, cmap="jet")
    heatmap_b64 = generate_heatmap(X, Y, final_ppfd, cmap="jet")
    
    if progress_callback:
        progress_callback("PROGRESS:100")
        progress_callback("[INFO] Simulation complete!")

    return {
        "optimized_lumens_by_layer": best_params.tolist(),
        "mad": float(mad),
        "optimized_ppfd": float(mean_ppfd),
        "floor_width": floor_width_ft,
        "floor_length": floor_length_ft,
        "target_ppfd": target_ppfd,
        "floor_height": 3.0,
        "surface_graph": surface_graph_b64,
        "heatmap": heatmap_b64
    }


# ------------------------------
# Standalone Execution
# ------------------------------
def run_simulation(floor_width_ft=14.0, floor_length_ft=14.0, target_ppfd=1250.0):
    result = run_ml_simulation(floor_width_ft, floor_length_ft, target_ppfd)
    print("\n[RESULT] Simulation Output:")
    print(result)

def main():
    run_simulation()

if __name__ == "__main__":
    main()

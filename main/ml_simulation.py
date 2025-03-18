#!/usr/bin/env python3
"""
Lighting Optimization Script (ml_simulation.py)
Now supports dynamic COB positioning and layer‐specific luminous flux parameters.
Example usage: run_simulation()
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
from scipy.interpolate import griddata, RegularGridInterpolator
import threading
import time
import os

# ========================================
# Additional Imports for Surrogate Methods
# ========================================
from scipy.interpolate import CubicSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# Define a fixed candidate vector length for surrogate operations.
# (Adjust this constant to match your simulation candidate vector dimension.)
FIXED_NUM_LAYERS = 10  # Example value; you may set this dynamically if needed.

# ------------------------------
# Configuration & Constants
# ------------------------------
REFL_WALL = 0.8
REFL_CEIL = 0.8
REFL_FLOOR = 0.1

LUMINOUS_EFFICACY = 182.0         # lm/W
SPD_FILE = "./backups/spd_data.csv"

NUM_RADIOSITY_BOUNCES = 5
WALL_SUBDIVS_X = 10
WALL_SUBDIVS_Y = 5
CEIL_SUBDIVS_X = 10
CEIL_SUBDIVS_Y = 10

# Floor grid resolution (meters)
FLOOR_GRID_RES = 0.08

MIN_LUMENS = 1000.0
MAX_LUMENS_MAIN = 24000.0

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
# Prepare Geometry Once
# ------------------------------
def prepare_geometry(W, L, H):
    """
    Build and cache geometry used in the simulation:
    - COB positions
    - Floor grid (X,Y)
    - Patches (centers, areas, normals, reflectances)
    Returns a tuple (cob_positions, X, Y, (patch_centers, patch_areas, patch_normals, patch_refl)).
    """
    cob_positions = build_cob_positions(W, L, H)
    X, Y = build_floor_grid(W, L)  # lru_cached but we also keep references
    patches = build_patches(W, L, H)
    
    return (cob_positions, X, Y, patches)

# ------------------------------
# Dynamic COB Positioning (Staggered)
# ------------------------------
def build_cob_positions(W, L, H):
    ft2m = 3.28084
    floor_width_ft = W * ft2m
    floor_length_ft = L * ft2m
    max_dim_ft = max(floor_width_ft, floor_length_ft)
    n = max(1, int(max_dim_ft / 2) - 1)
    
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
    scale_x = (W / 2 * 0.95 * math.sqrt(2)) / n
    scale_y = (L / 2 * 0.95 * math.sqrt(2)) / n
    
    transformed = []
    for (x, y, h, layer) in positions:
        rx = x * cos_t - y * sin_t
        ry = x * sin_t + y * cos_t
        px = centerX + rx * scale_x
        py = centerY + ry * scale_y
        transformed.append((px, py, H * 0.95, layer))  # 95% of ceiling height
    
    return np.array(transformed, dtype=np.float64)

def pack_luminous_flux_dynamic(params, cob_positions):
    led_intensities = []
    for pos in cob_positions:
        layer = int(pos[3])
        # Directly use the corresponding parameter.
        intensity = params[layer]
        led_intensities.append(intensity)
    return np.array(led_intensities, dtype=np.float64)

# ------------------------------
# Floor Grid
# ------------------------------
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

# ------------------------------
# Direct Irradiance on Floor
# ------------------------------
@njit
def compute_direct_floor(light_positions, light_fluxes, X, Y):
    # Set a minimum squared distance to avoid singularities.
    # Here we use half the floor grid resolution as a minimum distance.
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
                    d2 = min_dist2  # enforce a minimum distance squared
                dist = math.sqrt(d2)
                cos_th = -dz / dist
                if cos_th < 0:
                    cos_th = 0.0
                val += (light_fluxes[k] / math.pi) * (cos_th / d2)
            out[r, c] = val
    return out


# ------------------------------
# Patches (Ceiling / Walls / Floor)
# ------------------------------
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

def compute_patch_direct(light_positions, light_fluxes, patch_centers, patch_normals, patch_areas):
    # Similarly, enforce a minimum distance squared.
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
        patch_rad = patch_direct + new_flux / patch_areas
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
# Main Simulation Using Prebuilt Geometry
# ------------------------------
def simulate_lighting(params, geo):
    """
    Uses pre-built geometry (cob_positions, X, Y, patches).
    Returns floor PPFD array.
    """
    cob_positions, X, Y, (p_centers, p_areas, p_normals, p_refl) = geo

    # Convert lumens to "power-like" intensities
    led_intensities = pack_luminous_flux_dynamic(params, cob_positions)
    power_arr = led_intensities / LUMINOUS_EFFICACY

    # Direct floor irradiance
    direct_irr = compute_direct_floor(cob_positions, power_arr, X, Y)
    # Patch direct + radiosity
    patch_direct = compute_patch_direct(cob_positions, power_arr, p_centers, p_normals, p_areas)
    patch_rad = iterative_radiosity_loop(p_centers, p_normals, patch_direct, p_areas, p_refl, NUM_RADIOSITY_BOUNCES)
    # Reflections onto the floor
    reflect_irr = compute_reflection_on_floor(X, Y, p_centers, p_normals, p_areas, patch_rad, p_refl)

    # Convert irradiance to PPFD
    return (direct_irr + reflect_irr) * CONVERSION_FACTOR

# ------------------------------
# Objective & Optimization
# ------------------------------
def objective_function(params, geo, target_ppfd):
    floor_ppfd = simulate_lighting(params, geo)
    mean_ppfd = np.mean(floor_ppfd)
    mad = np.mean(np.abs(floor_ppfd - mean_ppfd))
    ppfd_penalty = (mean_ppfd - target_ppfd) ** 2
    return mad + 2.0 * ppfd_penalty

def optimize_lighting(W, L, H, target_ppfd, progress_callback=None, x0=None):
    geo = prepare_geometry(W, L, H)
    # Force candidate dimension to FIXED_NUM_LAYERS
    if x0 is None:
        x0 = np.array([8000.0] * FIXED_NUM_LAYERS, dtype=np.float64)
    bounds = [(MIN_LUMENS, MAX_LUMENS_MAIN)] * FIXED_NUM_LAYERS

    total_iterations_estimate = 500
    iteration = 0

    def wrapped_obj(p):
        nonlocal iteration
        iteration += 1
        val = objective_function(p, geo, target_ppfd)
        if progress_callback:
            # Determine dynamic layer count based on generated COB positions
            dynamic_layer_count = int(np.max(geo[0][:, 3])) + 1
            progress_pct = min(98, (iteration / total_iterations_estimate) * 100)
            progress_callback(f"PROGRESS:{progress_pct}")
            progress_callback(f"[DEBUG] param={p[:dynamic_layer_count]}, obj={val:.3f}")
        return val

    if progress_callback:
        progress_callback("[INFO] Starting SLSQP optimization...")

    from scipy.optimize import minimize
    res = minimize(
        wrapped_obj, x0, method='SLSQP', bounds=bounds,
        options={'maxiter': total_iterations_estimate, 'disp': True}
    )
    if progress_callback and not res.success:
        progress_callback(f"[WARN] Optimization did not converge: {res.message}")

    if progress_callback:
        progress_callback("PROGRESS:99")
    return res.x, geo


# ------------------------------
# Grid Simulation (Uniform, no optimization)
# ------------------------------
def build_grid_cob_positions(W, L, rows, cols, H):
    # Calculate cell dimensions
    dx = W / cols
    dy = L / rows
    # Add an offset equal to half the floor grid resolution to avoid alignment issues
    offset_x = FLOOR_GRID_RES / 2.0
    offset_y = FLOOR_GRID_RES / 2.0
    xs = (np.arange(cols) + 0.5) * dx + offset_x
    ys = (np.arange(rows) + 0.5) * dy + offset_y
    # Clip to ensure COB positions remain within the room dimensions
    xs = np.clip(xs, 0, W)
    ys = np.clip(ys, 0, L)
    Xg, Yg = np.meshgrid(xs, ys)
    positions = []
    for i in range(rows):
        for j in range(cols):
            positions.append((Xg[i, j], Yg[i, j], H, 0))
    return np.array(positions, dtype=np.float64)


def simulate_lighting_grid(uniform_flux, W, L, H, cob_positions):
    power_arr = np.full(len(cob_positions), uniform_flux) / LUMINOUS_EFFICACY
    X, Y = build_floor_grid(W, L)
    direct_irr = compute_direct_floor(cob_positions, power_arr, X, Y)
    p_centers, p_areas, p_normals, p_refl = build_patches(W, L, H)
    patch_direct = compute_patch_direct(cob_positions, power_arr, p_centers, p_normals, p_areas)
    patch_rad = iterative_radiosity_loop(
        p_centers, p_normals, patch_direct, p_areas, p_refl, NUM_RADIOSITY_BOUNCES
    )
    reflect_irr = compute_reflection_on_floor(
        X, Y, p_centers, p_normals, p_areas, patch_rad, p_refl
    )
    floor_irr = direct_irr + reflect_irr
    floor_ppfd = floor_irr * CONVERSION_FACTOR
    return floor_ppfd

# ------------------------------
# Visualization
# ------------------------------
def generate_surface_graph(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, cmap="jet") -> str:
    x_ = X.flatten()
    y_ = Y.flatten()
    z_ = Z.flatten()

    triang = mtri.Triangulation(x_, y_)

    step = 0.15
    x_grid, y_grid = np.mgrid[x_.min():x_.max():step, y_.min():y_.max():step]
    interp_lin = mtri.LinearTriInterpolator(triang, z_)
    z_grid = interp_lin(x_grid, y_grid)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x_grid, y_grid, np.nan_to_num(z_grid),
                           cmap=cmap, vmin=z_.min(), vmax=z_.max(),
                           edgecolor='none', antialiased=True)

    ax.plot_wireframe(x_grid[::50, ::50],
                      y_grid[::50, ::50],
                      np.nan_to_num(z_grid)[::50, ::50],
                      color='k', linewidth=0.4)

    ax.contourf(x_grid, y_grid, np.nan_to_num(z_grid),
                zdir='z', offset=0, cmap=cmap, alpha=0.7)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("PPFD (µmol/m²/s)")
    ax.set_title("Light Intensity Surface Graph")
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(0, z_.max())

    fig.colorbar(surf, fraction=0.032, pad=0.04)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_heatmap(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, cmap="jet", 
                     overlay_intensity: bool=False, overlay_step: int=10) -> str:
    theta = math.radians(45)
    xs = X[0, :]
    ys = Y[:, 0]
    centerX = xs.mean()
    centerY = ys.mean()

    X_rot = centerX + (X - centerX) * math.cos(theta) - (Y - centerY) * math.sin(theta)
    Y_rot = centerY + (X - centerX) * math.sin(theta) + (Y - centerY) * math.cos(theta)

    xs_rot = X_rot[0, :]
    ys_rot = Y_rot[:, 0]

    factor = 10
    x_hr = np.linspace(xs_rot.min(), xs_rot.max(), len(xs_rot) * factor)
    y_hr = np.linspace(ys_rot.min(), ys_rot.max(), len(ys_rot) * factor)
    X_hr, Y_hr = np.meshgrid(x_hr, y_hr)

    interpolator = RegularGridInterpolator((ys_rot, xs_rot), Z, method="linear")
    pts_hr = np.column_stack((Y_hr.flatten(), X_hr.flatten()))
    Z_hr = interpolator(pts_hr).reshape(Y_hr.shape)

    # Ensure corners match
    Z_hr[0, 0]       = Z[0, 0]
    Z_hr[0, -1]      = Z[0, -1]
    Z_hr[-1, 0]      = Z[-1, 0]
    Z_hr[-1, -1]     = Z[-1, -1]
    center_hr_i = Z_hr.shape[0] // 2
    center_hr_j = Z_hr.shape[1] // 2
    center_orig_i = len(ys_rot) // 2
    center_orig_j = len(xs_rot) // 2
    Z_hr[center_hr_i, center_hr_j] = Z[center_orig_i, center_orig_j]

    extent = [xs_rot.min(), xs_rot.max(), ys_rot.min(), ys_rot.max()]

    fig, ax = plt.subplots(figsize=(8, 6))
    colorbar_max = Z_hr.max() * 1.2
    im = ax.imshow(Z_hr, cmap=cmap, origin="lower", extent=extent, vmin=0, vmax=colorbar_max, aspect="auto")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Heatmap")
    cbar = fig.colorbar(im, ax=ax)
    ticks = np.linspace(0, colorbar_max, 6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(round(t))}" for t in ticks])

    if overlay_intensity:
        for i in range(0, len(ys_rot), overlay_step):
            for j in range(0, len(xs_rot), overlay_step):
                ax.text(xs_rot[j], ys_rot[i], f"{int(round(Z[i, j]))}",
                        color="white", ha="center", va="center",
                        bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1))

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ========================================
# Surrogate & Persistence Functions Begin
# ========================================

def adjust_candidate_dimension(candidate, fixed_dim=FIXED_NUM_LAYERS, default_value=8000.0):
    candidate = np.asarray(candidate, dtype=np.float64)
    current_dim = candidate.shape[0]
    if current_dim < fixed_dim:
        pad = np.full((fixed_dim - current_dim,), default_value)
        return np.concatenate([candidate, pad])
    elif current_dim > fixed_dim:
        return candidate[:fixed_dim]
    return candidate

def forward_transform(x):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    T = np.sum(x)
    if T < 1e-12:
        T = 1e-12
    if n == 1:
        u = np.array([0.0])
        y = x / T
    else:
        u = np.linspace(0.0, 1.0, n)
        y = x / T
    return T, u, y

def inverse_transform(T, u, y, m):
    if m == 1:
        return np.array([T], dtype=np.float64)
    spline = CubicSpline(u, y, bc_type='natural')
    v = np.linspace(0.0, 1.0, m)
    y_new = spline(v)
    x_new = T * y_new
    return x_new

def load_training_data():
    TRAINING_DATA_FILE = "./backups/training_data.npz"
    if os.path.exists(TRAINING_DATA_FILE):
        with np.load(TRAINING_DATA_FILE, allow_pickle=True) as data:
            X_train = data["X_train"]
            y_train = data["y_train"]
            G_train = data["G_train"] if "G_train" in data else None
        print(f"[INFO] Loaded {len(X_train)} training samples (feature dimension = {X_train.shape[1]}).")
        return X_train, y_train, G_train
    else:
        print("[INFO] No persistent training data found. Starting fresh.")
        return None, None, None

def save_training_data(X_train, y_train, G_train):
    TRAINING_DATA_FILE = "training_data.npz"
    np.savez(TRAINING_DATA_FILE, X_train=X_train, y_train=y_train, G_train=G_train)
    print(f"[INFO] Saved {len(X_train)} training samples.")

def update_training_data(geo, target_ppfd, new_sample, geom_features):
    M = 10  # number of spline samples; feature dimension = M+1
    raw_candidate = adjust_candidate_dimension(new_sample[0], fixed_dim=FIXED_NUM_LAYERS)
    print(f"[DEBUG] Received candidate of shape {np.asarray(new_sample[0]).shape}, adjusted to {raw_candidate.shape}")
    obj_value = new_sample[1]
    T, u, y = forward_transform(raw_candidate)
    spline = CubicSpline(u, y, bc_type='natural')
    v = np.linspace(0.0, 1.0, M)
    y_fixed = spline(v)
    x_norm = np.hstack((T, y_fixed))
    X_old, y_old, G_old = load_training_data()
    if X_old is not None:
        if X_old.shape[1] != x_norm.shape[0]:
            print("[WARN] Dimension mismatch in training data. Discarding old data.")
            X_train = np.array([x_norm], dtype=np.float64)
            y_train = np.array([obj_value], dtype=np.float64)
            G_train = np.array([geom_features], dtype=np.float64)
        else:
            X_train = np.vstack((X_old, x_norm))
            y_train = np.hstack((y_old, [obj_value]))
            G_train = np.vstack((G_old, geom_features))
    else:
        X_train = np.array([x_norm], dtype=np.float64)
        y_train = np.array([obj_value], dtype=np.float64)
        G_train = np.array([geom_features], dtype=np.float64)
    
    save_training_data(X_train, y_train, G_train)
    print(f"[INFO] Saved training sample #{len(X_train)} with normalized dimension {x_norm.shape[0]}")
    return X_train, y_train, G_train

def train_surrogate(X_train, y_train):
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
    gp.fit(X_train, y_train)
    print("[INFO] Surrogate trained. Final kernel:", gp.kernel_)
    return gp

def surrogate_objective(params, gp_model):
    params = np.atleast_2d(params)
    y_pred, sigma = gp_model.predict(params, return_std=True)
    return y_pred[0]

def optimize_with_surrogate(geo, target_ppfd, gp_model, x0=None):
    M = gp_model.X_train_.shape[1] - 1
    if x0 is None:
        x0 = np.full(FIXED_NUM_LAYERS, 8000.0, dtype=np.float64)
    x0 = adjust_candidate_dimension(x0, fixed_dim=FIXED_NUM_LAYERS)
    
    T0, u0, y0 = forward_transform(x0)
    spline0 = CubicSpline(u0, y0, bc_type='natural')
    v0 = np.linspace(0.0, 1.0, M)
    y0_fixed = spline0(v0)
    z0 = np.hstack((T0, y0_fixed))

    def objective_in_normalized_space(z):
        z_reshaped = z[np.newaxis, :]
        return gp_model.predict(z_reshaped)[0]

    bounds = [(1000.0, 30000.0)] + [(0.0, 1.0)] * M
    from scipy.optimize import minimize
    res = minimize(objective_in_normalized_space, z0, method='SLSQP', bounds=bounds, options={'maxiter':200})
    z_best = res.x
    T_best = z_best[0]
    shape_best = z_best[1:]
    x_new = inverse_transform(T_best, np.linspace(0, 1, M), shape_best, FIXED_NUM_LAYERS)
    print(f"[INFO] Surrogate optimization complete. Best objective= {res.fun:.3f}, success={res.success}")
    return x_new

def filter_training_data_by_layers(X_train, y_train, expected_feature_dim):
    filtered_X = []
    filtered_y = []
    for i, x in enumerate(X_train):
        if len(x) == expected_feature_dim:
            filtered_X.append(x)
            filtered_y.append(y_train[i])
    if filtered_X:
        return np.vstack(filtered_X), np.array(filtered_y)
    else:
        return None, None

def train_regression_model_for_normalized(M=10):
    X_train, y_train, G_train = load_training_data()
    if X_train is None or len(X_train) < 5:
        print("[INFO] Not enough training data for regression model.")
        return None
    reg = MultiOutputRegressor(RandomForestRegressor(n_estimators=30, random_state=42))
    reg.fit(G_train, X_train)
    print(f"[INFO] Regression model trained on {len(X_train)} samples.")
    return reg

def predict_initial_guess_regression(geom_features, n_layers):
    M = 10
    model = train_regression_model_for_normalized(M=M)
    if model is None:
        print("[WARN] No regression model available; using fallback guess.")
        return np.full(FIXED_NUM_LAYERS, 8000.0, dtype=np.float64)
    z_pred = model.predict([geom_features])[0]
    T_pred = z_pred[0]
    shape_pred = z_pred[1:]
    x_guess = inverse_transform(T_pred, np.linspace(0, 1, M), shape_pred, FIXED_NUM_LAYERS)
    print(f"[INFO] Regression-based guess computed: T={T_pred:.1f} for {FIXED_NUM_LAYERS} layers.")
    return x_guess

def generate_new_sample(geo, target_ppfd, sample_params):
    obj_value = objective_function(sample_params, geo, target_ppfd)
    return sample_params, obj_value

# ========================================
# Surrogate & Persistence Functions End
# ========================================

# ------------------------------
# Main Entry Point
# ------------------------------
# ================================================================
# Modified run_ml_simulation Using Surrogate-Based Initial Guess
# ================================================================
def run_ml_simulation(floor_width_ft, floor_length_ft, target_ppfd, floor_height=3.0, progress_callback=None, side_by_side=False):
    if progress_callback:
        progress_callback("[INFO] Simulation started.")
        progress_callback("PROGRESS:5")
        
    try:
        floor_width_ft = float(floor_width_ft)
        floor_length_ft = float(floor_length_ft)
        target_ppfd = float(target_ppfd)
        floor_height = float(floor_height)
    except Exception as e:
        print("Error converting inputs to float:", e)
        return {}
    
    ft2m = 3.28084
    W_m = floor_width_ft / ft2m
    L_m = floor_length_ft / ft2m
    H_m = floor_height / ft2m

    if progress_callback:
        progress_callback("[INFO] Preparing geometry...")
        progress_callback("PROGRESS:10")
    
    # Prepare geometry.
    geo = prepare_geometry(W_m, L_m, H_m)
    # Determine the dynamic layer count from the COB positions
    dynamic_layer_count = int(np.max(geo[0][:, 3])) + 1
    
    if progress_callback:
        progress_callback("[INFO] Loading training data...")
        progress_callback("PROGRESS:15")
    
    # Try loading precomputed training data.
    X_train, y_train, G_train = load_training_data()
    
    # Current input features for matching: [floor_width_ft, floor_length_ft, target_ppfd, floor_height]
    current_geom_features = np.array([floor_width_ft, floor_length_ft, target_ppfd, floor_height])
    
    # If training data exists, check for a near instantaneous candidate.
    if X_train is not None:
        # Define thresholds for a "close enough" match.
        thresholds = np.array([0.5, 0.5, 100.0, 0.5])  # Adjust thresholds as needed.
        differences = np.abs(G_train - current_geom_features)
        matching_indices = np.where((differences <= thresholds).all(axis=1))[0]
        if len(matching_indices) > 0:
            index = matching_indices[0]
            candidate_norm = X_train[index]
            M = 10  # number of spline samples used in training data
            best_params_full = inverse_transform(candidate_norm[0], np.linspace(0, 1, M), candidate_norm[1:], FIXED_NUM_LAYERS)
            if progress_callback:
                progress_callback("[INFO] Found matching training data sample. Using cached candidate.")
        else:
            if progress_callback:
                progress_callback("[INFO] No matching training sample found. Running surrogate optimization...")
                progress_callback("[INFO] Training surrogate model...")
                progress_callback("PROGRESS:20")
            gp_model = train_surrogate(X_train, y_train)
            
            if progress_callback:
                progress_callback("[INFO] Predicting initial guess via regression...")
                progress_callback("PROGRESS:30")
            informed_candidate = predict_initial_guess_regression(current_geom_features.tolist(), FIXED_NUM_LAYERS)
            
            if progress_callback:
                progress_callback("[INFO] Refining candidate with surrogate optimization...")
                progress_callback("PROGRESS:40")
            candidate = optimize_with_surrogate(geo, target_ppfd, gp_model, x0=informed_candidate)
            print(f"[INFO] Surrogate candidate: {candidate}")
            if progress_callback:
                progress_callback(f"[INFO] Using surrogate candidate as initial guess: {candidate}")
            
            if progress_callback:
                progress_callback("[INFO] Running full optimization...")
                progress_callback("PROGRESS:50")
            best_params_full, _ = optimize_lighting(W_m, L_m, H_m, target_ppfd, progress_callback=progress_callback, x0=candidate)
            
            if progress_callback:
                progress_callback("[INFO] Updating training data with new sample...")
                progress_callback("PROGRESS:70")
            new_sample = generate_new_sample(geo, target_ppfd, best_params_full)
            update_training_data(geo, target_ppfd, new_sample, current_geom_features.tolist())
    else:
        if progress_callback:
            progress_callback("[INFO] No training data found, running full optimization...")
            progress_callback("PROGRESS:20")
        best_params_full, _ = optimize_lighting(W_m, L_m, H_m, target_ppfd, progress_callback=progress_callback)
    
    if progress_callback:
        progress_callback("[INFO] Computing final PPFD and metrics...")
        progress_callback("PROGRESS:80")
    # Compute final PPFD using the optimized candidate.
    final_ppfd = simulate_lighting(best_params_full, geo)
    mean_ppfd = np.mean(final_ppfd)
    mad = np.mean(np.abs(final_ppfd - mean_ppfd))
    rmse = np.sqrt(np.mean((final_ppfd - mean_ppfd)**2))
    if mean_ppfd:
        dou = 100 * (1 - rmse / mean_ppfd)
        cv = 100 * (np.std(final_ppfd) / mean_ppfd)
    else:
        dou = 0
        cv = 0

    if progress_callback:
        progress_callback("[INFO] Generating visualizations...")
        progress_callback("PROGRESS:90")
    X, Y = geo[1], geo[2]
    surface_graph_b64 = generate_surface_graph(X, Y, final_ppfd, cmap="jet")
    heatmap_b64 = generate_heatmap(X, Y, final_ppfd, cmap="jet", overlay_intensity=False)

    result = {
        "optimized_lumens_by_layer": best_params_full[:dynamic_layer_count].tolist(),
        "mad": float(mad),
        "optimized_ppfd": float(mean_ppfd),
        "rmse": float(rmse),
        "dou": float(dou),
        "cv": float(cv),
        "floor_width": floor_width_ft,
        "floor_length": floor_length_ft,
        "target_ppfd": target_ppfd,
        "floor_height": floor_height,
        "surface_graph": surface_graph_b64,
        "heatmap": heatmap_b64,
        "heatmapGrid": final_ppfd.tolist()
    }

    if side_by_side:
        if progress_callback:
            progress_callback("[INFO] Running side-by-side (grid) simulation...")
            progress_callback("PROGRESS:95")
        grid_mapping = {
            (2,2): (3,3),
            (4,4): (4,4),
            (6,6): (5,5),
            (8,8): (6,6),
            (10,10): (7,7),
            (12,12): (8,8),
            (14,14): (9,9),
            (16,16): (10,10),
            (12,16): (8,10)
        }
        key = (int(floor_width_ft), int(floor_length_ft))
        grid_dims = grid_mapping.get(key, (8,8))
        rows, cols = grid_dims
        grid_cob_positions = build_grid_cob_positions(W_m, L_m, rows, cols, H_m)
        ppfd_test = simulate_lighting_grid(1.0, W_m, L_m, H_m, grid_cob_positions)
        mean_ppfd_test = np.mean(ppfd_test)
        if mean_ppfd_test == 0:
            required_flux = 0
        else:
            required_flux = target_ppfd / mean_ppfd_test
        grid_final_ppfd = simulate_lighting_grid(required_flux, W_m, L_m, H_m, grid_cob_positions)
        grid_mean_ppfd = np.mean(grid_final_ppfd)
        grid_mad = np.mean(np.abs(grid_final_ppfd - grid_mean_ppfd))
        grid_rmse = np.sqrt(np.mean((grid_final_ppfd - grid_mean_ppfd)**2))
        if grid_mean_ppfd:
            grid_dou = 100 * (1 - grid_rmse / grid_mean_ppfd)
            grid_cv = 100 * (np.std(grid_final_ppfd) / grid_mean_ppfd)
        else:
            grid_dou = 0
            grid_cv = 0
        grid_surface_graph_b64 = generate_surface_graph(X, Y, grid_final_ppfd, cmap="jet")
        grid_heatmap_b64 = generate_heatmap(X, Y, grid_final_ppfd, cmap="jet", overlay_intensity=False)
        result.update({
            "grid_cob_arrangement": {"rows": rows, "cols": cols},
            "grid_uniform_flux": required_flux,
            "grid_ppfd": float(grid_mean_ppfd),
            "grid_mad": float(grid_mad),
            "grid_rmse": float(grid_rmse),
            "grid_dou": float(grid_dou),
            "grid_cv": float(grid_cv),
            "grid_surface_graph": grid_surface_graph_b64,
            "grid_heatmap": grid_heatmap_b64
        })

    if progress_callback:
        progress_callback("PROGRESS:100")
    
    return result



def run_simulation(floor_width_ft=14.0, floor_length_ft=14.0, target_ppfd=1250.0, floor_height=3.0):
    result = run_ml_simulation(floor_width_ft, floor_length_ft, target_ppfd, floor_height, side_by_side=True)
    print("\n[RESULT] Simulation Output:")
    print(result)
    return result  # Return the result so that the API response is defined

def main():
    run_simulation()

if __name__ == "__main__":
    main()

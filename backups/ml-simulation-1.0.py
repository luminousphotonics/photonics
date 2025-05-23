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
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D  # just for side-effects
from scipy.interpolate import griddata, RegularGridInterpolator
import threading
import time


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
FLOOR_GRID_RES = 0.01

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
            cos_th_led = abs(dz)/dist
            E_led = (light_fluxes[j]/math.pi)*(cos_th_led/dist2)
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

def optimize_lighting(W, L, H, target_ppfd, progress_callback=None):
    """
    1. Prepare geometry once.
    2. Build an objective that uses that geometry.
    3. Minimize with SLSQP.
    """
    geo = prepare_geometry(W, L, H)
    cob_positions = geo[0]
    n = int(np.max(cob_positions[:, 3]))
    x0 = np.array([8000.0] * (n + 1), dtype=np.float64)
    bounds = [(MIN_LUMENS, MAX_LUMENS_MAIN)] * (n + 1)

    total_iterations_estimate = 500
    iteration = 0

    def wrapped_obj(p):
        nonlocal iteration
        iteration += 1
        val = objective_function(p, geo, target_ppfd)
        # For debug logging
        floor_ppfd = simulate_lighting(p, geo)
        mp = np.mean(floor_ppfd)
        msg = f"[DEBUG] param={p}, mean_ppfd={mp:.1f}, obj={val:.3f}"
        if progress_callback:
            progress_pct = min(98, (iteration / total_iterations_estimate) * 100)
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

    if progress_callback:
        progress_callback("PROGRESS:99")

    return res.x, geo

# ------------------------------
# Grid Simulation (Uniform, no optimization)
# ------------------------------
def build_grid_cob_positions(W, L, rows, cols, H):
    xs = np.linspace(0, W, cols)
    ys = np.linspace(0, L, rows)
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
def display_surface_graph(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, cmap="jet") -> None:
    """Display a 3D surface graph."""
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
    plt.show()  # display interactively

def display_heatmap(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, cmap="jet", overlay_intensity: bool=False, overlay_step: int=10) -> None:
    """Display a heatmap."""
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
    Z_hr[0, 0]   = Z[0, 0]
    Z_hr[0, -1]  = Z[0, -1]
    Z_hr[-1, 0]  = Z[-1, 0]
    Z_hr[-1, -1] = Z[-1, -1]
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
    plt.show()  # display interactively

# ------------------------------
# Main Entry Point
# ------------------------------
def run_ml_simulation(floor_width_ft, floor_length_ft, target_ppfd, floor_height=3.0, progress_callback=None, side_by_side=False):
    try:
        floor_width_ft = float(floor_width_ft)
        floor_length_ft = float(floor_length_ft)
        target_ppfd = float(target_ppfd)
        floor_height = float(floor_height)  # New: parse light height
    except Exception as e:
        print("Error converting floor dimensions, target PPFD, or light height to float:", e)
        return {}

    ft2m = 3.28084
    W_m = floor_width_ft / ft2m
    L_m = floor_length_ft / ft2m
    H_m = floor_height / ft2m  # Use the input light height

    # Start heartbeat thread if progress_callback is provided.
    heartbeat_active = [True]  # mutable flag to control the heartbeat thread
    if progress_callback:
        def heartbeat():
            while heartbeat_active[0]:
                # Send a heartbeat message. This helps keep the client connection active,
                # but you can consider disabling this entirely if you prefer relying solely on Redis updates.
                progress_callback(": heartbeat")
                time.sleep(15)
        hb_thread = threading.Thread(target=heartbeat)
        hb_thread.daemon = True
        hb_thread.start()

    # Optimize lighting using the new geometry with variable light height.
    best_params, geo = optimize_lighting(W_m, L_m, H_m, target_ppfd, progress_callback=progress_callback)
    final_ppfd = simulate_lighting(best_params, geo)
    mean_ppfd = np.mean(final_ppfd)
    mad = np.mean(np.abs(final_ppfd - mean_ppfd))
    rmse = np.sqrt(np.mean((final_ppfd - mean_ppfd)**2))
    dou = 100 * (1 - rmse / mean_ppfd)
    cv = 100 * (np.std(final_ppfd) / mean_ppfd)

    X, Y = geo[1], geo[2]
    surface_graph_b64 = display_surface_graph(X, Y, final_ppfd, cmap="jet")
    heatmap_b64 = display_heatmap(X, Y, final_ppfd, cmap="jet", overlay_intensity=False)

    result = {
        "optimized_lumens_by_layer": best_params.tolist(),
        "mad": float(mad),
        "optimized_ppfd": float(mean_ppfd),
        "rmse": float(rmse),
        "dou": float(dou),
        "cv": float(cv),
        "floor_width": floor_width_ft,
        "floor_length": floor_length_ft,
        "target_ppfd": target_ppfd,
        "floor_height": floor_height,  # Return the input light height
        "surface_graph": surface_graph_b64,
        "heatmap": heatmap_b64,
        "heatmapGrid": final_ppfd.tolist()
    }
    
    # Optional side-by-side uniform grid approach
    if side_by_side:
        # In run_ml_simulation, inside the side_by_side branch, update grid_mapping:
        grid_mapping = {
            (2, 2): (3, 3),
            (4, 4): (4, 4),
            (6, 6): (5, 5),
            (8, 8): (6, 6),
            (10, 10): (7, 7),
            (12, 12): (8, 8),
            (14, 14): (9, 9),
            (16, 16): (10, 10),
            (20, 20): (12, 12),  # New mapping for 20' x 20' floor size
            (12, 16): (8, 10),
            (30, 30): (17, 17),
            (40, 40): (25, 25)
        }

        key = (int(floor_width_ft), int(floor_length_ft))
        grid_dims = grid_mapping.get(key, (8,8))
        rows, cols = grid_dims
        grid_cob_positions = build_grid_cob_positions(W_m, L_m, rows, cols, H_m)
        # 1) Check baseline with uniform flux=1.0
        ppfd_test = simulate_lighting_grid(1.0, W_m, L_m, H_m, grid_cob_positions)
        mean_ppfd_test = np.mean(ppfd_test)
        required_flux = target_ppfd / mean_ppfd_test
        # 2) Now set uniform_flux to required_flux
        grid_final_ppfd = simulate_lighting_grid(required_flux, W_m, L_m, H_m, grid_cob_positions)
        grid_mean_ppfd = np.mean(grid_final_ppfd)
        grid_mad = np.mean(np.abs(grid_final_ppfd - grid_mean_ppfd))
    
        grid_rmse = np.sqrt(np.mean((grid_final_ppfd - grid_mean_ppfd)**2))
        grid_dou = 100 * (1 - grid_rmse / grid_mean_ppfd)
        grid_cv = 100 * (np.std(grid_final_ppfd) / grid_mean_ppfd)
        
        grid_surface_graph_b64 = display_surface_graph(X, Y, grid_final_ppfd, cmap="jet")
        grid_heatmap_b64 = display_heatmap(X, Y, grid_final_ppfd, cmap="jet", overlay_intensity=False)
        
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
        # Signal the heartbeat thread to stop once the simulation is complete.
        heartbeat_active[0] = False
        # Send a final progress update to ensure the frontend reaches 100%.
        progress_callback("PROGRESS:100")
    
    return result

def run_uniform_grid_simulation(floor_width_ft=40.0, floor_length_ft=40.0, target_ppfd=1250.0, floor_height=3.0):
    """
    Runs the uniform grid simulation only (without dynamic COB arrangement).
    Computes the uniform flux required to achieve the target average PPFD,
    calculates metrics, and displays the surface graph and heatmap.
    """
    ft2m = 3.28084
    # Convert floor dimensions and light height to meters.
    W_m = floor_width_ft / ft2m
    L_m = floor_length_ft / ft2m
    H_m = floor_height / ft2m

    # Grid mapping dictionary (keys are floor dimensions in feet).
    grid_mapping = {
        (2, 2): (3, 3),
        (4, 4): (4, 4),
        (6, 6): (5, 5),
        (8, 8): (6, 6),
        (10, 10): (7, 7),
        (12, 12): (8, 8),
        (14, 14): (9, 9),
        (16, 16): (10, 10),
        (20, 20): (12, 12),
        (30, 30): (19, 19),
        (12, 16): (8, 10),
        (40, 40): (25, 25)
    }
    key = (int(floor_width_ft), int(floor_length_ft))
    grid_dims = grid_mapping.get(key, (8, 8))
    rows, cols = grid_dims

    # Build the uniform grid COB positions.
    grid_cob_positions = build_grid_cob_positions(W_m, L_m, rows, cols, H_m)

    # 1) Baseline simulation with uniform flux = 1.0.
    baseline_ppfd = simulate_lighting_grid(1.0, W_m, L_m, H_m, grid_cob_positions)
    mean_ppfd_baseline = np.mean(baseline_ppfd)
    # Compute the uniform flux needed to reach target_ppfd.
    required_flux = target_ppfd / mean_ppfd_baseline

    # 2) Run the simulation with the calculated uniform flux.
    grid_final_ppfd = simulate_lighting_grid(required_flux, W_m, L_m, H_m, grid_cob_positions)
    grid_mean_ppfd = np.mean(grid_final_ppfd)
    grid_mad = np.mean(np.abs(grid_final_ppfd - grid_mean_ppfd))
    grid_rmse = np.sqrt(np.mean((grid_final_ppfd - grid_mean_ppfd)**2))
    grid_dou = 100 * (1 - grid_rmse / grid_mean_ppfd)
    grid_cv = 100 * (np.std(grid_final_ppfd) / grid_mean_ppfd)

    # Print metrics for the uniform grid simulation.
    print("\n[RESULT] Uniform Grid Simulation Metrics:")
    print(f"Grid Arrangement (rows x cols): {rows} x {cols}")
    print(f"Uniform Flux: {required_flux:.3f}")
    print(f"Grid Average PPFD: {grid_mean_ppfd:.1f} µmol/m²/s")
    print(f"Grid MAD: {grid_mad:.3f}")
    print(f"Grid RMSE: {grid_rmse:.3f}")
    print(f"Grid Uniformity (DOU): {grid_dou:.1f}%")
    print(f"Grid CV: {grid_cv:.1f}%")

    # Build the floor grid for plotting.
    X, Y = build_floor_grid(W_m, L_m)
    # Display the graphs interactively.
    display_surface_graph(X, Y, grid_final_ppfd, cmap="jet")
    display_heatmap(X, Y, grid_final_ppfd, cmap="jet", overlay_intensity=False)

    return {
        "grid_arrangement": {"rows": rows, "cols": cols},
        "uniform_flux": required_flux,
        "grid_avg_ppfd": grid_mean_ppfd,
        "grid_mad": grid_mad,
        "grid_rmse": grid_rmse,
        "grid_dou": grid_dou,
        "grid_cv": grid_cv,
        "grid_ppfd_data": grid_final_ppfd.tolist()
    }


def verbose_progress_callback(msg):
    # This callback simply prints every progress message.
    # You can add additional filtering or formatting if needed.
    print(msg)

def run_simulation_verbose(floor_width_ft=12.0, floor_length_ft=12.0, target_ppfd=1250.0, floor_height=3.0):
    """
    Runs the dynamic simulation (optimization of COB luminous flux) with a verbose
    progress callback that prints out messages (including optimized_lumens_by_layer updates)
    in real time.
    """
    # Run the simulation with verbose progress output.
    result = run_ml_simulation(floor_width_ft, floor_length_ft, target_ppfd, floor_height,
                               progress_callback=verbose_progress_callback, side_by_side=True)
    
    # Print final results, including the optimized lumens by layer.
    print("\n[FINAL RESULT] Optimized Simulation Metrics:")
    print(f"Optimized Lumens by Layer: {result['optimized_lumens_by_layer']}")
    print(f"Average PPFD: {result['optimized_ppfd']:.1f} µmol/m²/s")
    print(f"MAD: {result['mad']:.3f}")
    print(f"RMSE: {result['rmse']:.3f}")
    print(f"Uniformity (DOU): {result['dou']:.1f}%")
    print(f"CV: {result['cv']:.1f}%")
    
    # Also print the grid simulation metrics if computed.
    if "grid_cob_arrangement" in result:
        print("\n[FINAL RESULT] Grid Simulation Metrics:")
        print(f"Grid Arrangement (rows x cols): {result['grid_cob_arrangement']['rows']} x {result['grid_cob_arrangement']['cols']}")
        print(f"Uniform Flux: {result['grid_uniform_flux']:.3f}")
        print(f"Grid Average PPFD: {result['grid_ppfd']:.1f} µmol/m²/s")
        print(f"Grid MAD: {result['grid_mad']:.3f}")
        print(f"Grid RMSE: {result['grid_rmse']:.3f}")
        print(f"Grid Uniformity (DOU): {result['grid_dou']:.1f}%")
        print(f"Grid CV: {result['grid_cv']:.1f}%")
    
    # Retrieve geometry for plotting.
    ft2m = 3.28084
    W_m = floor_width_ft / ft2m
    L_m = floor_length_ft / ft2m
    geo = prepare_geometry(W_m, L_m, floor_height/ft2m)
    final_ppfd = simulate_lighting(result['optimized_lumens_by_layer'], geo)
    X, Y = geo[1], geo[2]
    
    # Display the graphs interactively.
    display_surface_graph(X, Y, final_ppfd, cmap="jet")
    display_heatmap(X, Y, final_ppfd, cmap="jet", overlay_intensity=False)
    
    return result
def visualize_cob_placement(floor_width_ft, floor_length_ft, floor_height=3.0, grid_dims=None):
    """
    Generates a side-by-side visualization of COB placements:
      - Left: Dynamic COB arrangement (build_cob_positions)
      - Right: Uniform grid COB placement (build_grid_cob_positions)
    
    Both plots use a square aspect ratio. The dynamic plot's color legend is
    placed externally so it doesn't affect the axis dimensions.
    
    Parameters:
      floor_width_ft (float): Floor width in feet.
      floor_length_ft (float): Floor length in feet.
      floor_height (float): Light (ceiling) height in feet.
      grid_dims (tuple): Optional (rows, cols) for the uniform grid.
                         If not provided, defaults to (8, 8).
    """
    import matplotlib.pyplot as plt

    ft2m = 3.28084
    # Convert dimensions to meters.
    W_m = floor_width_ft / ft2m
    L_m = floor_length_ft / ft2m
    H_m = floor_height / ft2m

    # Build dynamic COB positions.
    dynamic_cobs = build_cob_positions(W_m, L_m, H_m)
    # Build uniform grid COB positions.
    if grid_dims is None:
        grid_dims = (8, 8)
    rows, cols = grid_dims
    uniform_cobs = build_grid_cob_positions(W_m, L_m, rows, cols, H_m)

    # Create a figure with two subplots side by side.
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot dynamic COB placement.
    scatter1 = axs[0].scatter(dynamic_cobs[:, 0], dynamic_cobs[:, 1],
                              c=dynamic_cobs[:, 3], cmap="viridis", s=50)
    axs[0].set_title("Dynamic COB Placement")
    axs[0].set_xlabel("X (m)")
    axs[0].set_ylabel("Y (m)")
    # Set equal aspect ratio.
    axs[0].set_aspect('equal', 'box')
    
    # Create an external colorbar so that it doesn't resize the subplot.
    cbar = fig.colorbar(scatter1, ax=axs[0], fraction=0.046, pad=0.04)
    cbar.set_label("Layer")
    
    # Plot uniform grid COB placement.
    axs[1].scatter(uniform_cobs[:, 0], uniform_cobs[:, 1],
                   c="blue", s=50, marker="s")
    axs[1].set_title("Uniform Grid COB Placement")
    axs[1].set_xlabel("X (m)")
    axs[1].set_ylabel("Y (m)")
    axs[1].set_aspect('equal', 'box')

    # Optionally, enforce the same limits on both plots if desired.
    # Enforce the same limits on both plots with a small margin.
    margin_x = (W_m) * 0.05
    margin_y = (L_m) * 0.05
    x_min = 0 - margin_x
    x_max = W_m + margin_x
    y_min = 0 - margin_y
    y_max = L_m + margin_y
    for ax in axs:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    
    plt.suptitle(f"COB Placement Visualizations\nFloor: {floor_width_ft}' x {floor_length_ft}', Height: {floor_height}'")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def generate_line_graphs_for_both(floor_width_ft=14.0, floor_length_ft=14.0, target_ppfd=1250.0, floor_height=3.0):
    """
    Runs both the dynamic simulation (optimized COB arrangement) and the uniform grid simulation.
    For each simulation, it computes the absolute deviation (DPS) of the PPFD at each measurement point
    from the average PPFD, sorts these deviations, prints key metrics, and then generates a line graph.
    
    Both line graphs are shown side by side.
    """
    # Run dynamic simulation (optimized COB arrangement)
    result_dynamic = run_ml_simulation(floor_width_ft, floor_length_ft, target_ppfd, floor_height, side_by_side=False)
    avg_ppfd_dynamic = result_dynamic.get("optimized_ppfd", None)
    ppfd_grid_dynamic = result_dynamic.get("heatmapGrid", None)
    
    if avg_ppfd_dynamic is None or ppfd_grid_dynamic is None:
        print("Error: Dynamic simulation did not return PPFD data.")
        return
    
    # Convert dynamic PPFD grid to a NumPy array and compute deviations (DPS)
    ppfd_array_dynamic = np.array(ppfd_grid_dynamic)
    dps_dynamic = np.abs(ppfd_array_dynamic.flatten() - avg_ppfd_dynamic)
    sorted_dps_dynamic = np.sort(dps_dynamic)
    
    # Print dynamic simulation metrics
    print("\n[Dynamic Simulation Metrics]")
    print(f"Average PPFD: {avg_ppfd_dynamic:.1f} µmol/m²/s")
    print(f"MAD: {result_dynamic.get('mad', 0):.3f}")
    print(f"RMSE: {result_dynamic.get('rmse', 0):.3f}")
    print(f"Uniformity (DOU): {result_dynamic.get('dou', 0):.1f}%")
    print(f"CV: {result_dynamic.get('cv', 0):.1f}%")
    
    # Run uniform grid simulation
    result_uniform = run_uniform_grid_simulation(floor_width_ft, floor_length_ft, target_ppfd, floor_height)
    avg_ppfd_uniform = result_uniform.get("grid_avg_ppfd", None)
    ppfd_grid_uniform = result_uniform.get("grid_ppfd_data", None)
    
    if avg_ppfd_uniform is None or ppfd_grid_uniform is None:
        print("Error: Uniform grid simulation did not return PPFD data.")
        return
    
    # Convert uniform PPFD grid to a NumPy array and compute deviations (DPS)
    ppfd_array_uniform = np.array(ppfd_grid_uniform)
    dps_uniform = np.abs(ppfd_array_uniform.flatten() - avg_ppfd_uniform)
    sorted_dps_uniform = np.sort(dps_uniform)
    
    # Print uniform grid simulation metrics
    print("\n[Uniform Grid Simulation Metrics]")
    print(f"Grid Arrangement (rows x cols): {result_uniform['grid_arrangement']['rows']} x {result_uniform['grid_arrangement']['cols']}")
    print(f"Uniform Flux: {result_uniform.get('uniform_flux', 0):.3f}")
    print(f"Average PPFD: {avg_ppfd_uniform:.1f} µmol/m²/s")
    print(f"MAD: {result_uniform.get('grid_mad', 0):.3f}")
    print(f"RMSE: {result_uniform.get('grid_rmse', 0):.3f}")
    print(f"Uniformity (DOU): {result_uniform.get('grid_dou', 0):.1f}%")
    print(f"CV: {result_uniform.get('grid_cv', 0):.1f}%")
    
    # Create a figure with two subplots for the line graphs
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot dynamic simulation DPS line graph
    axs[0].plot(sorted_dps_dynamic, color='dodgerblue', marker='o', linestyle='-', linewidth=1, markersize=3)
    axs[0].set_title('Dynamic Simulation DPS')
    axs[0].set_xlabel('Sorted Measurement Point Index')
    axs[0].set_ylabel('Absolute Deviation (DPS) [µmol/m²/s]')
    axs[0].grid(True)
    axs[0].set_ylim([0, sorted_dps_dynamic.max() * 1.05])
    
    # Plot uniform grid simulation DPS line graph
    axs[1].plot(sorted_dps_uniform, color='darkorange', marker='o', linestyle='-', linewidth=1, markersize=3)
    axs[1].set_title('Uniform Grid Simulation DPS')
    axs[1].set_xlabel('Sorted Measurement Point Index')
    axs[1].set_ylabel('Absolute Deviation (DPS) [µmol/m²/s]')
    axs[1].grid(True)
    axs[1].set_ylim([0, sorted_dps_uniform.max() * 1.05])
    
    plt.suptitle(f"DPS Comparison for Floor {floor_width_ft}' x {floor_length_ft}', Target PPFD: {target_ppfd}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Example usage in main:
def main():
    # Uncomment any other simulation functions you wish to run, then:
    run_simulation_verbose()  # (Dynamic simulation with verbose logging)
    generate_line_graphs_for_both(floor_width_ft=14.0, floor_length_ft=14.0, target_ppfd=1250.0, floor_height=3.0)
    # run_uniform_grid_simulation()  # if you want to run it separately as well

if __name__ == "__main__":
    main()

# Example usage in main:
def main():
    # Uncomment any other simulation functions you wish to run, then:
    run_simulation_verbose()  # (Dynamic simulation with verbose logging)
    #generate_line_graphs_for_both(floor_width_ft=14.0, floor_length_ft=14.0, target_ppfd=1250.0, floor_height=3.0)
    # run_uniform_grid_simulation()  # if you want to run it separately as well

    #run_uniform_grid_simulation()
if __name__ == "__main__":
    main()

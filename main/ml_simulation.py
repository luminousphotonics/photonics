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
from scipy.interpolate import griddata, RegularGridInterpolator


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

# Floor grid resolution (meters)
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
# Dynamic COB Positioning
# ------------------------------
def build_cob_positions(W, L, H):
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
    
    # Rotate by 45°, scale so outer COBs are near floor edge
    theta = math.radians(45)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
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
    # Use linspace so domain 0..W, 0..L is included
    num_x = int(round(W / FLOOR_GRID_RES)) + 1
    num_y = int(round(L / FLOOR_GRID_RES)) + 1
    xs = np.linspace(0, W, num_x)
    ys = np.linspace(0, L, num_y)
    X, Y = np.meshgrid(xs, ys)
    return X, Y

def build_floor_grid(W, L):
    return cached_build_floor_grid(W, L)


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
# Patches (Ceiling / Walls)
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
            patch_centers.append((cx, cy, H))
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
    
    patch_rad = iterative_radiosity_loop(
        patch_centers, patch_normals, patch_direct, patch_areas, patch_refl, NUM_RADIOSITY_BOUNCES
    )
    
    reflect_irr = compute_reflection_on_floor(
        X, Y, patch_centers, patch_normals, patch_areas, patch_rad, patch_refl
    )
    
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
    n = int(np.max(cob_positions[:, 3]))
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


# ------------------------------
# Visualization
# ------------------------------
def generate_surface_graph(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, cmap="jet") -> str:
    grid_density = 100
    x_grid, y_grid = np.mgrid[X.min():X.max():grid_density*1j,
                              Y.min():Y.max():grid_density*1j]
    
    z_grid = griddata(
        np.column_stack((X.flatten(), Y.flatten())),
        Z.flatten(), (x_grid, y_grid),
        method='linear', fill_value=np.nan
    )
    
    colorbar_max = Z.max() * 1.2
    norm = plt.Normalize(vmin=0, vmax=colorbar_max)
    facecolors = plt.get_cmap(cmap)(norm(z_grid))
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(
        x_grid, y_grid, np.zeros_like(z_grid),
        rstride=1, cstride=1, facecolors=facecolors,
        shade=False, antialiased=False
    )
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("PPFD")
    ax.set_title("3D Floor Heatmap")
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(0, 500)
    
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(z_grid)
    fig.colorbar(mappable, ax=ax, fraction=0.032, pad=0.04)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_heatmap(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, cmap="jet", 
                     overlay_intensity: bool=False, overlay_step: int=10) -> str:
    """
    Creates a heatmap for display by interpolating the full simulation dataset onto a
    high-resolution grid after rotating the floor grid by 45° (to match the COB rotation).
    
    The four corners and the center of the rotated grid are forced to exactly equal
    the true simulation values (after rotation), and the interior is smoothly interpolated.
    
    Optionally overlays text labels (downsampled by overlay_step) using the rotated grid.
    
    Returns a base64-encoded PNG string.
    """

    # Define the rotation angle (same as in build_cob_positions)
    theta = math.radians(45)

    # Extract original 1D coordinate arrays from X and Y (assumes uniform grid)
    xs = X[0, :]  # original x coordinates from 0 to W
    ys = Y[:, 0]  # original y coordinates from 0 to L

    # Compute the center of the floor
    centerX = xs.mean()  # or simply xs.min() + (xs.max()-xs.min())/2
    centerY = ys.mean()

    # Rotate the floor grid by theta about the center.
    # For each (x, y) in the original grid, compute rotated coordinates (x_rot, y_rot).
    X_rot = centerX + (X - centerX) * math.cos(theta) - (Y - centerY) * math.sin(theta)
    Y_rot = centerY + (X - centerX) * math.sin(theta) + (Y - centerY) * math.cos(theta)

    # Use the rotated grid for interpolation.
    # Extract 1D rotated coordinates from X_rot, Y_rot:
    xs_rot = X_rot[0, :]
    ys_rot = Y_rot[:, 0]

    # Create a high-resolution rotated grid.
    factor = 10  # adjust as needed
    x_hr = np.linspace(xs_rot.min(), xs_rot.max(), len(xs_rot) * factor)
    y_hr = np.linspace(ys_rot.min(), ys_rot.max(), len(ys_rot) * factor)
    X_hr, Y_hr = np.meshgrid(x_hr, y_hr)

    # Interpolate the simulation data (Z) onto the high-res grid.
    # RegularGridInterpolator expects grid in (y, x) order.
    interpolator = RegularGridInterpolator((ys_rot, xs_rot), Z, method="linear")
    pts_hr = np.column_stack((Y_hr.flatten(), X_hr.flatten()))
    Z_hr = interpolator(pts_hr).reshape(Y_hr.shape)

    # Force the anchor points (four corners and center) on the high-res grid to be the true values.
    # Since the simulation data is now considered in the rotated coordinate system,
    # the anchor points come from the rotated grid.
    Z_hr[0, 0]       = Z[0, 0]           # bottom-left
    Z_hr[0, -1]      = Z[0, -1]          # bottom-right
    Z_hr[-1, 0]      = Z[-1, 0]          # top-left
    Z_hr[-1, -1]     = Z[-1, -1]         # top-right
    center_hr_i = Z_hr.shape[0] // 2
    center_hr_j = Z_hr.shape[1] // 2
    center_orig_i = len(ys_rot) // 2
    center_orig_j = len(xs_rot) // 2
    Z_hr[center_hr_i, center_hr_j] = Z[center_orig_i, center_orig_j]

    # Build an extent for imshow based on the rotated grid.
    extent = [xs_rot.min(), xs_rot.max(), ys_rot.min(), ys_rot.max()]

    # Plot using imshow.
    fig, ax = plt.subplots(figsize=(8, 6))
    colorbar_max = Z_hr.max() * 1.2
    im = ax.imshow(Z_hr, cmap=cmap, origin="lower", extent=extent, vmin=0, vmax=colorbar_max, aspect="auto")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Heatmap with Anchored Corners & Center (Rotated)")
    cbar = fig.colorbar(im, ax=ax)
    ticks = np.linspace(0, colorbar_max, 6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(round(t))}" for t in ticks])

    # Optionally overlay text labels at the original (rotated) grid points.
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


# ------------------------------
# Final Entry Point
# ------------------------------
def run_ml_simulation(floor_width_ft, floor_length_ft, target_ppfd, progress_callback=None):
    ft2m = 3.28084
    W_m = floor_width_ft / ft2m
    L_m = floor_length_ft / ft2m
    H_m = 3.0 / ft2m
    best_params = optimize_lighting(W_m, L_m, H_m, target_ppfd, progress_callback=progress_callback)
    final_ppfd = simulate_lighting(best_params, W_m, L_m, H_m)
    mean_ppfd = np.mean(final_ppfd)
    mad = np.mean(np.abs(final_ppfd - mean_ppfd))
    
    X, Y = build_floor_grid(W_m, L_m)
    surface_graph_b64 = generate_surface_graph(X, Y, final_ppfd, cmap="jet")
    heatmap_b64 = generate_heatmap(X, Y, final_ppfd, cmap="jet", overlay_intensity=False)
    heatmap_overlay_b64 = generate_heatmap(X, Y, final_ppfd, cmap="jet", overlay_intensity=True, overlay_step=10)
    
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
        "heatmap": heatmap_b64,
        "heatmap_overlay": heatmap_overlay_b64,
        "heatmapGrid": final_ppfd.tolist()
    }


def run_simulation(floor_width_ft=14.0, floor_length_ft=14.0, target_ppfd=1250.0):
    result = run_ml_simulation(floor_width_ft, floor_length_ft, target_ppfd)
    print("\n[RESULT] Simulation Output:")
    print(result)

def main():
    run_simulation()

if __name__ == "__main__":
    main()

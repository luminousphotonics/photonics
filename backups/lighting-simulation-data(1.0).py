#!/usr/bin/env python3
import csv
import math
import numpy as np
from numba import njit, prange
from functools import lru_cache
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import argparse

# ------------------------------------
# 1) Basic Config & Reflectances
# ------------------------------------
REFL_WALL = 0.8
REFL_CEIL = 0.8
REFL_FLOOR = 0.1

LUMINOUS_EFFICACY = 182.0  # lumens/W
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

# --- LED Strip Module Configuration (Datasheet Based) --- ## NEW/REPLACE ##
STRIP_IES_FILE = "/Users/austinrouse/photonics/backups/Standard_Horti_G2.ies" # <<< !!! SET ACTUAL PATH !!!
STRIP_MODULE_LENGTH = 0.561 # meters (from 561.0 mm)
STRIP_MODULE_LUMENS = 8960.0  # lumens per module (from datasheet)
# Optional: Store other datasheet values if needed later
# STRIP_MODULE_PPF = 150.0
# STRIP_MODULE_POWER = 50.4

# --- Constants for emitter types --- ## ADD THESE ##
EMITTER_TYPE_COB = 0
EMITTER_TYPE_STRIP = 1

# --- COB Configuration --- ## MODIFY/ADD ##
COB_IES_FILE = "/Users/austinrouse/photonics/backups/cob.ies" # <<< !!! SET PATH TO THE COB IES FILE


# ------------------------------------
# 4.5) Load and Prepare IES Data for Strips (CORRECTED IMPORT)
# ------------------------------------

def parse_specific_ies_file(ies_filepath):
    """
    Parses a specific IESNA:LM-63-1995 file format.
    Extracts data needed for Numba simulation, assuming axial symmetry.
    """
    print(f"[IES - Custom] Attempting to parse file: {ies_filepath}")
    try:
        with open(ies_filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()] # Read non-empty lines
    except FileNotFoundError:
        print(f"[IES - Custom] Error: File not found at '{ies_filepath}'")
        return None, None, None
    except Exception as e:
        print(f"[IES - Custom] Error reading file '{ies_filepath}': {e}")
        return None, None, None

    try:
        line_idx = 0
        # Skip header lines until TILT=
        while not lines[line_idx].upper().startswith('TILT='):
            line_idx += 1
        tilt_line = lines[line_idx] # Keep tilt info if needed later
        line_idx += 1

        # --- Parameter Line 1 ---
        params1 = lines[line_idx].split()
        num_lamps = int(params1[0])
        lumens_per_lamp = float(params1[1]) # This is our normalization base
        multiplier = float(params1[2])
        num_v_angles = int(params1[3])
        num_h_angles = int(params1[4])
        photometric_type = int(params1[5]) # 1=C, 2=B, 3=A
        units_type = int(params1[6]) # 1=feet, 2=meters
        # width, length, height (indices 7, 8, 9) - ignored for point sources
        line_idx += 1

        # --- Parameter Line 2 ---
        params2 = lines[line_idx].split()
        ballast_factor = float(params2[0])
        # future_use = float(params2[1]) # Ignored
        input_watts = float(params2[2])
        line_idx += 1

        # --- Read Vertical Angles ---
        v_angles_list = []
        while len(v_angles_list) < num_v_angles:
            v_angles_list.extend([float(a) for a in lines[line_idx].split()])
            line_idx += 1
        vertical_angles = np.array(v_angles_list, dtype=np.float64)
        if len(vertical_angles) != num_v_angles:
            raise ValueError("Mismatch in vertical angle count")

        # --- Read Horizontal Angles ---
        h_angles_list = []
        while len(h_angles_list) < num_h_angles:
            h_angles_list.extend([float(a) for a in lines[line_idx].split()])
            line_idx += 1
        horizontal_angles = np.array(h_angles_list, dtype=np.float64)
        if len(horizontal_angles) != num_h_angles:
            raise ValueError("Mismatch in horizontal angle count")

        # --- Read Candela Values ---
        candela_list_flat = []
        while line_idx < len(lines):
            candela_list_flat.extend([float(c) for c in lines[line_idx].split()])
            line_idx += 1

        if len(candela_list_flat) != num_v_angles * num_h_angles:
            raise ValueError(f"Candela value count mismatch: Found {len(candela_list_flat)}, expected {num_v_angles * num_h_angles}")

        # Reshape based on IES standard order (V changes fastest for each H block)
        # Values are ordered: v0h0, v1h0, ..., vNh0, v0h1, v1h1, ..., vNh1, ...
        # So reshape directly into (num_h, num_v) then transpose for (num_v, num_h)
        candela_data_raw = np.array(candela_list_flat, dtype=np.float64)
        candela_data_2d = candela_data_raw.reshape((num_h_angles, num_v_angles)).T # Transpose needed

        # --- Prepare data for Numba (Axial Symmetry Assumption) ---
        ies_angles_deg = vertical_angles # Assume already sorted, verify if needed
        # Find index for horizontal angle 0
        zero_h_angle_index = np.argmin(np.abs(horizontal_angles - 0.0))
        if abs(horizontal_angles[zero_h_angle_index]) > 1.0:
             print(f"[IES - Custom] Warning: Horizontal angle closest to 0 is {horizontal_angles[zero_h_angle_index]:.1f}. Using this slice.")

        # Extract candela values for H=0
        ies_candelas = candela_data_2d[:, zero_h_angle_index]

        # Use lumens/lamp from header for normalization factor
        ies_file_lumens_norm = lumens_per_lamp

        print(f"[IES - Custom] Successfully parsed data.")
        print(f"[IES - Custom] Lumens/lamp (for norm): {ies_file_lumens_norm:.2f}, Watts: {input_watts:.2f}")
        print(f"[IES - Custom] V Angles: {num_v_angles} (0 to {ies_angles_deg[-1]:.1f}), H Angles: {num_h_angles}")
        return ies_angles_deg, ies_candelas, ies_file_lumens_norm

    except IndexError:
        print(f"[IES - Custom] Error: File ended unexpectedly or format incorrect near line {line_idx+1}.")
        return None, None, None
    except ValueError as ve:
        print(f"[IES - Custom] Error: Could not convert data to number or count mismatch near line {line_idx+1}: {ve}")
        return None, None, None
    except Exception as e:
        print(f"[IES - Custom] Unexpected error during parsing near line {line_idx+1}: {e}")
        return None, None, None


# --- Rest of script uses the returned NumPy arrays ---

# --- Load STRIP IES Data ---
print("\n--- Loading Strip IES Data ---")
STRIP_IES_ANGLES_DEG, STRIP_IES_CANDELAS, STRIP_IES_FILE_LUMENS_NORM = parse_specific_ies_file(STRIP_IES_FILE)
if STRIP_IES_ANGLES_DEG is None:
     raise SystemExit("Failed to load or process strip IES file using custom parser. Exiting.")

# --- Load COB IES Data ---
print("\n--- Loading COB IES Data ---")
COB_IES_ANGLES_DEG, COB_IES_CANDELAS, COB_IES_FILE_LUMENS_NORM = parse_specific_ies_file(COB_IES_FILE)
if COB_IES_ANGLES_DEG is None:
     raise SystemExit("Failed to load or process COB IES file using custom parser. Exiting.")

# ------------------------------------
# 3) Compute SPD-based µmol/J Factor
# ------------------------------------
def compute_conversion_factor(spd_file):
    try:
        # Load data, skipping header
        raw_spd = np.loadtxt(spd_file, delimiter=' ', skiprows=1)

        # Sort by wavelength and handle duplicates (average intensity)
        unique_wl, indices, counts = np.unique(raw_spd[:, 0], return_inverse=True, return_counts=True)
        avg_intens = np.bincount(indices, weights=raw_spd[:, 1]) / counts
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
    tot = np.trapz(intens, wl)
    tot_par = np.trapz(intens[mask_par], wl[mask_par])
    PAR_fraction = tot_par / tot if tot > 0 else 1.0

    # Calculate conversion factor using effective photon energy in PAR range
    wl_m = wl * 1e-9
    h, c, N_A = 6.626e-34, 3.0e8, 6.022e23 # Planck, Speed of Light, Avogadro

    # Weighted average wavelength in PAR range
    numerator = np.trapz(wl_m[mask_par] * intens[mask_par], wl_m[mask_par])
    denominator = np.trapz(intens[mask_par], wl_m[mask_par])
    # Prevent division by zero if PAR intensity is zero
    lambda_eff = numerator / denominator if denominator > 1e-15 else 0.0

    # Energy per photon at effective wavelength
    E_photon = (h * c / lambda_eff) if lambda_eff > 0 else 1.0 # Avoid division by zero

    # µmol/J = (photons/J) * (mol/photons) * (µmol/mol) * PAR_fraction
    conversion_factor = (1.0 / E_photon) * (1e6 / N_A) * PAR_fraction

    print(f"[INFO] SPD: Processed {len(wl)} unique points. PAR fraction={PAR_fraction:.3f}, conv_factor={conversion_factor:.5f} µmol/J")
    return conversion_factor

CONVERSION_FACTOR = compute_conversion_factor(SPD_FILE)

# ------------------------------------
# 4.6) Numba-Compatible Intensity Functions (Keep COB, Add IES) ## MODIFY/ADD ##
# ------------------------------------

# Rename the existing IES function slightly for clarity
@njit
def calculate_ies_intensity(angle_deg, total_emitter_lumens,
                            ies_angles_deg, ies_candelas, ies_file_lumens_norm):
    """
    Calculates luminous intensity (candela) for a given vertical angle (theta)
    based on axially symmetric IES data (works for both COB and Strip).
    Scales output based on the emitter's actual total lumens vs. the
    IES file's reference lumens.
    """
    candela_raw = np.interp(angle_deg, ies_angles_deg, ies_candelas)
    norm_factor = ies_file_lumens_norm if ies_file_lumens_norm > 1e-9 else 1.0
    scaling_factor = total_emitter_lumens / norm_factor
    return candela_raw * scaling_factor

# ------------------------------------
# 5) Geometry Building
# ------------------------------------
# Modify prepare_geometry to accept and pass strip_module_lumen_params
def prepare_geometry(W, L, H, cob_lumen_params, strip_module_lumen_params): # Add strip params
    """Prepares all geometry: light sources, floor grid, patches, strip vertices."""
    print("[Geometry] Building light sources (COBs + Strip Modules)...")
    # Pass both lumen parameter arrays to build_all_light_sources ## FIX CALL ##
    light_positions, light_lumens, light_types, cob_positions_only, ordered_strip_vertices = build_all_light_sources(
        W, L, H, cob_lumen_params, strip_module_lumen_params # Pass both here
    )
    # ... (rest of prepare_geometry is the same) ...
    print("[Geometry] Building floor grid...")
    X, Y = build_floor_grid(W, L)
    print("[Geometry] Building room patches (floor, ceiling, walls)...")
    patches = build_patches(W, L, H)
    return light_positions, light_lumens, light_types, cob_positions_only, ordered_strip_vertices, X, Y, patches

def build_cob_positions(W, L, H):
    # Creates a specific diamond-like pattern, scaled and rotated
    n = FIXED_NUM_LAYERS - 1
    positions = []
    positions.append((0, 0, H, 0)) # Center COB, layer 0
    for i in range(1, n + 1): # Layers 1 to n
        for x in range(-i, i + 1):
            y_abs = i - abs(x)
            if y_abs == 0:
                positions.append((x, 0, H, i)) # On x-axis
            else:
                positions.append((x, y_abs, H, i))  # Upper half
                positions.append((x, -y_abs, H, i)) # Lower half

    # Rotate by 45 degrees and scale to fit room dimensions
    theta = math.radians(45)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    centerX, centerY = W / 2, L / 2
    # Scaling factor based on max layer index 'n' to reach near corners
    scale_x = (W / 2 * math.sqrt(2)) / n if n > 0 else W / 2
    scale_y = (L / 2 * math.sqrt(2)) / n if n > 0 else L / 2

    transformed = []
    for (xx, yy, _, layer) in positions:
        # Apply rotation
        rx = xx * cos_t - yy * sin_t
        ry = xx * sin_t + yy * cos_t
        # Apply scaling and translate to center
        px = centerX + rx * scale_x
        py = centerY + ry * scale_y
        # Use H * 0.95 for Z-position (slightly below ceiling height H)
        transformed.append((px, py, H * 0.95, layer))

    return np.array(transformed, dtype=np.float64)

# --- Potential alternative for cached_build_floor_grid ---
@lru_cache(maxsize=32)
def cached_build_floor_grid(W: float, L: float):
    # Calculate number of cells
    num_cells_x = int(round(W / FLOOR_GRID_RES))
    num_cells_y = int(round(L / FLOOR_GRID_RES))
    # Calculate actual resolution based on cell count
    actual_res_x = W / num_cells_x if num_cells_x > 0 else W
    actual_res_y = L / num_cells_y if num_cells_y > 0 else L
    # Create coordinates for cell centers
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

    # Floor (single patch)
    patch_centers.append((W/2, L/2, 0.0))
    patch_areas.append(W * L)
    patch_normals.append((0.0, 0.0, 1.0)) # Normal pointing up (into room)
    patch_refl.append(REFL_FLOOR)

    # Ceiling Patches
    xs_ceiling = np.linspace(0, W, CEIL_SUBDIVS_X + 1)
    ys_ceiling = np.linspace(0, L, CEIL_SUBDIVS_Y + 1)
    for i in range(CEIL_SUBDIVS_X):
        for j in range(CEIL_SUBDIVS_Y):
            cx = (xs_ceiling[i] + xs_ceiling[i+1]) / 2
            cy = (ys_ceiling[j] + ys_ceiling[j+1]) / 2
            area = (xs_ceiling[i+1] - xs_ceiling[i]) * (ys_ceiling[j+1] - ys_ceiling[j])
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
        coords1 = np.linspace(range1[0], range1[1], subdivs1 + 1)
        coords2 = np.linspace(range2[0], range2[1], subdivs2 + 1) # Always Z axis (height)

        for i in range(subdivs1):
            for j in range(subdivs2):
                c1 = (coords1[i] + coords1[i+1]) / 2
                c2 = (coords2[j] + coords2[j+1]) / 2 # This is cz
                area = (coords1[i+1] - coords1[i]) * (coords2[j+1] - coords2[j])

                if axis == 'y':
                    center = (c1, fixed_val, c2) # c1 is cx
                else: # axis == 'x'
                    center = (fixed_val, c1, c2) # c1 is cy

                patch_centers.append(center)
                patch_areas.append(area)
                patch_normals.append(normal) # Use the CORRECTED inward-pointing normal
                patch_refl.append(REFL_WALL)

    return (np.array(patch_centers, dtype=np.float64),
            np.array(patch_areas, dtype=np.float64),
            np.array(patch_normals, dtype=np.float64),
            np.array(patch_refl, dtype=np.float64))

def build_patches(W, L, H):
    return cached_build_patches(W, L, H)

# --- New/Modified functions for COBs and Strips ---

def _get_cob_abstract_coords_and_transform(W, L, H, num_total_layers):
    """Internal helper to get abstract COB coords and transform parameters."""
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
    scale_x = W / (n * math.sqrt(2)) if n > 0 else W / 2 # Avoid division by zero
    scale_y = L / (n * math.sqrt(2)) if n > 0 else L / 2 # Avoid division by zero

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

# ------------------------------------
# 5) Geometry Building (Modify build_all_light_sources and prepare_geometry)
# ------------------------------------
# --- Keep _get_cob_abstract_coords_and_transform ---
# --- Keep _apply_transform ---
# --- Keep cached_build_floor_grid, build_floor_grid ---
# --- Keep cached_build_patches, build_patches ---

# !! Modify build_all_light_sources function SIGNATURE and STRIP Lumen Assignment !!
# Add strip_module_lumen_params to the signature
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
         raise ValueError("Length mismatch between cob_lumen_params and strip_module_lumen_params")
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
        px, py, pz = _apply_transform(abs_cob, transform_params)
        lumens = cob_lumen_params[layer] if 0 <= layer < num_total_layers else 0
        all_positions_list.append([px, py, pz])
        all_lumens_list.append(lumens)
        all_types_list.append(EMITTER_TYPE_COB)
        cob_positions_only_list.append([px, py, pz, layer])

    # Process Strips (Layers 1 onwards) using strip_module_lumen_params
    print("[Geometry] Placing strip modules...")
    total_modules_placed = 0
    for layer_idx in range(1, num_total_layers):
        # ... (Vertex finding and sorting remains the same) ...
        layer_vertices_abstract = [p for p in abstract_cob_coords if p['layer'] == layer_idx and p['is_vertex']]
        layer_vertices_abstract.sort(key=lambda p: math.atan2(p['y'], p['x']))
        if not layer_vertices_abstract: continue
        transformed_vertices = [_apply_transform(av, transform_params) for av in layer_vertices_abstract]
        ordered_strip_vertices[layer_idx] = transformed_vertices
        num_vertices = len(transformed_vertices)
        modules_this_layer = 0

        # Get the target lumens for modules in this specific layer ## GET LAYER LUMENS ##
        target_module_lumens = strip_module_lumen_params[layer_idx] if 0 <= layer_idx < num_total_layers else 0

        if target_module_lumens <= 1e-6: # Check target lumens ## MODIFY CHECK ##
             print(f"[INFO] Layer {layer_idx}: Skipping modules due to zero/low target lumens.")
             continue

        # Iterate through sections connecting the main COB vertices
        for i in range(num_vertices):
            # ... (Module placement logic remains the same) ...
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
                # Assign the TARGET lumens for this layer's modules ## MODIFY THIS ##
                all_lumens_list.append(target_module_lumens)
                all_types_list.append(EMITTER_TYPE_STRIP)
                modules_this_layer += 1

        if modules_this_layer > 0:
             print(f"[INFO] Layer {layer_idx}: Placed {modules_this_layer} strip modules (Target Lumens={target_module_lumens:.1f}).") # Updated print
             total_modules_placed += modules_this_layer

    # ... (Convert lists to NumPy arrays - check and error handling remains the same) ...
    try:
        light_positions = np.array(all_positions_list, dtype=np.float64)
        if not all(isinstance(l, (int, float)) for l in all_lumens_list):
             print("[ERROR] Invalid data found in lumen list:", all_lumens_list)
             raise TypeError("Lumen list contains non-numeric data")
        light_lumens = np.array(all_lumens_list, dtype=np.float64)
        light_types = np.array(all_types_list, dtype=np.int32)
        cob_positions_only = np.array(cob_positions_only_list, dtype=np.float64)
    except Exception as e:
         print(f"[ERROR] Failed to convert geometry lists to NumPy arrays: {e}")
         raise

    print(f"[INFO] Generated {len(cob_positions_only)} main COBs.")
    if total_modules_placed > 0:
         print(f"[INFO] Placed {total_modules_placed} strip modules across all layers.")

    return light_positions, light_lumens, light_types, cob_positions_only, ordered_strip_vertices

# ------------------------------------
# 6) Numba-JIT Computations (Modify Kernels) ## MODIFY FUNCTIONS ##
# ------------------------------------

@njit(parallel=True)
def compute_direct_floor(light_positions, light_lumens, light_types,
                         X, Y,
                         # Pass both sets of IES data
                         cob_ies_angles, cob_ies_candelas, cob_ies_norm,
                         strip_ies_angles, strip_ies_candelas, strip_ies_norm
                         ):
    min_dist2 = (FLOOR_GRID_RES / 2.0) ** 2
    rows, cols = X.shape
    out = np.zeros_like(X, dtype=np.float64)
    num_lights = light_positions.shape[0]

    for r in prange(rows):
        for c in range(cols):
            fx, fy = X[r, c], Y[r, c]
            val = 0.0
            for k in range(num_lights):
                # ... (Get lx, ly, lz, lumens_k, type_k - same) ...
                lx, ly, lz = light_positions[k, 0], light_positions[k, 1], light_positions[k, 2]
                lumens_k = light_lumens[k]
                type_k = light_types[k]

                # ... (Calculate dx, dy, dz, d2, dist, cos_theta, angle_deg_theta - same) ...
                dx, dy, dz = fx - lx, fy - ly, 0.0 - lz
                d2 = dx*dx + dy*dy + dz*dz
                d2 = max(d2, min_dist2)
                dist = math.sqrt(d2)
                cos_theta = -dz / dist
                if cos_theta < 1e-9: continue
                cos_theta = min(cos_theta, 1.0)
                angle_deg_theta = math.degrees(math.acos(cos_theta))

                # --- Select intensity calculation using correct IES data --- ## MODIFY LOGIC ##
                I_theta = 0.0
                if type_k == EMITTER_TYPE_COB:
                    # Use the single IES function with COB data
                    I_theta = calculate_ies_intensity(angle_deg_theta, lumens_k,
                                                      cob_ies_angles, cob_ies_candelas, cob_ies_norm)
                elif type_k == EMITTER_TYPE_STRIP:
                    # Use the single IES function with STRIP data
                    I_theta = calculate_ies_intensity(angle_deg_theta, lumens_k,
                                                      strip_ies_angles, strip_ies_candelas, strip_ies_norm)
                # ---------------------------------------------------------

                # ... (Calculate E_local, val += E_local - same) ...
                cos_in_floor = cos_theta
                E_local = (I_theta / d2) * cos_in_floor
                val += E_local
            out[r, c] = val
    return out


@njit
def compute_patch_direct(light_positions, light_lumens, light_types,
                         patch_centers, patch_normals, patch_areas,
                         # Pass both sets of IES data
                         cob_ies_angles, cob_ies_candelas, cob_ies_norm,
                         strip_ies_angles, strip_ies_candelas, strip_ies_norm
                         ):
    min_dist2 = (FLOOR_GRID_RES / 2.0) ** 2
    Np = patch_centers.shape[0]
    num_lights = light_positions.shape[0]
    out = np.zeros(Np, dtype=np.float64)

    for ip in range(Np):
        # ... (Get pc, n, norm_n - same) ...
        pc = patch_centers[ip]
        n = patch_normals[ip]
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-9: continue

        accum = 0.0
        for k in range(num_lights):
            # ... (Get lx, ly, lz, lumens_k, type_k - same) ...
            lx, ly, lz = light_positions[k, 0], light_positions[k, 1], light_positions[k, 2]
            lumens_k = light_lumens[k]
            type_k = light_types[k]

            # ... (Calculate dx, dy, dz, d2, dist, cos_theta, angle_deg_theta - same) ...
            dx, dy, dz = pc[0] - lx, pc[1] - ly, pc[2] - lz
            d2 = dx*dx + dy*dy + dz*dz
            d2 = max(d2, min_dist2)
            dist = math.sqrt(d2)
            cos_theta = -dz / dist
            if cos_theta < 1e-9: continue
            cos_theta = min(cos_theta, 1.0)
            angle_deg_theta = math.degrees(math.acos(cos_theta))

            # --- Select intensity calculation using correct IES data --- ## MODIFY LOGIC ##
            I_theta = 0.0
            if type_k == EMITTER_TYPE_COB:
                I_theta = calculate_ies_intensity(angle_deg_theta, lumens_k,
                                                  cob_ies_angles, cob_ies_candelas, cob_ies_norm)
            elif type_k == EMITTER_TYPE_STRIP:
                I_theta = calculate_ies_intensity(angle_deg_theta, lumens_k,
                                                  strip_ies_angles, strip_ies_candelas, strip_ies_norm)
            # ---------------------------------------------------------

            # ... (Calculate dot_patch, cos_in_patch, E_local, accum += E_local - same) ...
            dot_patch = -(dx*n[0] + dy*n[1] + dz*n[2])
            cos_in_patch = dot_patch / (dist * norm_n)
            cos_in_patch = max(0.0, cos_in_patch)
            E_local = (I_theta / d2) * cos_in_patch
            accum += E_local
        out[ip] = accum
    return out


@njit
def iterative_radiosity_loop(patch_centers, patch_normals, patch_direct, patch_areas, patch_refl,
                             max_bounces, convergence_threshold):
    # Calculates the total radiosity (outgoing flux density) of each patch including reflections
    Np = patch_direct.shape[0]
    patch_rad = patch_direct.copy() # Initialize with direct illuminance (converted to radiosity implicitly)
    patch_exitance = patch_direct * patch_refl # Initial exitance for first bounce calculation
    epsilon = 1e-6 # For relative change calculation

    for bounce in range(max_bounces):
        new_incident_flux = np.zeros(Np, dtype=np.float64)

        # Calculate flux transfer between all pairs of patches
        for j in range(Np): # Source patch j
            if patch_refl[j] <= 0: continue # Skip if patch doesn't reflect

            # Total flux leaving patch j
            outgoing_flux_j = patch_exitance[j] * patch_areas[j] # Flux = Exitance * Area
            if outgoing_flux_j <= 0: continue

            pj = patch_centers[j]
            nj = patch_normals[j]
            norm_nj = np.linalg.norm(nj)

            for i in range(Np): # Destination patch i
                if i == j: continue # Patch doesn't reflect to itself

                pi = patch_centers[i]
                ni = patch_normals[i]
                norm_ni = np.linalg.norm(ni)

                # Vector from center of patch j to center of patch i
                vij = pi - pj
                dist2 = np.dot(vij, vij)
                if dist2 < 1e-15: continue # Avoid self-transfer or coincident patches
                dist = math.sqrt(dist2)

                # Cosines of angles relative to normals
                cos_j = np.dot(nj, vij) / (norm_nj * dist) # Angle at source patch j
                cos_i = np.dot(ni, -vij) / (norm_ni * dist) # Angle at destination patch i

                # Check visibility and orientation
                if cos_j <= epsilon or cos_i <= epsilon: continue # Patches facing away or edge-on

                # Form Factor approximation (point-to-point)
                form_factor_ji = (cos_j * cos_i) / (math.pi * dist2) * patch_areas[i] # Should be dAj here for F_ji? No, this is dFlux_i = B_j * dA_j * F_ji
                form_factor_approx_ji = (cos_j * cos_i) / (math.pi * dist2) # Geometric term G = cos_j * cos_i / (pi * r^2)
                flux_transfer = outgoing_flux_j * form_factor_approx_ji * patch_areas[i] # Flux_j * G * Area_i
                new_incident_flux[i] += flux_transfer

        # Update radiosity and exitance for next bounce, check convergence
        max_rel_change = 0.0
        new_patch_rad = np.empty_like(patch_rad)
        new_patch_exitance = np.empty_like(patch_exitance)

        for i in range(Np):
            # New incident irradiance (flux density) on patch i
            incident_irradiance_i = new_incident_flux[i] / patch_areas[i] if patch_areas[i] > 0 else 0.0
            # New total radiosity = Emitted (none here) + Reflected
            # Reflected = Incident * Reflectance
            # Total Radiosity (B) = E_direct + E_indirect = E_direct + Incident_from_others * Reflectance
            # The variable `patch_rad` here seems to store total incident irradiance (direct + indirect)
            new_total_incident_i = patch_direct[i] + incident_irradiance_i
            new_patch_rad[i] = new_total_incident_i # Update total incident irradiance
            new_patch_exitance[i] = new_total_incident_i * patch_refl[i] # Update exitance

            # Check convergence based on change in incident irradiance
            change = abs(new_patch_rad[i] - patch_rad[i])
            denom = abs(patch_rad[i]) + epsilon
            rel_change = change / denom
            if rel_change > max_rel_change:
                max_rel_change = rel_change

        patch_rad = new_patch_rad # Store total incident irradiance for next iteration's convergence check
        patch_exitance = new_patch_exitance # Store exitance for next iteration's flux transfer calc

        if max_rel_change < convergence_threshold:
            #print(f"Radiosity converged after {bounce+1} bounces. Max change: {max_rel_change:.2e}")
            break
    #else:
        #print(f"Radiosity loop reached max {max_bounces} bounces. Max change: {max_rel_change:.2e}")

    # Return the final total incident irradiance on each patch
    return patch_rad


# Using joblib for parallel processing of floor points/rows
def compute_reflection_on_floor(X, Y, patch_centers, patch_normals, patch_areas, patch_rad, patch_refl,
                                mc_samples=MC_SAMPLES):
    # Calculates the indirect illuminance on the floor grid from reflected light off patches
    rows, cols = X.shape
    # Calculate patch exitance B = Incident_Total * Reflectance
    patch_exitance = patch_rad * patch_refl

    # Parallel computation over rows of the floor grid
    results = Parallel(n_jobs=-1)(delayed(compute_row_reflection)(
        r, X, Y, patch_centers, patch_normals, patch_areas, patch_exitance, mc_samples
    ) for r in range(rows))

    # Assemble results into the output array
    out = np.zeros((rows, cols), dtype=np.float64)
    for r, row_vals in enumerate(results):
        out[r, :] = row_vals
    return out

@njit
def compute_row_reflection(r, X, Y, patch_centers, patch_normals, patch_areas, patch_exitance, mc_samples):
    # Computes indirect illuminance for a single row of the floor grid using Monte Carlo
    row_vals = np.empty(X.shape[1], dtype=np.float64) # Pre-allocate row results

    for c in range(X.shape[1]): # Iterate through columns (points in the row)
        fx, fy = X[r, c], Y[r, c] # Target floor point (z=0)
        total_indirect_E = 0.0

        for p in range(patch_centers.shape[0]): # Iterate through all source patches
            # Flux leaving patch p = Exitance * Area
            outgoing_flux_p = patch_exitance[p] * patch_areas[p]
            if outgoing_flux_p <= 1e-9: continue # Skip patches contributing negligible flux

            pc = patch_centers[p]
            n = patch_normals[p] # Normal of source patch p
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-9: continue # Skip invalid patches

            # Define tangent vectors for sampling on the patch surface
            # Robustly find a vector not parallel to n
            if abs(n[0]) > 0.9: # If normal is mostly along X
                v_tmp = np.array((0.0, 1.0, 0.0))
            else:
                v_tmp = np.array((1.0, 0.0, 0.0))
            # Create tangent 1 using cross product, normalize
            tangent1 = np.cross(n, v_tmp)
            tangent1_norm = np.linalg.norm(tangent1)
            if tangent1_norm < 1e-9: # Handle case where n and v_tmp might be parallel (shouldn't happen with above logic)
                 tangent1 = np.cross(n, np.array((0.0, 0.0, 1.0))) # Try Z axis
                 tangent1_norm = np.linalg.norm(tangent1)
                 if tangent1_norm < 1e-9 : tangent1 = np.array((1.0, 0.0, 0.0)) # Failsafe
                 else: tangent1 /= tangent1_norm
            else:
                tangent1 = tangent1 / tangent1_norm
            # Create tangent 2 using cross product, ensure normalization
            tangent2 = np.cross(n, tangent1)
            tangent2_norm = np.linalg.norm(tangent2)
            if tangent2_norm > 1e-9 : tangent2 /= tangent2_norm
            else: tangent2 = np.cross(n, tangent1 + np.array((0.1,0.2,0.3))) # Failsafe slightly perturbed
            # Assume square patch for sampling offsets for simplicity
            # More accurate would be to know the patch dimensions
            half_side = math.sqrt(patch_areas[p]) / 2.0

            sample_sum_E = 0.0 # Accumulator for illuminance from this patch p
            for _ in range(mc_samples):
                # Generate random offsets within the patch area
                offset1 = np.random.uniform(-half_side, half_side)
                offset2 = np.random.uniform(-half_side, half_side)
                sample_point = pc + offset1 * tangent1 + offset2 * tangent2 # Random point on patch p

                # Vector from sample point on patch p to floor point (fx, fy, 0)
                dx = fx - sample_point[0]
                dy = fy - sample_point[1]
                dz = 0.0 - sample_point[2] # dz = -sample_point_z
                dist2 = dx*dx + dy*dy + dz*dz
                if dist2 < 1e-15: continue # Avoid singularity
                dist = math.sqrt(dist2)

                # Cosine angle at floor point (surface normal is (0,0,1))
                cos_f = -dz / dist # = (sample_point_z) / dist
                cos_f = max(0.0, cos_f) # Clamp if point is below floor (shouldn't happen)

                # Cosine angle at source patch sample point (normal n)
                # Vector is (dx, dy, dz)
                dot_p = n[0]*dx + n[1]*dy + n[2]*dz
                cos_p = dot_p / (norm_n * dist)
                cos_p = max(0.0, cos_p) # Clamp if floor point is behind patch surface

                exitance_p = patch_exitance[p]
                if exitance_p > 0:
                    sample_E = (exitance_p / math.pi) * (cos_p * cos_f) / dist2
                    sample_sum_E += sample_E

            # Average the contribution and multiply by patch area
            avg_sample_E = sample_sum_E / mc_samples if mc_samples > 0 else 0.0
            total_indirect_E += avg_sample_E * patch_areas[p] # E = (Area/N)*Sum(...) -> Add Area * Avg(Samples)

        row_vals[c] = total_indirect_E # Store calculated indirect illuminance for this point
    return row_vals

# ------------------------------------
# 7) Heatmap Plotting Function (Modified)
# ------------------------------------
# Modify the function signature to accept ordered_strip_vertices
def plot_heatmap(floor_ppfd, X, Y, cob_marker_positions, ordered_strip_vertices, W, L, annotation_step=10):
    fig, ax = plt.subplots(figsize=(10, 8))
    extent = [0, W, 0, L]
    im = ax.imshow(floor_ppfd, cmap='viridis', interpolation='nearest',
                   origin='lower', extent=extent, aspect='equal')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('PPFD (µmol/m²/s)')

    # --- Annotations (Keep existing code) ---
    # ... (annotation logic remains the same) ...
    rows, cols = floor_ppfd.shape
    if annotation_step > 0:
        step_r = max(1, rows // annotation_step)
        step_c = max(1, cols // annotation_step)
        for r in range(0, rows, step_r):
             for c in range(0, cols, step_c):
                text_x = X[0, c] + (X[0,1]-X[0,0])/2 if c > 0 else X[0,0]/2
                text_y = Y[r, 0] + (Y[1,0]-Y[0,0])/2 if r > 0 else Y[0,0]/2
                text_x = np.clip(text_x, extent[0], extent[1])
                text_y = np.clip(text_y, extent[2], extent[3])
                ax.text(text_x, text_y, f"{floor_ppfd[r, c]:.1f}",
                        ha="center", va="center", color="white", fontsize=6,
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.5))


    # --- Plot Strip Lines ---
    strip_handles = [] # For legend
    cmap_strips = plt.cm.cool # Use a different colormap for strips? Or same as COBs?
    num_strip_layers = len(ordered_strip_vertices)
    colors_strips = cmap_strips(np.linspace(0, 1, max(1, num_strip_layers))) # Avoid div by zero

    for layer_idx, vertices in ordered_strip_vertices.items():
        if not vertices: continue
        num_vertices = len(vertices)
        strip_color = colors_strips[layer_idx - 1] # Index from 0 for color map

        for i in range(num_vertices):
            p1 = vertices[i]        # Start vertex (x,y,z)
            p2 = vertices[(i + 1) % num_vertices] # End vertex (wraps around)

            # Use plot for lines
            line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], # Use only x, y for 2D plot
                     color=strip_color, linewidth=2.0, linestyle='--', # Dashed line for strips?
                     alpha=0.7, zorder=2) # Plot strips behind COB markers

        # Add proxy artist for legend
        if num_vertices > 0 and layer_idx not in [h.get_label().split()[-1] for h in strip_handles]:
            strip_handles.append(plt.Line2D([0], [0], color=strip_color, lw=2.0, linestyle='--', label=f'Strip Layer {layer_idx}'))
    # --- End Plot Strip Lines ---


    # Plot ONLY Main COB positions as markers
    if cob_marker_positions is not None and len(cob_marker_positions) > 0:
        ax.scatter(cob_marker_positions[:, 0], cob_marker_positions[:, 1], marker='o',
                   color='red', edgecolors='black', s=50, label="Main COB positions", alpha=0.8, zorder=3)

    ax.set_title("Floor PPFD Distribution (COBs + Strips)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # Combine legends
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + strip_handles, fontsize='small', loc='center left', bbox_to_anchor=(1.02, 0.5)) # Adjust bbox

    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend

# ------------------------------------
# 8) CSV Output Function (Organized by Layer Rings) - CORRECTED
# ------------------------------------
def write_ppfd_to_csv(filename, floor_ppfd, X, Y, cob_positions, W, L):
    """
    Writes PPFD data to CSV, assigning each floor point to a distance-based ring (layer).
    Ensures layer 0 has a non-zero radius if FIXED_NUM_LAYERS > 1.
    """
    layer_data = {i: [] for i in range(FIXED_NUM_LAYERS)} # Initialize list for each layer

    # Calculate layer radii based on COB positions
    center_x, center_y = W / 2, L / 2
    max_dist_per_layer = np.zeros(FIXED_NUM_LAYERS)

    if cob_positions is not None and len(cob_positions) > 0:
        cob_layers = cob_positions[:, 3].astype(int)
        distances_from_center = np.sqrt((cob_positions[:, 0] - center_x)**2 +
                                        (cob_positions[:, 1] - center_y)**2)

        for i in range(FIXED_NUM_LAYERS):
            layer_mask = (cob_layers == i)
            if np.any(layer_mask):
                max_dist_per_layer[i] = np.max(distances_from_center[layer_mask])
            elif i > 0:
                max_dist_per_layer[i] = max_dist_per_layer[i-1] # If layer empty, use previous max radius
            # Layer 0 distance is handled correctly as 0 if it exists

    # Define ring boundaries (outer radius of each layer's ring)
    layer_radii_outer = np.sort(max_dist_per_layer)

    # --- START FIX ---
    # Ensure layer 0 has a small radius if it's currently 0 and other layers exist
    if FIXED_NUM_LAYERS > 1 and abs(layer_radii_outer[0]) < 1e-9 and layer_radii_outer[1] > 1e-9:
         # Set layer 0 radius to half the grid resolution, or half the distance to layer 1, whichever is smaller
         radius_1 = layer_radii_outer[1]
         min_radius = min(FLOOR_GRID_RES / 2.0, radius_1 / 2.0)
         # Ensure the minimum radius is still positive
         layer_radii_outer[0] = max(min_radius, 1e-6)
    elif FIXED_NUM_LAYERS == 1:
         # If there's only one layer (layer 0), let its radius be half the grid res
         layer_radii_outer[0] = max(layer_radii_outer[0], FLOOR_GRID_RES / 2.0)
    # --- END FIX ---


    # Add a small epsilon to the last radius to ensure the furthest points are included
    layer_radii_outer[-1] += 0.01 * FLOOR_GRID_RES

    # Assign each floor grid point to a layer
    rows, cols = floor_ppfd.shape
    for r in range(rows):
        for c in range(cols):
            fx, fy = X[r, c], Y[r, c]
            dist_to_center = np.sqrt((fx - center_x)**2 + (fy - center_y)**2)

            # Find which ring the point falls into
            assigned_layer = -1
            for i in range(FIXED_NUM_LAYERS):
                # Layer i covers the range: (radius[i-1], radius[i]]
                # Except for layer 0 which covers [0, radius[0]]
                outer_radius = layer_radii_outer[i]
                inner_radius = layer_radii_outer[i-1] if i > 0 else 0.0

                # Check if distance is within the bounds [inner, outer]
                # Handle layer 0 separately to include 0 distance explicitly
                if i == 0:
                    if 0.0 <= dist_to_center <= outer_radius:
                        assigned_layer = 0
                        break
                else:
                    # Use inner < dist <= outer for subsequent layers
                    if inner_radius < dist_to_center <= outer_radius:
                        assigned_layer = i
                        break

            # Fallback: if slightly outside the largest radius due to grid/float issues, assign to the last layer
            if assigned_layer == -1 and dist_to_center > layer_radii_outer[-1] and dist_to_center < layer_radii_outer[-1] * 1.01:
                 assigned_layer = FIXED_NUM_LAYERS - 1

            if assigned_layer != -1:
                layer_data[assigned_layer].append(floor_ppfd[r, c])
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
# 9) Main Simulation Function (Corrected Conversion)
# ------------------------------------
def simulate_lighting(light_positions, light_lumens, light_types, X, Y, patches):
    p_centers, p_areas, p_normals, p_refl = patches

    print("[Simulation] Calculating direct floor illuminance...")
    floor_lux_direct = compute_direct_floor(light_positions, light_lumens, light_types,
                                            X, Y,
                                            COB_IES_ANGLES_DEG, COB_IES_CANDELAS, COB_IES_FILE_LUMENS_NORM,
                                            STRIP_IES_ANGLES_DEG, STRIP_IES_CANDELAS, STRIP_IES_FILE_LUMENS_NORM)

    print("[Simulation] Calculating direct patch illuminance...")
    patch_direct_lux = compute_patch_direct(light_positions, light_lumens, light_types,
                                            p_centers, p_normals, p_areas,
                                            COB_IES_ANGLES_DEG, COB_IES_CANDELAS, COB_IES_FILE_LUMENS_NORM,
                                            STRIP_IES_ANGLES_DEG, STRIP_IES_CANDELAS, STRIP_IES_FILE_LUMENS_NORM)

    print("[Simulation] Running radiosity...")
    patch_total_incident_lux = iterative_radiosity_loop(
        p_centers, p_normals, patch_direct_lux, p_areas, p_refl,
        MAX_RADIOSITY_BOUNCES, RADIOSITY_CONVERGENCE_THRESHOLD
    )

    print("[Simulation] Calculating indirect floor illuminance (Monte Carlo)...")
    floor_lux_indirect = compute_reflection_on_floor(
        X, Y, p_centers, p_normals, p_areas, patch_total_incident_lux, p_refl, MC_SAMPLES
    )

    # --- Combining results and converting to PPFD --- ## ENSURE THIS PART IS CORRECT ##
    total_floor_lux = floor_lux_direct + floor_lux_indirect

    # Convert total lux to W/m^2 using luminous efficacy
    # Add protection against division by zero if efficacy is somehow zero
    effic = LUMINOUS_EFFICACY if LUMINOUS_EFFICACY > 1e-9 else 1.0
    total_radiant_Wm2 = total_floor_lux / effic # <<< THIS LINE WAS LIKELY MISSING/COMMENTED

    # Convert W/m^2 to PPFD (µmol/m²/s) using the PAR conversion factor
    floor_ppfd = total_radiant_Wm2 * CONVERSION_FACTOR
    # --------------------------------------------------------------------------

    return floor_ppfd

# ------------------------------------
# 10) Execution Block (Corrected and Complete)
# ------------------------------------
def main():
    # --- Simulation-Specific Parameters ---
    W = 6.096
    L = 6.096
    H = 0.9144

    # COB Lumen parameters per layer
    cob_params = np.array([
        10000, 
        10000,
        10000,
        10000,
        10000,
        10000,
        10000,
        10000,
        10000,
        10000,
    ], dtype=np.float64)

    # --- LED Strip Module Configuration & Brightness Params --- ## MODIFY/ADD ##
    COB_IES_FILE = "/Users/austinrouse/photonics/backups/cob.ies" # Your COB IES path
    STRIP_IES_FILE = "/Users/austinrouse/photonics/backups/Standard_Horti_G2.ies" # Your Strip IES path
    STRIP_MODULE_LENGTH = 0.561 # meters
    # Remove the fixed STRIP_MODULE_LUMENS constant
    # STRIP_MODULE_LUMENS = 8960.0 # No longer needed here

    # NEW: Strip Module Lumen parameters per layer (like cob_params)
    # Set the desired total lumens for each module in each layer
    strip_module_lumen_params = np.array([
        0.0,  # Layer 0 (No strip, value ignored but needed for length match)
        1000.0,  # Layer 1: Target lumens for each module (e.g., slightly dimmed)
        1000.0,  # Layer 2: Target lumens
        1000.0,  # Layer 3
        1000.0,  # Layer 4
        2000.0,  # Layer 5: Full brightness (matches datasheet value)
        3000.0,  # Layer 6
        4000.0,  # Layer 7
        5000.0,  # Layer 8
        8000.0,  # Layer 9
    ], dtype=np.float64)
    # -----------------------------------------------------------

    global FIXED_NUM_LAYERS
    FIXED_NUM_LAYERS = len(cob_params)
    # Add validation for strip_module_lumen_params length
    if len(strip_module_lumen_params) != FIXED_NUM_LAYERS:
        raise ValueError("strip_module_lumen_params length must match FIXED_NUM_LAYERS")

    # --- Ensure IES data is loaded (Check globals - remains same) ---
    if STRIP_IES_ANGLES_DEG is None or COB_IES_ANGLES_DEG is None:
         raise SystemExit("One or both IES data files failed to load. Cannot proceed.")

    print("Preparing geometry (COBs, Strip Modules, Room)...")
    # Update call to prepare_geometry - pass both lumen parameter arrays ## FIX CALL ##
    light_positions, light_lumens, light_types, cob_positions_only, ordered_strip_vertices, X, Y, patches = prepare_geometry(
        W, L, H, cob_params, strip_module_lumen_params # Pass both here
    )

    total_lights = len(light_positions)
    if total_lights == 0:
        print("Error: No light sources generated. Exiting.")
        return

    print(f"\nStarting simulation for {total_lights} total light emitters...")
    # Pass light_types etc. to simulate_lighting
    floor_ppfd = simulate_lighting(light_positions, light_lumens, light_types, X, Y, patches)

    print("Simulation complete. Calculating statistics...")
    # --- Statistics Calculation ---
    mean_ppfd = np.mean(floor_ppfd)
    std_dev = np.std(floor_ppfd)
    min_ppfd = np.min(floor_ppfd)
    max_ppfd = np.max(floor_ppfd)
    mad = np.mean(np.abs(floor_ppfd - mean_ppfd))
    rmse = np.sqrt(np.mean((floor_ppfd - mean_ppfd)**2))

    # Uniformity Metrics
    if mean_ppfd > 1e-9:
        cv_percent = (std_dev / mean_ppfd) * 100
        min_max_ratio = min_ppfd / max_ppfd if max_ppfd > 0 else 0
        min_avg_ratio = min_ppfd / mean_ppfd
        cu_percent = (1 - mad / mean_ppfd) * 100
        dou_percent = (1 - rmse / mean_ppfd) * 100
    else:
        cv_percent, min_max_ratio, min_avg_ratio, cu_percent, dou_percent = 0, 0, 0, 0, 0

    # --- Output Results ---
    print(f"\n--- Results ---")
    print(f"Room Dimensions (WxLxH): {W:.2f}m x {L:.2f}m x {H:.2f}m")
    print(f"Floor Grid Resolution: {FLOOR_GRID_RES:.3f}m ({X.shape[1]}x{X.shape[0]} points)")
    print(f"Number of Main COBs: {len(cob_positions_only)}")
    strip_emitters_count = total_lights - len(cob_positions_only)
    print(f"Number of Strip Modules Placed: {strip_emitters_count}")
    print(f"Total Emitters Simulated: {total_lights}")
    print(f"Average PPFD: {mean_ppfd:.2f} µmol/m²/s")
    print(f"Std Deviation: {std_dev:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAD: {mad:.2f}")
    print(f"Min PPFD: {min_ppfd:.2f}")
    print(f"Max PPFD: {max_ppfd:.2f}")
    print(f"\n--- Uniformity ---")
    print(f"CV (%): {cv_percent:.2f}%")
    print(f"Min/Max Ratio: {min_max_ratio:.3f}")
    print(f"Min/Avg Ratio: {min_avg_ratio:.3f}")
    print(f"CU (%) (MAD-based): {cu_percent:.2f}%")
    print(f"DOU (%) (RMSE-based): {dou_percent:.2f}%")

    # --- CSV Output ---
    csv_filename = "ppfd_layer_data.csv"
    print(f"\nWriting layer-based PPFD data to {csv_filename}...")
    write_ppfd_to_csv(csv_filename, floor_ppfd, X, Y, cob_positions_only, W, L)
    print("CSV writing complete.")

    # --- Command Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Lighting Simulation Script with COBs and Strips")
    parser.add_argument('--no-plot', action='store_true', help='Disable the heatmap plot')
    parser.add_argument('--anno', type=int, default=15, help='Annotation density step for plot (0 disables)')
    args = parser.parse_args()

    # --- Heatmap Plot (Optional) ---
    if not args.no_plot:
        print("\nGenerating heatmap plot...")
        plot_heatmap(floor_ppfd, X, Y, cob_positions_only, ordered_strip_vertices, W, L, annotation_step=args.anno)
        print("Plot window opened. Close plot window to exit.")
        plt.show()

# --- Execution Guard (Remains the same) ---
if __name__ == "__main__":
    # IES loading happens globally before main
    if STRIP_IES_ANGLES_DEG is None or COB_IES_ANGLES_DEG is None: # Check both
         print("Error: One or both IES data files failed to load. Cannot run main().")
    else:
         main()
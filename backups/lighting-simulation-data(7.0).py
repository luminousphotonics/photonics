#!/usr/bin/env python3
import csv
import math
import numpy as np
from numba import njit, prange
from functools import lru_cache
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import argparse
import os  # Import os for path joining

# ------------------------------------
# 1) Basic Config & Reflectances
# ------------------------------------
REFL_WALL = 0.8
REFL_CEIL = 0.8
REFL_FLOOR = 0.1

LUMINOUS_EFFICACY = 182.0  # lumens/W (Used if SPD conversion is needed)
# SPD_FILE = "/Users/austinrouse/photonics/backups/spd_data.csv" # Keep if needed for PPFD
SPD_FILE = "/Users/austinrouse/photonics/backups/spd_data.csv" # Example relative path

MAX_RADIOSITY_BOUNCES = 10
RADIOSITY_CONVERGENCE_THRESHOLD = 1e-3

WALL_SUBDIVS_X = 10
WALL_SUBDIVS_Y = 5
CEIL_SUBDIVS_X = 10
CEIL_SUBDIVS_Y = 10

FLOOR_GRID_RES = 0.08  # m
FIXED_NUM_LAYERS = 19
MC_SAMPLES = 16

# --- Epsilon for numerical stability ---
EPSILON = 1e-9

# --- Default IES File ---
# Will be overridden by command-line arg if provided
DEFAULT_IES_FILE = "/Users/austinrouse/photonics/backups/cob_corrected.ies"

# ------------------------------------
# 2) IES File Parser (NEW)
# ------------------------------------
class IESParser:
    """Parses IES LM-63 Photometric Data Files."""
    def __init__(self, filename):
        self.filename = filename
        self.keywords = {}
        self.tilt_info = None
        self.num_lamps = 1
        self.lumens_per_lamp = 0.0
        self.multiplier = 1.0
        self.num_v_angles = 0
        self.num_h_angles = 0
        self.photometric_type = 1  # 1: Type C, 2: Type B, 3: Type A
        self.units_type = 1       # 1: Feet, 2: Meters
        self.width = 0.0
        self.length = 0.0
        self.height = 0.0
        self.ballast_factor = 1.0
        self.future_use = 1.0 # Often Ballast-Lamp Photometric Factor
        self.input_watts = 0.0
        self.v_angles = np.array([], dtype=np.float64)
        self.h_angles = np.array([], dtype=np.float64)
        self.candelas = np.array([], dtype=np.float64) # 2D: (v_angle_idx, h_angle_idx)
        self.is_symmetric = False # Assume Type C initially

        self._parse()

    def _parse(self):
        """Reads and parses the IES file."""
        try:
            with open(self.filename, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"[ERROR] IES file not found: {self.filename}")
            raise
        except Exception as e:
            print(f"[ERROR] Could not read IES file: {e}")
            raise

        line_idx = 0
        # Read format line (optional)
        if lines[line_idx].strip().upper().startswith("IESNA"):
            self.keywords["IESNA_FORMAT"] = lines[line_idx].strip()
            line_idx += 1

        # Read keywords
        while lines[line_idx].strip().startswith('['):
            line = lines[line_idx].strip()
            end_key = line.find(']')
            if end_key != -1:
                key = line[1:end_key].upper()
                value = line[end_key+1:].strip()
                self.keywords[key] = value
            line_idx += 1

        # Read TILT line
        if not lines[line_idx].strip().upper().startswith("TILT="):
             print(f"[Warning] Expected TILT= line, found: {lines[line_idx].strip()}")
             # Attempt to proceed, assuming TILT=NONE equivalent if line looks like parameters
        else:
            self.tilt_info = lines[line_idx].strip()
            line_idx += 1
            # TODO: Handle TILT data if not NONE

        # Read primary parameter line
        try:
            params1 = list(map(float, lines[line_idx].split()))
            self.num_lamps = int(params1[0])
            self.lumens_per_lamp = params1[1]
            self.multiplier = params1[2]
            self.num_v_angles = int(params1[3])
            self.num_h_angles = int(params1[4])
            self.photometric_type = int(params1[5])
            self.units_type = int(params1[6])
            self.width = params1[7]
            self.length = params1[8]
            self.height = params1[9]
            line_idx += 1
        except Exception as e:
            print(f"[ERROR] Could not parse primary parameter line: {lines[line_idx].strip()} - {e}")
            raise

        # Read secondary parameter line
        try:
            params2 = list(map(float, lines[line_idx].split()))
            self.ballast_factor = params2[0]
            # LM-63-1995 and earlier had 1 value here, 2002 has 2
            if len(params2) > 1:
                self.future_use = params2[1] # Often Ballast-Lamp Photometric Factor
                if len(params2) > 2:
                    self.input_watts = params2[2]
                else: # Try reading watts from next line (older format extension)
                    line_idx += 1
                    try:
                         self.input_watts = float(lines[line_idx].strip())
                    except ValueError:
                         print(f"[Warning] Could not parse input watts on separate line: {lines[line_idx].strip()}")
                         line_idx -= 1 # Go back if it wasn't watts
            else: # Might be older format, try reading watts from next line
                line_idx += 1
                try:
                    self.input_watts = float(lines[line_idx].strip())
                except ValueError:
                    print(f"[Warning] Could not parse input watts on separate line: {lines[line_idx].strip()}")
                    line_idx -= 1 # Go back if it wasn't watts

            line_idx += 1
        except Exception as e:
            print(f"[ERROR] Could not parse secondary parameter line: {lines[line_idx].strip()} - {e}")
            # Don't raise, might be missing in older files, defaults are set.


        # Read Vertical Angles
        v_angles_list = []
        while len(v_angles_list) < self.num_v_angles:
            try:
                v_angles_list.extend(list(map(float, lines[line_idx].split())))
                line_idx += 1
            except Exception as e:
                print(f"[ERROR] Could not parse vertical angles: {lines[line_idx].strip()} - {e}")
                raise
        self.v_angles = np.array(v_angles_list[:self.num_v_angles], dtype=np.float64)

        # Read Horizontal Angles
        h_angles_list = []
        while len(h_angles_list) < self.num_h_angles:
            try:
                h_angles_list.extend(list(map(float, lines[line_idx].split())))
                line_idx += 1
            except Exception as e:
                print(f"[ERROR] Could not parse horizontal angles: {lines[line_idx].strip()} - {e}")
                raise
        self.h_angles = np.array(h_angles_list[:self.num_h_angles], dtype=np.float64)

        # Read Candela Values
        candela_list = []
        expected_candelas = self.num_v_angles * self.num_h_angles
        while len(candela_list) < expected_candelas and line_idx < len(lines):
            try:
                candela_list.extend(list(map(float, lines[line_idx].split())))
                line_idx += 1
            except Exception as e:
                print(f"[ERROR] Could not parse candela values: {lines[line_idx].strip()} - {e}")
                # Decide whether to raise or try to continue
                # For now, let's raise if parsing fails mid-candela list
                if len(candela_list) > 0:
                     raise
                else: # Allow skipping potentially problematic lines before candelas start
                     line_idx += 1

        if len(candela_list) < expected_candelas:
            raise ValueError(f"Not enough candela values. Expected {expected_candelas}, got {len(candela_list)}")
        elif len(candela_list) > expected_candelas:
            print(f"[Warning] Truncating extra candela values: expected {expected_candelas}, got {len(candela_list)}")
            candela_list = candela_list[:expected_candelas]


        # Reshape candelas: LM-63 stores them H-angle major (all V for H1, then all V for H2...)
        # We want V-angle major for easier interpolation (row=V, col=H)
        # So reshape as (num_h, num_v) then transpose to (num_v, num_h)
        try:
            self.candelas = np.array(candela_list, dtype=np.float64).reshape((self.num_h_angles, self.num_v_angles)).T
        except ValueError as e:
            print(f"[ERROR] Could not reshape candela values. Expected shape ({self.num_h_angles}, {self.num_v_angles}). Data length: {len(candela_list)} - {e}")
            raise

        # Check for axial symmetry (Type C)
        if self.photometric_type == 1:
            if self.num_h_angles == 1 or np.allclose(self.h_angles, 0.0) or np.allclose(self.h_angles, [0.0, 360.0]):
                 self.is_symmetric = True
                 print("[INFO] IES Data appears axially symmetric (single horizontal angle).")
            elif np.allclose(self.candelas[:, 0], self.candelas[:, -1]) and np.allclose(self.h_angles[0], 0.0) and np.allclose(self.h_angles[-1], 360.0):
                 # Handle 0/360 redundancy if candela values match
                 self.h_angles = self.h_angles[:-1] # Remove 360
                 self.candelas = self.candelas[:, :-1] # Remove last column
                 self.num_h_angles -= 1
                 print("[INFO] IES Data: Removed redundant 360 deg horizontal angle.")


        print(f"[INFO] Parsed IES file: {self.filename}")
        print(f"[INFO]   Lumens: {self.lumens_per_lamp}, Multiplier: {self.multiplier}")
        print(f"[INFO]   Angles (Vert/Horiz): {self.num_v_angles} / {self.num_h_angles}")
        print(f"[INFO]   Photometric Type: {self.photometric_type}")


# ------------------------------------
# 3) Compute SPD-based µmol/J Factor
# ------------------------------------
def compute_conversion_factor(spd_file):
    """ Computes PPFD conversion factor from SPD data. """
    try:
        raw_spd = np.loadtxt(spd_file, delimiter=' ', skiprows=1)
        unique_wl, indices, counts = np.unique(raw_spd[:, 0], return_inverse=True, return_counts=True)
        counts_nonzero = np.maximum(counts, 1)
        avg_intens = np.bincount(indices, weights=raw_spd[:, 1]) / counts_nonzero
        spd = np.column_stack((unique_wl, avg_intens))
    except FileNotFoundError:
        print(f"[Warning] SPD file not found: {spd_file}. Using fallback conversion factor.")
        return 0.0138 # Fallback value
    except Exception as e:
        print(f"[Warning] Error loading SPD: {e}. Using fallback conversion factor.")
        return 0.0138 # Fallback value

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
# CONVERSION_FACTOR computation deferred until main() after args parsing

# ------------------------------------
# 4) Luminous Intensity from IES Data (Numba JIT version)
# ------------------------------------
@njit
def luminous_intensity_njit(theta_deg, phi_deg, target_lumens,
                            ies_lumens, ies_multiplier,
                            v_angles, h_angles, candelas):
    """
    Calculates luminous intensity (candela) for a given direction using IES data.
    Performs bilinear interpolation for Type C data.
    Scales result based on target_lumens vs ies_lumens.

    Args:
        theta_deg (float): Vertical angle from Nadir (0 = straight down).
        phi_deg (float): Horizontal angle (0 along +X axis, counter-clockwise).
        target_lumens (float): The desired total lumens for this specific light source instance.
        ies_lumens (float): The total lumens specified in the IES file (lumens_per_lamp * num_lamps).
        ies_multiplier (float): Candela multiplier from the IES file.
        v_angles (np.array): Sorted vertical angles from IES [degrees].
        h_angles (np.array): Sorted horizontal angles from IES [degrees].
        candelas (np.array): 2D array of candela values (rows=v_angle, cols=h_angle).

    Returns:
        float: Luminous intensity (candela) in the specified direction.
    """
    # --- Scaling Factor ---
    # Avoid division by zero if ies_lumens is 0
    lumen_scale_factor = target_lumens / ies_lumens if ies_lumens > EPSILON else 1.0

    # --- Angle Normalization and Edge Cases ---
    # Clamp theta to the range of vertical angles
    if theta_deg < v_angles[0]:
        theta_deg = v_angles[0]
    elif theta_deg > v_angles[-1]:
        theta_deg = v_angles[-1]


    # Normalize phi to be within [0, 360)
    phi_deg = phi_deg % 360.0

    # --- Vertical Interpolation ---
    # Find indices bounding theta_deg
    v_idx = np.searchsorted(v_angles, theta_deg, side='right') - 1
    # Handle exact match at the beginning or end
    if v_idx < 0: v_idx = 0
    if v_idx >= len(v_angles) - 1:
        v_idx = len(v_angles) - 2 # Use second to last index for interpolation range
        if theta_deg >= v_angles[-1]: # If exactly last angle or beyond
             v_idx = len(v_angles) - 1 # Set to exact last index
             v_weight = 0.0 # Only use last angle data
        else:
             v_weight = (theta_deg - v_angles[v_idx]) / (v_angles[v_idx + 1] - v_angles[v_idx] + EPSILON)
    elif np.abs(theta_deg - v_angles[v_idx]) < EPSILON:
         v_weight = 0.0 # Exactly on lower angle
    elif np.abs(theta_deg - v_angles[v_idx+1]) < EPSILON:
         v_idx += 1
         v_weight = 0.0 # Exactly on upper angle (now indexed by v_idx)
    else:
        # Ensure denominator is not zero
        v_denom = v_angles[v_idx + 1] - v_angles[v_idx]
        v_weight = (theta_deg - v_angles[v_idx]) / (v_denom + EPSILON) # Weight for v_idx+1

    if v_weight < 0.0:
        v_weight = 0.0
    elif v_weight > 1.0:
        v_weight = 1.0



    # --- Horizontal Interpolation ---
    num_h = len(h_angles)
    if num_h <= 1:
        # Axially symmetric or single plane
        h_idx = 0
        h_weight = 0.0 # No horizontal interpolation needed
    else:
        # Find indices bounding phi_deg, handle wrap-around (0 to 360)
        if phi_deg >= h_angles[-1]: # Check wrap-around case first
            h_idx = num_h - 1 # Index of last angle (e.g., 337.5)
            h_angle_upper = h_angles[0] + 360.0 # Upper angle is first angle + 360 (e.g., 0+360)
            h_denom = h_angle_upper - h_angles[h_idx]
            h_weight = (phi_deg - h_angles[h_idx]) / (h_denom + EPSILON) # Weight for h_angles[0]
            h_idx_upper = 0 # The upper index wraps around to 0
        else:
            h_idx = np.searchsorted(h_angles, phi_deg, side='right') - 1
            if h_idx < 0: h_idx = 0 # Should not happen with phi % 360, but safety

            if h_idx >= num_h - 1: # Should only happen if phi == h_angles[-1]
                h_idx = num_h - 1
                h_weight = 0.0 # Exactly on last angle
                h_idx_upper = h_idx # Doesn't matter
            elif np.abs(phi_deg - h_angles[h_idx]) < EPSILON:
                 h_weight = 0.0 # Exactly on lower angle
                 h_idx_upper = h_idx + 1
            elif np.abs(phi_deg - h_angles[h_idx+1]) < EPSILON:
                 h_idx += 1
                 h_weight = 0.0 # Exactly on upper angle (now indexed by h_idx)
                 h_idx_upper = h_idx # Doesn't matter
            else:
                 h_denom = h_angles[h_idx + 1] - h_angles[h_idx]
                 h_weight = (phi_deg - h_angles[h_idx]) / (h_denom + EPSILON) # Weight for h_idx+1
                 h_idx_upper = h_idx + 1

        if h_weight < 0.0:
            h_weight = 0.0
        elif h_weight > 1.0:
            h_weight = 1.0

    # --- Bilinear Interpolation ---
    if num_h <= 1: # Symmetric case
        cd_v1 = candelas[v_idx, 0]
        cd_v2 = candelas[min(v_idx + 1, len(v_angles) - 1), 0]
        candela_interp = cd_v1 * (1.0 - v_weight) + cd_v2 * v_weight
    else: # Type C bilinear
        # Get the 4 candela values
        c11 = candelas[v_idx, h_idx]                  # Bottom-left (v_low, h_low)
        c12 = candelas[v_idx, h_idx_upper]            # Bottom-right (v_low, h_high)
        c21 = candelas[min(v_idx + 1, len(v_angles) - 1), h_idx]       # Top-left (v_high, h_low)
        c22 = candelas[min(v_idx + 1, len(v_angles) - 1), h_idx_upper] # Top-right (v_high, h_high)

        # Interpolate horizontally along bottom edge (v_low)
        cd_h_bottom = c11 * (1.0 - h_weight) + c12 * h_weight
        # Interpolate horizontally along top edge (v_high)
        cd_h_top = c21 * (1.0 - h_weight) + c22 * h_weight

        # Interpolate vertically between the two horizontal results
        candela_interp = cd_h_bottom * (1.0 - v_weight) + cd_h_top * v_weight

    # Apply IES multiplier and lumen scaling
    final_candela = candela_interp * ies_multiplier * lumen_scale_factor

    return final_candela


# ------------------------------------
# 5) Geometry Building
# ------------------------------------
def prepare_geometry(W, L, H):
    cob_positions = build_cob_positions(W, L, H)
    X, Y = build_floor_grid(W, L)
    patches = build_patches(W, L, H)
    return (cob_positions, X, Y, patches)

# build_cob_positions remains the same
def build_cob_positions(W, L, H):
    n = FIXED_NUM_LAYERS - 1
    
    positions = []
    positions.append((0, 0, H, 0))  # Central COB at layer 0

    for i in range(1, n + 1):  # Layers 1 to n
        for x in range(-i, i + 1):
            y_abs = i - abs(x)
            if y_abs == 0:
                if x != 0:
                    positions.append((x, 0, H, i))
            else:
                positions.append((x, y_abs, H, i))
                positions.append((x, -y_abs, H, i))

    # Transform grid (rotation + scaling + translation)
    theta = math.radians(45)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    centerX = W / 2
    centerY = L / 2

    # Margin from walls
    margin_ratio = 0.02  # 2% inset from each wall
    usable_W = W * (1 - 2 * margin_ratio)
    usable_L = L * (1 - 2 * margin_ratio)

    cos_45 = math.cos(math.radians(45))
    scale_x = (usable_W / 2) / (n * cos_45) if n > 0 else usable_W / 2
    scale_y = (usable_L / 2) / (n * cos_45) if n > 0 else usable_L / 2

    transformed = []
    for (xx, yy, hh, layer) in positions:
        rx = xx * cos_t - yy * sin_t
        ry = xx * sin_t + yy * cos_t
        px = centerX + rx * scale_x
        py = centerY + ry * scale_y
        pz = H * 0.95  # 95% ceiling height
        transformed.append((px, py, pz, layer))

    return np.array(transformed, dtype=np.float64)



def pack_luminous_flux_dynamic(params, cob_positions):
    """Assigns target lumens from params based on the layer index of each COB."""
    led_intensities = []
    if len(params) != FIXED_NUM_LAYERS:
         raise ValueError(f"Length of params ({len(params)}) does not match FIXED_NUM_LAYERS ({FIXED_NUM_LAYERS})")
    for pos in cob_positions:
        layer = int(pos[3])
        if 0 <= layer < len(params):
            intensity = params[layer]
            led_intensities.append(intensity)
        else:
             print(f"[Warning] COB position {pos[:3]} has invalid layer index {layer}. Assigning 0 lumens.")
             led_intensities.append(0.0)

    return np.array(led_intensities, dtype=np.float64)

# build_floor_grid and build_patches updated for symmetry
@lru_cache(maxsize=32)
def cached_build_floor_grid(W: float, L: float):
    dx = FLOOR_GRID_RES
    num_x = int(W / dx)
    num_y = int(L / dx)

    half_W = W / 2
    half_L = L / 2

    # Generate centered coordinates in range (-W/2, W/2), then shift back to (0, W)
    x_coords = np.linspace(-half_W + dx / 2, half_W - dx / 2, num_x)
    y_coords = np.linspace(-half_L + dx / 2, half_L - dx / 2, num_y)

    X, Y = np.meshgrid(x_coords, y_coords)

    # Shift grid to be centered in room coordinates
    X += half_W
    Y += half_L

    return X, Y

def build_floor_grid(W, L):
    return cached_build_floor_grid(W, L)


@lru_cache(maxsize=32)
def cached_build_patches(W: float, L: float, H: float):
    patch_centers = []
    patch_areas = []
    patch_normals = []
    patch_refl = []
    # --- Floor Patch (single large patch for radiosity, not grid) ---
    # We calculate illumination *on* the fine floor grid later.
    # This floor patch participates in inter-reflection.
    patch_centers.append((W/2, L/2, 0.0))
    patch_areas.append(W * L)
    patch_normals.append((0.0, 0.0, 1.0)) # Normal points UP into the room
    patch_refl.append(REFL_FLOOR)

    # --- Ceiling Patches ---
    xs_ceiling = np.linspace(0, W, CEIL_SUBDIVS_X + 1)
    ys_ceiling = np.linspace(0, L, CEIL_SUBDIVS_Y + 1)
    for i in range(CEIL_SUBDIVS_X):
        for j in range(CEIL_SUBDIVS_Y):
            cx = (xs_ceiling[i] + xs_ceiling[i+1]) / 2
            cy = (ys_ceiling[j] + ys_ceiling[j+1]) / 2
            area = (xs_ceiling[i+1] - xs_ceiling[i]) * (ys_ceiling[j+1] - ys_ceiling[j])
            # Position slightly below ceiling H for robustness
            patch_centers.append((cx, cy, H - 0.01)) # Adjusted position slightly below H
            patch_areas.append(area)
            patch_normals.append((0.0, 0.0, -1.0)) # Normal points DOWN into the room
            patch_refl.append(REFL_CEIL)

    # --- Wall Patches (4 Walls) ---
    xs_wall = np.linspace(0, W, WALL_SUBDIVS_X + 1)
    zs_wall = np.linspace(0, H, WALL_SUBDIVS_Y + 1)
    # Wall at Y = 0 (Back Wall)
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cx = (xs_wall[i] + xs_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (xs_wall[i+1]-xs_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((cx, 0.0 + EPSILON, cz)) # Slightly offset from boundary
            patch_areas.append(area)
            patch_normals.append((0.0, 1.0, 0.0)) # Normal points INTO the room (+Y direction)
            patch_refl.append(REFL_WALL)
    # Wall at Y = L (Front Wall)
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cx = (xs_wall[i] + xs_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (xs_wall[i+1]-xs_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((cx, L - EPSILON, cz)) # Slightly offset
            patch_areas.append(area)
            patch_normals.append((0.0, -1.0, 0.0)) # Normal points INTO the room (-Y direction)
            patch_refl.append(REFL_WALL)

    ys_wall = np.linspace(0, L, WALL_SUBDIVS_X + 1) # Re-use WALL_SUBDIVS_X for consistency
    # Wall at X = 0 (Left Wall)
    for i in range(WALL_SUBDIVS_X): # Loop over Y subdivisions for this wall
        for j in range(WALL_SUBDIVS_Y): # Loop over Z subdivisions
            cy = (ys_wall[i] + ys_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (ys_wall[i+1]-ys_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((0.0 + EPSILON, cy, cz)) # Slightly offset
            patch_areas.append(area)
            patch_normals.append((1.0, 0.0, 0.0)) # Normal points INTO the room (+X direction)
            patch_refl.append(REFL_WALL)
    # Wall at X = W (Right Wall)
    for i in range(WALL_SUBDIVS_X): # Loop over Y subdivisions
        for j in range(WALL_SUBDIVS_Y): # Loop over Z subdivisions
            cy = (ys_wall[i] + ys_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (ys_wall[i+1]-ys_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((W - EPSILON, cy, cz)) # Slightly offset
            patch_areas.append(area)
            patch_normals.append((-1.0, 0.0, 0.0)) # Normal points INTO the room (-X direction)
            patch_refl.append(REFL_WALL)

    return (np.array(patch_centers, dtype=np.float64),
            np.array(patch_areas, dtype=np.float64),
            np.array(patch_normals, dtype=np.float64),
            np.array(patch_refl, dtype=np.float64))


def build_patches(W, L, H):
    return cached_build_patches(W, L, H)

# ------------------------------------
# 6) The Numba-JIT Computations
# ------------------------------------
@njit(parallel=True)
def compute_direct_floor(cob_positions, target_lumens_per_cob,
                         ies_lumens, ies_multiplier, v_angles, h_angles, candelas,
                         X, Y):
    """Calculates direct illuminance (lux) on the floor grid points."""
    min_dist2 = (FLOOR_GRID_RES / 4.0) ** 2 # Smaller minimum distance
    rows, cols = X.shape
    out = np.zeros_like(X, dtype=np.float64)

    for r in prange(rows):
        for c in range(cols):
            fx = X[r, c]
            fy = Y[r, c]
            fz = 0.0 # Floor points are at z=0
            val = 0.0
            for k in range(cob_positions.shape[0]):
                lx = cob_positions[k, 0]
                ly = cob_positions[k, 1]
                lz = cob_positions[k, 2]
                target_lumens_k = target_lumens_per_cob[k]

                if target_lumens_k <= EPSILON: # Skip COB if it's off
                    continue

                # Vector from light source (L) to floor point (F)
                dx = fx - lx
                dy = fy - ly
                dz = fz - lz # This will be negative

                d2 = dx*dx + dy*dy + dz*dz
                if d2 < min_dist2:
                    d2 = min_dist2
                dist = math.sqrt(d2)

                # --- Calculate angles for IES lookup ---
                # theta: Angle from Nadir (0 = straight down, +Z axis from light's perspective)
                # The vector L->F has components (dx, dy, dz).
                # Nadir vector is (0, 0, -1) in world coords.
                # Cos(theta) = (L->F vector) dot (Nadir vector) / (dist * |Nadir|)
                # Cos(theta) = (dx*0 + dy*0 + dz*(-1)) / (dist * 1) = -dz / dist
                cos_theta = -dz / dist
                # Clamp cos_theta to avoid math errors with acos
                cos_theta = max(0.0, min(1.0, cos_theta)) # Light only emits below horizon
                theta_deg = math.degrees(math.acos(cos_theta))

                # phi: Horizontal angle (0 along +X axis, counter-clockwise)
                # Project L->F vector onto XY plane (dx, dy)
                # atan2(y, x) gives angle from +X axis
                phi_deg = math.degrees(math.atan2(dy, dx))
                phi_deg = phi_deg % 360.0 # Ensure [0, 360) range

                # Get intensity from IES data
                I_theta_phi = luminous_intensity_njit(
                    theta_deg, phi_deg, target_lumens_k,
                    ies_lumens, ies_multiplier, v_angles, h_angles, candelas
                )

                # Illuminance = Intensity * cos(incidence_angle) / distance^2
                # Incidence angle for floor is the same as theta (angle with normal [0,0,1])
                cos_in_floor = cos_theta # Since floor normal is (0,0,1)

                if cos_in_floor <= EPSILON: # Light must hit from above
                    continue

                E_local = (I_theta_phi / d2) * cos_in_floor
                val += E_local

            out[r, c] = val
    return out

@njit
def compute_patch_direct(cob_positions, target_lumens_per_cob,
                         ies_lumens, ies_multiplier, v_angles, h_angles, candelas,
                         patch_centers, patch_normals, patch_areas):
    """Calculates direct illuminance (lux) on the center of each radiosity patch."""
    min_dist2 = (FLOOR_GRID_RES / 4.0) ** 2
    Np = patch_centers.shape[0]
    out = np.zeros(Np, dtype=np.float64)

    for ip in range(Np):
        pc = patch_centers[ip] # Patch center (x, y, z)
        n_vec = patch_normals[ip] # Patch normal vector (nx, ny, nz)
        norm_n = math.sqrt(n_vec[0]*n_vec[0] + n_vec[1]*n_vec[1] + n_vec[2]*n_vec[2])
        if norm_n < EPSILON: norm_n = 1.0 # Avoid division by zero

        accum = 0.0
        for k in range(cob_positions.shape[0]):
            lx = cob_positions[k, 0]
            ly = cob_positions[k, 1]
            lz = cob_positions[k, 2]
            target_lumens_k = target_lumens_per_cob[k]

            if target_lumens_k <= EPSILON:
                continue

            # Vector from light source (L) to patch center (P)
            dx = pc[0] - lx
            dy = pc[1] - ly
            dz = pc[2] - lz

            d2 = dx*dx + dy*dy + dz*dz
            if d2 < min_dist2:
                d2 = min_dist2
            dist = math.sqrt(d2)

            # --- Calculate angles for IES lookup ---
            # theta (angle from Nadir)
            cos_theta = -dz / dist # Same as for floor
            cos_theta = max(0.0, min(1.0, cos_theta))
            theta_deg = math.degrees(math.acos(cos_theta))

            # phi (horizontal angle)
            phi_deg = math.degrees(math.atan2(dy, dx))
            phi_deg = phi_deg % 360.0

            # Get intensity from IES data
            I_theta_phi = luminous_intensity_njit(
                theta_deg, phi_deg, target_lumens_k,
                ies_lumens, ies_multiplier, v_angles, h_angles, candelas
            )

            # Cosine of angle between L->P vector and patch normal N
            # dot_patch = (P - L) dot N = dx*nx + dy*ny + dz*nz
            dot_patch = dx * n_vec[0] + dy * n_vec[1] + dz * n_vec[2]
            cos_in_patch = dot_patch / (dist * norm_n)

            # Ensure light hits the front face of the patch
            if cos_in_patch <= EPSILON:
                continue

            # Illuminance = Intensity * cos(incidence_angle) / distance^2
            E_local = (I_theta_phi / d2) * cos_in_patch
            accum += E_local

        out[ip] = accum
    return out


# iterative_radiosity_loop remains the same
@njit
def iterative_radiosity_loop(patch_centers, patch_normals, patch_direct, patch_areas, patch_refl,
                             max_bounces, convergence_threshold):
    Np = patch_direct.shape[0]
    patch_rad = patch_direct.copy() # Initial radiosity = direct illuminance
    patch_exitance = np.zeros(Np, dtype=np.float64) # Exitance = Radiosity * Reflectance
    new_indirect_illuminance = np.zeros(Np, dtype=np.float64)
    total_flux_in = patch_direct * patch_areas # Flux initially received directly

    print("Radiosity Bounce 0: Initial Direct Illuminance calculated.")

    for bounce in range(1, max_bounces + 1):
        # 1. Calculate exitance (flux leaving each patch) from previous iteration's radiosity
        for j in range(Np):
            patch_exitance[j] = patch_rad[j] * patch_refl[j] # Flux density (W/m^2 or lm/m^2) leaving

        # 2. Calculate the indirect illuminance received by each patch (i) from all other patches (j)
        new_indirect_illuminance.fill(0.0) # Reset for this bounce
        total_flux_transfered = 0.0

        for j in range(Np): # Source patch
            if patch_refl[j] <= EPSILON or patch_exitance[j] <= EPSILON:
                continue # No light leaves this patch

            # Total flux leaving patch j = Exit_j * Area_j
            outgoing_flux_j = patch_exitance[j] * patch_areas[j]

            pj = patch_centers[j]
            nj = patch_normals[j]
            norm_nj = math.sqrt(nj[0]**2 + nj[1]**2 + nj[2]**2)
            if norm_nj < EPSILON: continue # Invalid normal

            for i in range(Np): # Receiving patch
                if i == j:
                    continue # Patch doesn't illuminate itself directly in this step

                pi = patch_centers[i]
                ni = patch_normals[i]
                norm_ni = math.sqrt(ni[0]**2 + ni[1]**2 + ni[2]**2)
                if norm_ni < EPSILON: continue # Invalid normal

                # Vector from center of j to center of i
                vij = pi - pj
                dist2 = vij[0]**2 + vij[1]**2 + vij[2]**2

                if dist2 < EPSILON: # Patches too close or coincident
                    continue
                dist = math.sqrt(dist2)

                # Cosine of angle between normal_j and vector_ij
                cos_theta_j = np.dot(nj, vij) / (norm_nj * dist)
                # Cosine of angle between normal_i and vector_ji (-vij)
                cos_theta_i = np.dot(ni, -vij) / (norm_ni * dist)

                # Check if patches face each other and are visible
                if cos_theta_j <= EPSILON or cos_theta_i <= EPSILON:
                    continue # Not facing or backface hit

                # Form Factor F_ji = (cos_theta_j * cos_theta_i * Area_i) / (pi * dist^2)
                # We need flux density on patch i, so use F_ji / Area_i
                # Differential Form Factor dF_ji = (cos_theta_j * cos_theta_i) / (pi * dist^2) dA_i
                # Flux arriving at i from j = OutgoingFlux_j * F_ji
                # Irradiance at i from j = OutgoingFlux_j * dF_ji / dA_i
                # Irradiance at i from j = (Exit_j * Area_j) * (cos_theta_j * cos_theta_i) / (pi * dist^2)

                form_factor_term = (cos_theta_j * cos_theta_i) / (math.pi * dist2)
                flux_density_received_i = patch_exitance[j] * patch_areas[j] * form_factor_term
                # This is the irradiance on patch i due to patch j

                new_indirect_illuminance[i] += flux_density_received_i
                total_flux_transfered += flux_density_received_i * patch_areas[i]


        # 3. Update Radiosity for each patch
        max_rel_change = 0.0
        new_patch_rad = np.empty_like(patch_rad)
        for i in range(Np):
            # New Radiosity = Direct Illuminance + Indirect Illuminance received this bounce
            # Note: Radiosity = Emittance + Exitance = Emittance + Reflectance * Total_Illuminance
            # Here, Total_Illuminance = Direct + Indirect
            # Radiosity = Direct + Indirect_Received (since Emittance is 0 for non-emitters after bounce 0)
            # This seems slightly off. Let's redefine:
            # Radiosity(B) = Illuminance_Direct + Illuminance_Indirect(B)
            # Where Illuminance_Indirect(B) is from Radiosity(B-1) * Reflectance of other patches.

            # Correct approach:
            # B_i = E_i + rho_i * H_i
            # H_i = H_direct_i + H_indirect_i
            # H_indirect_i = sum( B_j * F_ij * Area_j / Area_i ) ??? No, simpler.
            # H_indirect_i = sum( Exitance_j * Area_j * dF_ji / dA_i ) <- This is what we calculated in new_indirect_illuminance[i]

            new_rad_i = patch_direct[i] + new_indirect_illuminance[i]

            # Calculate relative change for convergence check
            change = abs(new_rad_i - patch_rad[i])
            denom = abs(patch_rad[i]) + EPSILON
            rel_change = change / denom
            if rel_change > max_rel_change:
                max_rel_change = rel_change

            new_patch_rad[i] = new_rad_i

        patch_rad = new_patch_rad.copy() # Update radiosity for next bounce

        print("Radiosity Bounce", bounce, "Max Rel Change =", max_rel_change, "Total Indirect Flux Transfered =", total_flux_transfered)

        # 4. Check convergence
        if max_rel_change < convergence_threshold:
            print("Radiosity converged after", bounce, "bounces.")
            break
    else: # Loop finished without break
        print("[Warning] Radiosity did not converge after", max_bounces, "bounces. Max Rel Change =", max_rel_change)

    # Final radiosity includes direct + all indirect contributions
    return patch_rad


# compute_reflection_on_floor (using Joblib Parallel) needs updating
# It should calculate illuminance on the floor grid from the final patch exitances
def compute_reflection_on_floor(X, Y, patch_centers, patch_normals, patch_areas, patch_rad, patch_refl,
                                mc_samples=MC_SAMPLES):
    """Calculates indirect illuminance on the floor grid from radiosity patches."""
    rows, cols = X.shape
    # Calculate final exitance (lm/m^2) for each patch
    patch_exitance = patch_rad * patch_refl

    # Use joblib for parallel processing of rows
    results = Parallel(n_jobs=-1)(delayed(compute_row_reflection_njit)(
        r, X, Y, patch_centers, patch_normals, patch_areas, patch_exitance, mc_samples
    ) for r in range(rows))

    # Combine results into the output array
    out = np.zeros((rows, cols), dtype=np.float64)
    for r, row_vals in enumerate(results):
        if row_vals is not None: # Handle potential errors in parallel task
             out[r, :] = row_vals
        else:
             print(f"[Warning] Row {r} computation failed in parallel reflection calculation.")

    return out

# Numba JIT function for computing reflections on a single row of the floor grid
@njit
def compute_row_reflection_njit(r, X, Y, patch_centers, patch_normals, patch_areas, patch_exitance, mc_samples):
    """Numba JIT function to compute indirect illuminance for one row of the floor grid."""
    cols = X.shape[1]
    row_vals = np.zeros(cols, dtype=np.float64)

    for c in range(cols):
        fx = X[r, c]
        fy = Y[r, c]
        fz = 0.0 # Floor point Z coordinate
        floor_normal = np.array([0.0, 0.0, 1.0]) # Normal vector of the floor

        accum_indirect_E = 0.0 # Accumulator for indirect illuminance at (fx, fy)

        # Sum contribution from each patch (p) to the floor point (f)
        for p_idx in range(patch_centers.shape[0]):
            # Skip floor patch itself and patches with no exitance
            if patch_normals[p_idx, 2] > 0.99 or patch_exitance[p_idx] <= EPSILON: # Skip floor patches (normal ~ [0,0,1])
                 continue

            pc = patch_centers[p_idx]
            n_patch = patch_normals[p_idx]
            norm_n_patch = math.sqrt(n_patch[0]**2 + n_patch[1]**2 + n_patch[2]**2)
            if norm_n_patch < EPSILON: continue

            area_p = patch_areas[p_idx]
            exitance_p = patch_exitance[p_idx] # Flux density leaving patch p (lm/m^2)

            # --- Monte Carlo Sampling over the patch area ---
            # Define patch local coordinate system (approxiate for square/rect)
            # Heuristic to find a reasonable tangent vector
            if abs(n_patch[2]) < 0.9: # If normal is not mostly vertical
                t1_vec = np.array([-n_patch[1], n_patch[0], 0.0]) # Perpendicular in XY plane
            else: # If normal is mostly vertical, use X-axis as tangent
                t1_vec = np.array([1.0, 0.0, 0.0])

            norm_t1 = np.linalg.norm(t1_vec)
            if norm_t1 > EPSILON:
                t1_vec /= norm_t1
            else: # Fallback if normal was purely Z and cross product failed
                 t1_vec = np.array([1.0, 0.0, 0.0])

            t2_vec = np.cross(n_patch, t1_vec)
            norm_t2 = np.linalg.norm(t2_vec)
            if norm_t2 > EPSILON:
                 t2_vec /= norm_t2
            else: # Fallback
                 t2_vec = np.array([0.0, 1.0, 0.0])


            # Approximate half-side length assuming square patch for sampling spread
            half_side = math.sqrt(area_p) / 2.0

            sample_sum_ff = 0.0
            num_valid_samples = 0

            for _ in range(mc_samples):
                # Generate random offsets within the patch plane
                offset1 = np.random.uniform(-half_side, half_side)
                offset2 = np.random.uniform(-half_side, half_side)
                # Calculate sample point on the patch surface
                sample_point = pc + offset1 * t1_vec + offset2 * t2_vec

                # Vector from sample point (S) to floor point (F)
                v_sf = np.array([fx - sample_point[0], fy - sample_point[1], fz - sample_point[2]])
                dist_sq = v_sf[0]**2 + v_sf[1]**2 + v_sf[2]**2

                if dist_sq < EPSILON: continue
                dist = math.sqrt(dist_sq)

                # Cosine of angle between patch normal and S->F vector
                cos_theta_patch = np.dot(n_patch, v_sf) / (norm_n_patch * dist)
                # Cosine of angle between floor normal and F->S vector (-v_sf)
                cos_theta_floor = np.dot(floor_normal, -v_sf) / (1.0 * dist)

                # Check visibility and orientation
                if cos_theta_patch <= EPSILON or cos_theta_floor <= EPSILON:
                    continue # Sample point doesn't face floor point or vice versa

                # Differential Form Factor term (dF_pf / dA_f) = (cos_theta_patch * cos_theta_floor) / (pi * dist^2)
                ff_term = (cos_theta_patch * cos_theta_floor) / (math.pi * dist_sq)

                sample_sum_ff += ff_term
                num_valid_samples += 1

            # Average Form Factor Term over samples
            if num_valid_samples > 0:
                avg_ff_term = sample_sum_ff / num_valid_samples
            else:
                avg_ff_term = 0.0 # No valid samples connected patch and floor point

            # Contribution to illuminance at floor point f from patch p:
            # dE_f = Exitance_p * dF_pf = Exitance_p * (avg_ff_term * dA_p)
            # We want dE_f / dA_f which is Illuminance at f
            # Irradiance_f = Exitance_p * Area_p * avg_ff_term ??? No...
            # Let L_p = Exitance_p / pi (assuming Lambertian patch)
            # dE_f = L_p * cos_theta_patch * cos_theta_floor * dA_p / dist^2
            # dE_f = (Exitance_p / pi) * cos_theta_patch * cos_theta_floor * dA_p / dist^2
            # E_f = integral over patch area... MC approx:
            # E_f approx = (Exitance_p / pi) * (1/N) * sum[ (cos_p * cos_f / dist^2)_sample ] * Area_p
            # E_f approx = Exitance_p * Area_p * (1/N) * sum[ (cos_p * cos_f / (pi * dist^2))_sample ]
            # E_f approx = Exitance_p * Area_p * avg_ff_term

            # Calculate illuminance contribution from this patch
            # This uses the averaged geometric term calculated via MC
            illuminance_from_patch = exitance_p * area_p * avg_ff_term
            accum_indirect_E += illuminance_from_patch

        row_vals[c] = accum_indirect_E

    return row_vals


# ------------------------------------
# 7) Heatmap Plotting Function
# ------------------------------------
def plot_heatmap(floor_values_flat, X, Y, cob_positions, title="Floor Heatmap", units="units", annotation_step=2):
    fig, ax = plt.subplots(figsize=(9, 7))  # Slightly larger figure

    # Reconstruct grid dimensions
    num_x = len(np.unique(X))
    num_y = len(np.unique(Y))
    W = X.max()
    L = Y.max()

    # Reshape flat PPFD values and coordinates into 2D grids
    floor_values = floor_values_flat.reshape((num_y, num_x))
    X_grid = X.reshape((num_y, num_x))
    Y_grid = Y.reshape((num_y, num_x))

    # Plot heatmap
    extent = [0, W, 0, L]
    im = ax.imshow(floor_values, cmap='hot', interpolation='nearest',
                   origin='lower', extent=extent, aspect='equal')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label(f"{title} ({units})")

    # Add annotations
    if annotation_step > 0:
        for i in range(0, num_y, annotation_step):
            for j in range(0, num_x, annotation_step):
                val = floor_values[i, j]
                x = X_grid[i, j]
                y = Y_grid[i, j]
                ax.text(x, y, f"{val:.1f}", ha="center", va="center",
                        color="white", fontsize=6,
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.5))

    # Plot COB positions
    if cob_positions is not None and len(cob_positions) > 0:
        ax.scatter(cob_positions[:, 0], cob_positions[:, 1], marker='o',
                   color='cyan', edgecolors='black', s=40, label="COB positions", alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

# ------------------------------------
# 8) CSV Output Function
# ------------------------------------
# write_ppfd_to_csv remains largely the same, ensure W, L are passed correctly
def write_ppfd_to_csv(filename, floor_ppfd, X, Y, cob_positions, W, L):
    """
    Writes PPFD data to CSV, organized by distance-based rings (layers),
    using boundaries defined between COB layer radii.
    """
    if floor_ppfd is None or X is None or Y is None or cob_positions is None:
        print("[Error] Cannot write CSV, input data is missing.")
        return

    layer_data = {}
    for i in range(FIXED_NUM_LAYERS):
        layer_data[i] = []

    center_x, center_y = W / 2.0, L / 2.0
    cob_layers = cob_positions[:, 3].astype(int) # Ensure integer layers

    # Calculate max radius for each layer that actually has COBs
    layer_radii = np.zeros(FIXED_NUM_LAYERS)
    for i in range(FIXED_NUM_LAYERS):
        layer_cob_indices = np.where(cob_layers == i)[0]
        if len(layer_cob_indices) > 0:
            layer_cob_positions = cob_positions[layer_cob_indices, :2] # Get X, Y coords
            distances = np.sqrt((layer_cob_positions[:, 0] - center_x)**2 +
                                (layer_cob_positions[:, 1] - center_y)**2)
            if len(distances) > 0:
                layer_radii[i] = np.max(distances)
        # If layer has no COBs, radius remains 0 initially

    # Fill in radii for empty layers by interpolating or using neighbours
    # This ensures boundaries can be calculated even with empty layers
    last_good_radius = 0.0
    last_good_idx = -1
    for i in range(FIXED_NUM_LAYERS):
        if layer_radii[i] > EPSILON:
            # Interpolate between last good radius and current one for empty layers in between
            if last_good_idx != -1:
                 num_empty = i - last_good_idx - 1
                 if num_empty > 0:
                     step = (layer_radii[i] - last_good_radius) / (num_empty + 1)
                     for j in range(1, num_empty + 1):
                         layer_radii[last_good_idx + j] = last_good_radius + j * step

            last_good_radius = layer_radii[i]
            last_good_idx = i
        elif i == 0 and layer_radii[i] < EPSILON:
             # Special case: Layer 0 is empty but should have radius 0
             layer_radii[i] = 0.0
             last_good_radius = 0.0
             last_good_idx = 0


    # Extrapolate for trailing empty layers if any
    if last_good_idx < FIXED_NUM_LAYERS - 1:
        if last_good_idx >= 0: # Need at least one non-empty layer
             # Estimate radius step based on previous layers
             prev_step = 0.0
             if last_good_idx > 0:
                 prev_step = layer_radii[last_good_idx] - layer_radii[last_good_idx - 1]
             elif last_good_idx == 0: # Only layer 0 had COBs
                  # Guess a step based on room size? Or just extend linearly?
                  # Let's just add a small amount or use W/L? Risky.
                  # Use the radius of layer 0 itself as a step guess?
                  prev_step = layer_radii[last_good_idx] * 0.5 # Guess based on first layer radius
                  if prev_step < EPSILON: prev_step = min(W, L) / (2*FIXED_NUM_LAYERS) # Fallback step


             if prev_step <= EPSILON: # If step is still zero, use a fallback
                  prev_step = min(W,L) / (2*FIXED_NUM_LAYERS) # Fallback step based on room

             for i in range(last_good_idx + 1, FIXED_NUM_LAYERS):
                 layer_radii[i] = layer_radii[i-1] + prev_step
        else: # All layers were empty - this shouldn't happen with valid params
              print("[Warning] All COB layers appear empty. Cannot determine radii for CSV.")
              # Assign arbitrary increasing radii for structure?
              step = min(W,L) / (2*FIXED_NUM_LAYERS)
              for i in range(FIXED_NUM_LAYERS):
                  layer_radii[i] = i * step


    # Define sampling boundaries BETWEEN layer radii midpoints
    sampling_boundaries = [0.0] # Boundary 0 is at radius 0
    for i in range(FIXED_NUM_LAYERS - 1):
        midpoint = (layer_radii[i] + layer_radii[i+1]) / 2.0
        # Ensure boundary increases, handle cases where radii are equal
        if midpoint <= sampling_boundaries[-1] + EPSILON:
             # If midpoint is not larger, nudge it slightly beyond the previous boundary
             # Use the radius of the next layer if it's larger, otherwise add epsilon
             if layer_radii[i+1] > sampling_boundaries[-1] + EPSILON:
                 sampling_boundaries.append(layer_radii[i+1])
             else:
                 sampling_boundaries.append(sampling_boundaries[-1] + FLOOR_GRID_RES/2.0) # Add small step
        else:
            sampling_boundaries.append(midpoint)

    # Add a final outer boundary slightly larger than the last layer's radius
    # Use distance to corner as max possible radius
    max_room_radius = math.sqrt((W/2)**2 + (L/2)**2)
    final_boundary = max(layer_radii[-1] + (layer_radii[-1]-layer_radii[-2] if FIXED_NUM_LAYERS > 1 else layer_radii[-1]*0.1), # Estimate next step
                         sampling_boundaries[-1] + FLOOR_GRID_RES/2.0) # Ensure larger than last boundary
    sampling_boundaries.append(max(final_boundary, max_room_radius * 1.05)) # Extend slightly beyond corner

    if len(sampling_boundaries) != FIXED_NUM_LAYERS + 1:
        print(f"[Error] Incorrect number of sampling boundaries generated ({len(sampling_boundaries)}). Expected {FIXED_NUM_LAYERS + 1}.")
        # Attempt to fix or fallback? Fallback: linear boundaries
        print("[Error] Falling back to linear boundaries.")
        sampling_boundaries = np.linspace(0, max_room_radius, FIXED_NUM_LAYERS + 1).tolist()


    # --- Assign Floor Points to Layers based on Boundaries ---
    rows, cols = floor_ppfd.shape
    assigned_counts = np.zeros(FIXED_NUM_LAYERS, dtype=int)
    for r in range(rows):
        for c in range(cols):
            fx = X[r, c]
            fy = Y[r, c]
            dist_to_center = math.sqrt((fx - center_x)**2 + (fy - center_y)**2)

            assigned_layer = -1
            # Find which ring the point falls into: [ B_i <= dist < B_{i+1} )
            for i in range(FIXED_NUM_LAYERS):
                inner_b = sampling_boundaries[i]
                outer_b = sampling_boundaries[i+1]

                # Check if distance is within the bounds for layer 'i'
                # Use tolerance for the inner boundary, especially for layer 0 at the center
                if (dist_to_center >= inner_b - EPSILON) and (dist_to_center < outer_b - EPSILON):
                    assigned_layer = i
                    break

            # Handle points potentially falling exactly on the outermost boundary
            if assigned_layer == -1 and np.abs(dist_to_center - sampling_boundaries[-1]) < EPSILON:
                 assigned_layer = FIXED_NUM_LAYERS - 1 # Assign to the last layer

            if assigned_layer != -1:
                if not (0 <= assigned_layer < FIXED_NUM_LAYERS):
                     print(f"[Warning] Point ({fx:.2f},{fy:.2f}) assigned to invalid layer {assigned_layer}. Skipping.")
                     continue
                # Ensure the layer list exists
                if assigned_layer not in layer_data: layer_data[assigned_layer] = []
                layer_data[assigned_layer].append(floor_ppfd[r, c])
                assigned_counts[assigned_layer] += 1
            #else:
                 # This point is outside the last boundary - should ideally not happen if final_boundary is large enough
                 # print(f"[Warning] Point ({fx:.2f},{fy:.2f}) with distance {dist_to_center:.3f} outside all boundaries. Max boundary: {sampling_boundaries[-1]:.3f}. Skipping.")


    # --- Write to CSV ---
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Layer', 'PPFD']) # Header row
            for layer in range(FIXED_NUM_LAYERS):
                if layer in layer_data and layer_data[layer]: # Check layer exists and has data
                    ppfd_values = layer_data[layer]
                    # Sort values within the layer? Optional.
                    # ppfd_values.sort()
                    for ppfd_value in ppfd_values:
                        writer.writerow([layer, f"{ppfd_value:.4f}"]) # Write layer index and PPFD value
                else:
                    # Optionally write a placeholder if a layer has no points assigned
                    # writer.writerow([layer, "N/A"])
                    print(f"[Info] Layer {layer} has no assigned floor points or no data.")

        print(f"PPFD data successfully written to {filename}")
        # Print assignment summary
        #for i in range(FIXED_NUM_LAYERS):
        #     print(f"  Layer {i}: {assigned_counts[i]} floor samples assigned.")

    except IOError as e:
        print(f"[Error] Could not write to CSV file {filename}: {e}")
    except Exception as e:
        print(f"[Error] An unexpected error occurred during CSV writing: {e}")

# ------------------------------------
# 9) Putting It All Together
# ------------------------------------
def simulate_lighting(params, geo, ies_data):
    """Runs the full lighting simulation using IES data."""
    cob_positions, X, Y, (p_centers, p_areas, p_normals, p_refl) = geo

    # Assign target lumens to each COB based on its layer and the params array
    target_lumens_per_cob = pack_luminous_flux_dynamic(params, cob_positions)

    # Extract necessary data from IESParser object for Numba functions
    ies_lumens_base = ies_data.lumens_per_lamp * ies_data.num_lamps
    ies_multiplier = ies_data.multiplier
    v_angles = ies_data.v_angles
    h_angles = ies_data.h_angles
    candelas = ies_data.candelas

    # --- Direct Illumination Calculation ---
    print("Calculating direct illumination on floor...")
    floor_lux_direct = compute_direct_floor(
        cob_positions, target_lumens_per_cob,
        ies_lumens_base, ies_multiplier, v_angles, h_angles, candelas,
        X, Y)
    print("Calculating direct illumination on patches...")
    patch_lux_direct = compute_patch_direct(
        cob_positions, target_lumens_per_cob,
        ies_lumens_base, ies_multiplier, v_angles, h_angles, candelas,
        p_centers, p_normals, p_areas)

    # --- Indirect Illumination (Radiosity) ---
    print("Starting radiosity calculation...")
    # patch_rad is the final radiosity (Direct + Indirect Illuminance) on each patch
    patch_rad = iterative_radiosity_loop(
        p_centers, p_normals, patch_lux_direct, p_areas, p_refl,
        MAX_RADIOSITY_BOUNCES, RADIOSITY_CONVERGENCE_THRESHOLD)

    print("Calculating indirect illumination reflected onto floor...")
    # This calculates the illuminance on the floor grid due to light *leaving* the patches
    floor_lux_indirect = compute_reflection_on_floor(
        X, Y, p_centers, p_normals, p_areas, patch_rad, p_refl, MC_SAMPLES)

    # --- Total Illumination and Conversion ---
    total_floor_lux = floor_lux_direct + floor_lux_indirect

    # Convert Lux to PPFD (µmol/m²/s)
    # Total Lux -> Total Radiant Power (W/m²) -> PPFD
    total_radiant_Wm2 = total_floor_lux / LUMINOUS_EFFICACY # Check if LUMINOUS_EFFICACY is appropriate for the *specific* light source IES/SPD
    # If SPD file was provided and factor computed, use it
    if 'CONVERSION_FACTOR' in globals() and CONVERSION_FACTOR > EPSILON:
        floor_ppfd = total_radiant_Wm2 * CONVERSION_FACTOR
        print(f"Using computed CONVERSION_FACTOR: {CONVERSION_FACTOR:.5f}")
    else:
        # Fallback or alternative conversion? Needs careful consideration.
        # Using a generic factor if SPD fails is an option but less accurate.
        print("[Warning] Using default/fallback CONVERSION_FACTOR.")
        # Maybe calculate conversion factor here if not done globally?
        fallback_factor = compute_conversion_factor(SPD_FILE) # Try again?
        floor_ppfd = total_radiant_Wm2 * fallback_factor

    return floor_ppfd, X, Y, cob_positions, total_floor_lux # Return total lux as well

# ------------------------------------
# 10) Main Execution Block
# ------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run lighting simulation using IES file data.")
    parser.add_argument('--ies-file', type=str, default=DEFAULT_IES_FILE,
                        help=f"Path to the IES photometric file (default: {DEFAULT_IES_FILE})")
    parser.add_argument('--spd-file', type=str, default=SPD_FILE,
                        help=f"Path to the SPD spectral data file for PPFD conversion (default: {SPD_FILE})")
    parser.add_argument('--output-csv', type=str, default="ppfd_data.csv",
                        help="Name of the output CSV file (default: ppfd_data.csv)")
    parser.add_argument('--plot', action='store_true', default=True, help='Show the PPFD heatmap plot.')
    parser.add_argument('--plot-lux', action='store_true', help='Show the Lux heatmap plot.')
    parser.add_argument('-W', '--width', type=float, default=11.86, help='Room width (m)') # 38.9 ft ~ 11.86m
    parser.add_argument('-L', '--length', type=float, default=11.86, help='Room length (m)') # 38.9 ft ~ 11.86m
    parser.add_argument('-H', '--height', type=float, default=0.9144, help='COB mounting height (m)') # 3 ft ~ 0.9144m

    args = parser.parse_args()

    # --- Setup based on args ---
    W = args.width
    L = args.length
    H = args.height
    ies_filepath = args.ies_file
    spd_filepath = args.spd_file
    output_csv_file = args.output_csv

    print("--- Simulation Parameters ---")
    print(f" Room Dimensions (WxLxH): {W:.2f}m x {L:.2f}m x {H:.2f}m")
    print(f" IES File: {ies_filepath}")
    print(f" SPD File: {spd_filepath}")
    print(f" Output CSV: {output_csv_file}")
    print(f" Reflectances (W/C/F): {REFL_WALL}/{REFL_CEIL}/{REFL_FLOOR}")
    print(f" Luminous Efficacy: {LUMINOUS_EFFICACY} lm/W")
    print(f" Radiosity Bounces/Threshold: {MAX_RADIOSITY_BOUNCES} / {RADIOSITY_CONVERGENCE_THRESHOLD}")
    print(f" Floor Grid Resolution: {FLOOR_GRID_RES} m")
    print(f" Fixed Number of Layers: {FIXED_NUM_LAYERS}")
    print("---------------------------")


    # --- Parse IES File ---
    try:
        ies_data = IESParser(ies_filepath)
    except Exception as e:
        print(f"[FATAL] Failed to parse IES file: {e}")
        return # Exit if IES parsing fails

    # --- Compute Conversion Factor ---
    # Define globally after computing it
    global CONVERSION_FACTOR
    CONVERSION_FACTOR = compute_conversion_factor(spd_filepath)


    # --- Define Lumens per Layer ---
    # These are the *target* lumens for all COBs within a specific layer.
    # The number of elements MUST match FIXED_NUM_LAYERS.
    # Example using your provided values for 19 layers:
    if FIXED_NUM_LAYERS == 19:
        params = np.array([
            1500, 2000, 3500, 8000, 14000, 8000, 8000, 13500, 11000, 12000,
            10000, 9000, 8500, 9000, 10000, 11000, 12167, 20000, 20000
        ], dtype=np.float64)
    else:
        # Define default/fallback params if FIXED_NUM_LAYERS is different
        print(f"[Warning] FIXED_NUM_LAYERS is {FIXED_NUM_LAYERS}. Using linearly scaled default params.")
        base_lumens = 10000 # Example average
        params = np.linspace(base_lumens*0.5, base_lumens*1.5, FIXED_NUM_LAYERS)


    if len(params) != FIXED_NUM_LAYERS:
        print(f"[FATAL] Length of 'params' array ({len(params)}) does not match FIXED_NUM_LAYERS ({FIXED_NUM_LAYERS}). Please correct the 'params' definition.")
        return

    # --- Prepare Geometry ---
    print("Preparing geometry...")
    geo = prepare_geometry(W, L, H) # Returns (cob_positions, X, Y, patches)

    # --- Run Simulation ---
    print("Starting lighting simulation...")
    floor_ppfd, X, Y, cob_positions, total_floor_lux = simulate_lighting(params, geo, ies_data)
    print("Simulation finished.")

    # --- Calculate Statistics ---
    if floor_ppfd is not None and floor_ppfd.size > 0:
        mean_ppfd = np.mean(floor_ppfd)
        std_dev_ppfd = np.std(floor_ppfd)
        min_ppfd = np.min(floor_ppfd)
        max_ppfd = np.max(floor_ppfd)

        # Uniformity Metrics
        mad = np.mean(np.abs(floor_ppfd - mean_ppfd)) # Mean Absolute Deviation
        rmse = np.sqrt(np.mean((floor_ppfd - mean_ppfd)**2)) # Root Mean Square Error (same as std dev here)
        cv = 100 * (std_dev_ppfd / mean_ppfd) if mean_ppfd > EPSILON else 0 # Coefficient of Variation (%)
        dou = 100 * (1 - rmse / mean_ppfd) if mean_ppfd > EPSILON else 0 # Distributional Operating Uniformity (%) - Based on RMSE
        mdou = 100 * (1 - mad / mean_ppfd) if mean_ppfd > EPSILON else 0 # Distributional Operating Uniformity (%) - Based on MAD
        min_max_ratio = (min_ppfd / max_ppfd) * 100 if max_ppfd > EPSILON else 0 # Min/Max Ratio (%)

        print("\n--- Simulation Results (PPFD) ---")
        print(f" Average PPFD: {mean_ppfd:.2f} µmol/m²/s")
        print(f" Standard Deviation: {std_dev_ppfd:.2f}")
        print(f" Minimum PPFD: {min_ppfd:.2f}")
        print(f" Maximum PPFD: {max_ppfd:.2f}")
        print(f" MAD: {mad:.2f}")
        print(f" RMSE: {rmse:.2f}")
        print(f" CV (%): {cv:.2f}")
        print(f" DOU (RMSE-based) (%): {dou:.2f}")
        print(f" M-DOU (MAD-based) (%): {mdou:.2f}")
        print(f" Min/Max Ratio (%): {min_max_ratio:.2f}")
        print("---------------------------------")

        # --- CSV Output ---
        write_ppfd_to_csv(output_csv_file, floor_ppfd, X, Y, cob_positions, W, L)

        # --- Plotting ---
        if args.plot:
            print("Generating PPFD heatmap...")
            X, Y = build_floor_grid(params[0], params[1])
            floor_positions = np.stack([X.ravel(), Y.ravel()], axis=1)
            plot_heatmap(floor_ppfd, X, Y, cob_positions, title="Floor PPFD", units="µmol/m²/s", annotation_step=10)
            plt.show(block=True)

        if args.plot_lux:
            print("Generating Lux heatmap...")
            plot_heatmap(total_floor_lux, X, Y, cob_positions, title="Floor Illuminance", units="lux", annotation_step=10)
            plt.show(block=True)


    else:
        print("[Error] Simulation did not produce valid PPFD results.")


if __name__ == "__main__":
    main()
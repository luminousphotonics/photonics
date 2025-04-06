import os
import sys
# Add builtins import here
import builtins
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: 
    from predictive_tool import predict_flux_assignments, TARGET_PPFD 
    SPLINE_AVAILABLE = True 
except ImportError: SPLINE_AVAILABLE = False

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import queue # Import queue for type hinting if needed
import time # For basic timing

# --- Define MAX_LAYERS early, as it's used in the import warning ---
MAX_LAYERS = 20  # Max layers this script *attempts* to handle (padding target)

# --- Simulation Import ---
try:
    # Now MAX_LAYERS is defined when this print statement runs
    from lighting_simulation_data_1 import prepare_geometry, simulate_lighting # Removed FIXED_NUM_LAYERS import
    SIMULATION_AVAILABLE = True
    # The print warning can now use MAX_LAYERS safely
    # print(f"\n*** INFO: Using simulation 'lighting_simulation_data_1'. Optimizer MAX_LAYERS set to {MAX_LAYERS}. ***\n")

except ImportError:
    print("ERROR: Could not import 'lighting_simulation_data_1'.")
    print("Make sure 'lighting_simulation_data_1.py' is in the same directory or accessible.")
    SIMULATION_AVAILABLE = False
    # FIXED_NUM_LAYERS = 9 # Keep this commented or remove if not needed as fallback
    if __name__ == "__main__":
        exit()
    else:
        raise ImportError("Missing required simulation module 'lighting_simulation_data_1'")
except Exception as e:
    print(f"ERROR: Unexpected error importing simulation: {e}")
    SIMULATION_AVAILABLE = False
    # FIXED_NUM_LAYERS = 9
    if __name__ == "__main__":
        exit()
    else:
        raise ImportError(f"Error importing simulation: {e}")


# --- ML / Optimization Imports ---
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from scipy.optimize import minimize_scalar, minimize, OptimizeResult
    from scipy.linalg import LinAlgError
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}. Please install scikit-learn and scipy.")
    if __name__ == "__main__":
        exit()
    else:
        raise ImportError(f"Missing dependency: {e}")


# -------------------------
# GLOBAL LOG CAPTURE SETUP
# -------------------------
global_log_lines = []
# builtins was imported earlier, this line is now correct
_original_print = builtins.print # Keep a reference to the original print

# --- Modified Log Function ---
def log(msg, queue_instance=None):
    """Logs a message. If queue_instance is provided, puts message ONLY on queue.
       Otherwise, prints to console. Always appends to global list."""
    str_msg = str(msg)
    global_log_lines.append(str_msg) # Always append to global list

    if queue_instance:
        try:
            # Ensure message ends with newline for GUI text widget
            if not str_msg.endswith('\n'):
                str_msg += '\n'
            queue_instance.put(str_msg)
        except Exception as e:
            _original_print(f"[Log Error] Failed to put message on queue: {e}")
            # Optionally print here too if queue fails? Might cause duplicates if redirection is on.
            # _original_print(str_msg, flush=True)
    else:
        # No queue provided (e.g., standalone run), print to console
        _original_print(str_msg, flush=True)

# -------------------------
# CONFIGURATION
# -------------------------
DATASET_FILE = "flux_learning_dataset.npz"
LOG_FILE = "flux_optimization_log.csv"
# MAX_LAYERS = 20 # Definition moved earlier
DOU_TARGET = 95.0 # Target Uniformity (float)
# PPFD_TOLERANCE = 15 # Informational

# --- Constraint Values ---
MAX_LUMENS_OUTER = 20000.0
MAX_LUMENS_INNER_HALF = 10000.0

# --- Penalty Factor ---
# Start with a moderate value. Increase if constraints are heavily violated.
# Decrease if penalties dominate the loss too much and hinder optimization.
FLUX_CONSTRAINT_PENALTY_FACTOR = 1.0 # Adjust as needed (e.g., 0.1, 1.0, 10.0)

# --- Floor Sizes Map ---
# IMPORTANT: Keys (10-20) currently correspond to user's desired layer count,
# but the simulation uses FIXED_NUM_LAYERS (likely 9). This mapping might need
# revision depending on how the simulation layer mismatch is resolved.
FLOOR_SIZES = {
    1: 0.34, 2: 0.98, 3: 1.62,
    4: 2.26, 5: 2.9, 6: 3.54, 7: 4.18, 8: 4.82, 9: 5.46,
    10: 6.10, 11: 6.74, 12: 7.38, 13: 8.02, 14: 8.66,
    15: 9.30, 16: 9.94, 17: 10.58, 18: 11.22, 19: 11.68, 20: 12.32
}

# Add sizes for layers < 10 if needed, e.g., using layer 10 size?
# Or maybe the simulation only works for a specific size? This needs clarification.
# For now, assume layers < 10 aren't intended or use layer 10's size.
DEFAULT_FLOOR_SIZE = FLOOR_SIZES[10]

# --- Sweeping parameters (for standalone run) ---
layer_counts_sweep = list(range(10, 12)) # Reduced range for faster testing
heights_ft_sweep = np.linspace(3, 5, 2) # Reduced range
ppfd_targets_sweep = np.linspace(800, 1200, 2) # Reduced range

# --- Optimization settings ---
# Try COBYLA first as it's often better than Nelder-Mead for this type of problem
# Other options: 'Nelder-Mead', 'Powell', 'L-BFGS-B' (needs gradient approx)
OPTIMIZER_METHOD = 'COBYLA'
OPTIMIZER_MAXITER = 50 # Reduced iterations for faster testing/debugging
OPTIMIZER_TOLERANCE = 1e-3 # Tolerance for convergence (used by some methods like COBYLA's rhobeg/rhoend)
HIGH_LOSS = 1e12

# --- ML Settings ---
MIN_SAMPLES_FOR_TRAINING = 5 # Reduced for testing
RF_ESTIMATORS = 100
RF_RANDOM_STATE = 42
RF_N_JOBS = -1

# --- Pretrained Profiles (Used for Fallback Initial Guess) ---
# IMPORTANT: These profiles have varying lengths. The get_fallback_profile
# function will attempt to use the one matching the requested layer count.
# If the requested layer count > 9, the simulation might still only use the first 9 values.
PRETRAINED_PROFILES = {
    # Add profiles for layers < 10 if they exist or default to uniform?
    # Using FIXED_NUM_LAYERS (e.g. 9) might be more appropriate if sim is fixed
     9: [1500, 2001, 3501, 8001, 14000, 8000, 5498, 13501, 16131], # Example 9 layer
    10: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 20000, 20000],
    11: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 11000, 20000, 20000],
    12: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 11000, 8000, 20000, 20000],
    13: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 11000, 8000, 10000, 20000, 20000],
    14: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 11000, 9000, 10000, 9000, 20000, 20000],
    15: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 11000, 9000, 10000, 9000, 9500, 20000, 20000],
    16: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 11000, 9000, 10000, 9000, 8500, 11000, 20000, 20000],
    17: [1500, 2000, 3500, 8000, 14000, 8000, 8000, 13500, 11000, 9000, 10000, 9000, 8500, 9000, 10000, 20000, 20000],
    18: [1500, 2000, 3500, 8000, 14000, 8000, 8000, 13500, 11000, 12000, 10000, 9000, 8500, 9000, 10000, 11000, 20000, 20000],
    19: [1500, 2000, 3500, 8000, 14000, 8000, 8000, 13500, 11000, 12000, 10000, 9000, 8500, 9000, 10000, 11000, 12167, 20000, 20000],
    20: [1500, 2000, 3500, 8000, 14000, 8000, 8000, 13500, 11000, 12000, 10000, 9000, 8500, 9000, 10000, 11000, 12167, 13000, 20000, 20000]
}


# -------------------------
# Helper Functions
# -------------------------

def get_floor_size(layer_count, queue_instance=None):
    """Gets floor size based on layer count, handling missing keys."""
    size = FLOOR_SIZES.get(layer_count)
    if size is None:
        # Fallback logic: Use default or closest? Warn user.
        log(f"Warning: Layer count {layer_count} not in FLOOR_SIZES map. Using default size {DEFAULT_FLOOR_SIZE}m.", queue_instance=queue_instance)
        size = DEFAULT_FLOOR_SIZE
    return size

def get_fallback_profile(layer_count, queue_instance=None):
    """Gets a normalized fallback profile, adjusting length if needed."""
    if layer_count <= 0:
        log(f"Error: Cannot get profile for invalid layer_count {layer_count}", queue_instance=queue_instance)
        return np.array([])

    profile = PRETRAINED_PROFILES.get(layer_count)

    if profile is not None:
        log(f"Using pre-trained profile for L={layer_count}", queue_instance=queue_instance)
        profile = np.array(profile, dtype=float)
        # Ensure correct length (should match key, but good practice)
        if len(profile) != layer_count:
            log(f"Warning: Pre-trained profile L={layer_count} length mismatch ({len(profile)}). Adjusting.", queue_instance=queue_instance)
            if len(profile) < layer_count:
                profile = np.pad(profile, (0, layer_count - len(profile)), mode='edge')
            else:
                profile = profile[:layer_count]
    else:
        # Try using the profile for the simulation's fixed layer count if request is different
        fixed_layer_profile = PRETRAINED_PROFILES.get(MAX_LAYERS)
        if fixed_layer_profile is not None and layer_count != MAX_LAYERS:
            log(f"Warning: No profile for L={layer_count}. Using profile for simulation's L={MAX_LAYERS} and adjusting length.", queue_instance=queue_instance)
            profile = np.array(fixed_layer_profile, dtype=float)
            if len(profile) < layer_count:
                profile = np.pad(profile, (0, layer_count - len(profile)), mode='edge')
            else:
                profile = profile[:layer_count]
        else:
            # Fallback to uniform if no specific or fixed profile found
            log(f"Warning: No specific profile for L={layer_count}. Using uniform distribution.", queue_instance=queue_instance)
            profile = np.ones(layer_count, dtype=float)

    # Normalize
    profile_sum = np.sum(profile)
    if profile_sum > 1e-9:
        return profile / profile_sum
    else:
        log(f"Warning: Fallback profile for L={layer_count} sums to zero. Returning uniform.", queue_instance=queue_instance)
        return np.ones(layer_count) / layer_count

# --- Dataset Handling (minor robustness added) ---
def load_dataset(queue_instance=None):
    X, Y = [], []
    if os.path.exists(DATASET_FILE):
        try:
            # Use context manager for file handling
            with np.load(DATASET_FILE, allow_pickle=True) as data:
                X_loaded = data["X"]
                Y_loaded = data["Y"] # This is likely an object array

                if len(X_loaded) != len(Y_loaded):
                    log(f"Warning: Mismatch in loaded data lengths. X: {len(X_loaded)}, Y: {len(Y_loaded)}. Resetting.", queue_instance=queue_instance)
                    return [], []

                # Convert X to list of lists (if not already)
                X = [list(item) for item in X_loaded]

                # Convert Y object array to list of numpy float arrays
                Y = []
                for item in Y_loaded:
                    try:
                        # Ensure item is array-like and convert to float array
                        y_arr = np.array(item, dtype=float)
                        if y_arr.ndim == 1 and y_arr.size > 0 : # Check it's a non-empty 1D array
                             Y.append(y_arr)
                        else:
                             log(f"Warning: Skipping invalid Y data item during load: {item}", queue_instance=queue_instance)
                    except Exception as e_conv:
                         log(f"Warning: Skipping Y data item due to conversion error: {item} ({e_conv})", queue_instance=queue_instance)

                # Ensure X and Y still match after filtering Y
                if len(X) != len(Y):
                     log(f"Warning: Mismatch after cleaning Y data. X:{len(X)}, Y:{len(Y)}. Resetting.", queue_instance=queue_instance)
                     return [], []

                log(f"Dataset loaded: {len(X)} valid entries.", queue_instance=queue_instance)
                return X, Y

        except Exception as e:
            log(f"Error loading dataset '{DATASET_FILE}': {e}", queue_instance=queue_instance)
            log("Starting with an empty dataset.", queue_instance=queue_instance)
            return [], []
    else:
        log("Dataset file not found. Starting with empty dataset.", queue_instance=queue_instance)
        return [], []

def save_dataset(X, Y, queue_instance=None):
    if not X or not Y or len(X) != len(Y):
         log("Warning: Attempted to save invalid or mismatched X/Y data. Save aborted.", queue_instance=queue_instance)
         return
    try:
        # Save Y as object array to handle potentially varying lengths
        Y_save = np.array(Y, dtype=object)
        np.savez(DATASET_FILE, X=np.array(X, dtype=float), Y=Y_save)
        log(f"Dataset saved to '{DATASET_FILE}' with {len(X)} entries.", queue_instance=queue_instance)
    except Exception as e:
        log(f"Error saving dataset '{DATASET_FILE}': {e}", queue_instance=queue_instance)

# --- Log Result (Unchanged) ---
def log_result(entry, queue_instance=None):
    df = pd.DataFrame([entry])
    file_exists = os.path.exists(LOG_FILE)
    try:
        df.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)
    except Exception as e:
        log(f"Error writing to log file '{LOG_FILE}': {e}", queue_instance=queue_instance)

# --- Normalization (Unchanged) ---
def normalize_input(layer, height_m, ppfd, queue_instance=None):
    max_layer_num = MAX_LAYERS # Use fixed MAX_LAYERS for consistent normalization
    max_height_m = 10 * 0.3048 # Max 10ft
    max_ppfd = 1500           # Max 1500 PPFD
    norm_layer = layer / max_layer_num if max_layer_num > 0 else 0
    norm_height = height_m / max_height_m if max_height_m > 0 else 0
    norm_ppfd = ppfd / max_ppfd if max_ppfd > 0 else 0
    return [np.clip(norm_layer, 0, 1),
            np.clip(norm_height, 0, 1),
            np.clip(norm_ppfd, 0, 1)]

# --- Simulation and Scaling Helpers (More Logging/Robustness) ---

def safe_simulate_lighting(params_padded, geo, queue_instance=None): # Renamed input for clarity
    """Wrapper for simulate_lighting with error handling and timing."""
    # ... (implementation mostly unchanged, receives padded params and geo) ...
    # Ensure params_padded has MAX_LAYERS length (checked by caller ideally)
    if len(params_padded) != MAX_LAYERS:
         log(f"Error: safe_simulate_lighting input length ({len(params_padded)}) != MAX_LAYERS ({MAX_LAYERS}).", queue_instance=queue_instance)
         return None, -1.0

    start_time = time.time()
    try:
        # Simulate_lighting uses the geo containing correct number of COBs
        # and pack_luminous_flux uses layer index to get data from params_padded
        floor_ppfd, _, _, _ = simulate_lighting(params_padded, geo)
        duration = time.time() - start_time
        #log(f"   Simulate_lighting duration: {duration:.3f}s", queue_instance=queue_instance) # Optional timing log

        # Validate simulation output
        if floor_ppfd is None or not isinstance(floor_ppfd, np.ndarray) or floor_ppfd.size == 0:
            log("Warning: Simulation returned invalid floor_ppfd (None, not array, or empty).", queue_instance=queue_instance)
            return None, duration
        # Check for NaNs/Infs
        if not np.all(np.isfinite(floor_ppfd)):
             num_non_finite = np.count_nonzero(~np.isfinite(floor_ppfd))
             log(f"Warning: Simulation returned {num_non_finite}/{floor_ppfd.size} non-finite PPFD values.", queue_instance=queue_instance)
             # Return the array as is, let downstream handle filtering

        return floor_ppfd, duration

    except LinAlgError as lae:
        duration = time.time() - start_time
        log(f"Linear Algebra Error during simulation: {lae} (Duration: {duration:.3f}s)", queue_instance=queue_instance)
        return None, duration
    except Exception as e:
        duration = time.time() - start_time
        log(f"Error during simulation: {type(e).__name__}: {e} (Duration: {duration:.3f}s)", queue_instance=queue_instance)
        # import traceback # Optional full traceback for debugging
        # log(traceback.format_exc(), queue_instance=queue_instance)
        return None, duration


def scale_flux_to_ppfd(shape_vector, layer_count, target_ppfd, height_m, queue_instance=None):
    """Scales normalized shape_vector to achieve target_ppfd."""
    log(f"  Scaling flux for L={layer_count}, H={height_m:.2f}m, TgtPPFD={target_ppfd:.0f}...", queue_instance=queue_instance)
    size = get_floor_size(layer_count, queue_instance)
    try:
        # <<< CHANGE: Pass layer_count to prepare_geometry >>>
        geo = prepare_geometry(size, size, height_m, layer_count)
        if geo is None: raise ValueError("Geometry preparation failed.")
    except Exception as e:
        log(f"Error preparing geometry L={layer_count} in scale_flux: {e}", queue_instance=queue_instance)
        return np.zeros(layer_count) # Return zeros on error

    shape_vector = np.array(shape_vector, dtype=float)
    if len(shape_vector) != layer_count:
        log(f"Error: scale_flux shape length ({len(shape_vector)}) != layer_count ({layer_count}).", queue_instance=queue_instance)
        return np.zeros(layer_count)

    shape_vector = np.abs(shape_vector) # Ensure non-negative
    sum_shape = np.sum(shape_vector)
    if sum_shape <= 1e-9:
        log("Warning: shape_vector sum near zero in scale_flux. Cannot normalize.", queue_instance=queue_instance)
        return np.zeros_like(shape_vector)
    norm_shape = shape_vector / sum_shape

    # --- Nested function for optimization target ---
    last_checked_scale = -1.0
    def ppfd_error(scale):
        nonlocal last_checked_scale
        last_checked_scale = scale
        if scale <= 1e-6: return HIGH_LOSS

        scaled_flux = norm_shape * scale
        # Pad the vector to MAX_LAYERS for the simulation function interface
        padded_flux = np.pad(scaled_flux, (0, MAX_LAYERS - layer_count), constant_values=0)

        # Use the safe wrapper for simulation, passing the prepared geo
        floor_ppfd, sim_duration = safe_simulate_lighting(padded_flux, geo, queue_instance)

        if floor_ppfd is None:
             #log(f"    Scale Check ({scale:.1f}): Simulation failed.", queue_instance=queue_instance) # Verbose
             return HIGH_LOSS # High loss if simulation fails

        finite_ppfd = floor_ppfd[np.isfinite(floor_ppfd)]
        if finite_ppfd.size == 0:
             #log(f"    Scale Check ({scale:.1f}): Only non-finite PPFD.", queue_instance=queue_instance) # Verbose
             return HIGH_LOSS # High loss if no valid PPFD values

        mean_ppfd = np.mean(finite_ppfd)
        error = abs(mean_ppfd - target_ppfd)
        return error

    # --- Optimization to find the best scale ---
    try:
        # Estimate bounds. Lower bound must be > 0. Upper bound needs to be generous.
        # Flux is roughly PPFD * Area * Factor. Factor ~ 1 / (LUM_EFFIC * CONV_FACTOR) ~ 1/(182*0.014) ~ 0.4
        # Rough Total Flux ~ TargetPPFD * Size^2 * 0.4
        # Upper Scale Guess ~ Rough Total Flux (maybe * 5 or 10 for safety?)
        rough_upper_flux = target_ppfd * size * size * 0.5 * 10 # Generous upper bound
        lower_bound = 1e-3
        upper_bound = max(lower_bound * 10, rough_upper_flux) # Ensure upper > lower

        log(f"   Running minimize_scalar for scaling (bounds: {lower_bound:.2e} to {upper_bound:.2e})", queue_instance=queue_instance)
        res = minimize_scalar(ppfd_error,
                              bounds=(lower_bound, upper_bound),
                              method='bounded',
                              options={'xatol': 1e-2, 'maxiter': 50}) # Tolerance & max iterations for scaling

        if res.success and res.x > lower_bound:
            final_scale = res.x
            scaled_result = norm_shape * final_scale
            scaled_result = np.maximum(scaled_result, 0) # Final safety check
            # Verify the result of the scaling
            final_error = ppfd_error(final_scale)
            log(f"  Scaling finished: Success={res.success}. Final Scale={final_scale:.1f}, Final PPFD Error={final_error:.1f}", queue_instance=queue_instance)
            return scaled_result
        else:
            log(f"Warning: Scaling optimization failed (Success={res.success}, Scale={res.x:.2e}, Fun={res.fun:.2e}). Last checked scale: {last_checked_scale:.2e}. Returning zero flux.", queue_instance=queue_instance)
            return np.zeros_like(shape_vector)

    except Exception as e:
        log(f"Error during scaling optimization: {type(e).__name__}: {e}", queue_instance=queue_instance)
        return np.zeros_like(shape_vector)


def simulate_loss(scaled_flux, layer_count, target_ppfd, height_m, queue_instance=None):
    """Calculates loss based on PPFD deviation and mDOU penalty."""
    log(f"  Simulating loss for L={layer_count}, H={height_m:.2f}m, TgtPPFD={target_ppfd:.0f}...", queue_instance=queue_instance)
    size = get_floor_size(layer_count, queue_instance)
    try:
        # <<< CHANGE: Pass layer_count to prepare_geometry >>>
        geo = prepare_geometry(size, size, height_m, layer_count)
        if geo is None: raise ValueError("Geometry preparation failed.")
    except Exception as e:
        log(f"Error preparing geometry L={layer_count} in simulate_loss: {e}", queue_instance=queue_instance)
        return HIGH_LOSS, -1.0, -1.0

    scaled_flux = np.array(scaled_flux, dtype=float)
    if len(scaled_flux) != layer_count:
         log(f"Error: simulate_loss flux length ({len(scaled_flux)}) != layer_count ({layer_count}).", queue_instance=queue_instance)
         return HIGH_LOSS, -1.0, -1.0

    scaled_flux = np.nan_to_num(scaled_flux, nan=0.0, posinf=0.0, neginf=0.0)
    scaled_flux = np.maximum(scaled_flux, 0) # Ensure non-negative

    padded_flux = np.pad(scaled_flux, (0, MAX_LAYERS - layer_count), constant_values=0)

    # Use safe wrapper
    floor_ppfd, sim_duration = safe_simulate_lighting(padded_flux, geo, queue_instance)

    if floor_ppfd is None:
        log("  Simulate_loss: Simulation failed. Returning HIGH_LOSS.", queue_instance=queue_instance)
        return HIGH_LOSS, -1.0, -1.0

    finite_ppfd = floor_ppfd[np.isfinite(floor_ppfd)]
    if finite_ppfd.size == 0:
        log("Warning: Simulation resulted in only non-finite PPFD values in simulate_loss.", queue_instance=queue_instance)
        mean_ppfd = 0.0
        m_dou = 0.0
        loss = HIGH_LOSS # High loss if no valid PPFD
        log(f"  Loss Calc: MeanPPFD=N/A, mDOU=N/A -> Loss={loss:.2e}", queue_instance=queue_instance)
        return loss, mean_ppfd, m_dou
    else:
        mean_ppfd = np.mean(finite_ppfd)

    # Calculate mDOU
    if mean_ppfd <= 1e-6: # Avoid division by near-zero
         mad = 0.0
         m_dou = 0.0 # Assign 0 mDOU if average PPFD is effectively zero
    else:
         mad = np.mean(np.abs(finite_ppfd - mean_ppfd))
         m_dou = 100.0 * (1.0 - mad / mean_ppfd)
         m_dou = np.clip(m_dou, 0.0, 100.0) # Clamp mDOU

    if not np.isfinite(m_dou):
         log(f"Warning: Calculated non-finite mDOU ({m_dou}). Setting to 0.", queue_instance=queue_instance)
         m_dou = 0.0

    # Calculate Loss = PPFD Error + Uniformity Penalty
    ppfd_term = abs(mean_ppfd - target_ppfd)
    dou_penalty = 10.0 * max(0.0, DOU_TARGET - m_dou)**2 # Squared penalty for larger deviations? Or keep linear?
    #dou_penalty = 10.0 * max(0.0, DOU_TARGET - m_dou) # Linear penalty

    loss = ppfd_term + dou_penalty

    if not np.isfinite(loss):
         log(f"Warning: Calculated non-finite loss (PPFD Term: {ppfd_term:.2f}, DOU Penalty: {dou_penalty:.2f}). Setting to HIGH_LOSS.", queue_instance=queue_instance)
         loss = HIGH_LOSS

    log(f"  Loss Calc: MeanPPFD={mean_ppfd:.1f}, mDOU={m_dou:.1f}% -> PPFD_Err={ppfd_term:.1f}, DOU_Pen={dou_penalty:.1f} -> Loss={loss:.3f}", queue_instance=queue_instance)
    return loss, mean_ppfd, m_dou


# --- Optimization Function ---
def optimize_flux(initial_shape_guess, layer_count, target_ppfd, height_m, queue_instance=None):
    """Optimizes the flux shape vector to minimize loss.
    
    - Each layer's flux is capped at a maximum of 20,000 lumens.
    - PPFD error within ±50 lumens is not penalized to favor higher uniformity (DOU).
    """
    iteration_count = 0  # Counter for logging iterations
    MAX_FLUX = 20000.0
    PPFD_TOLERANCE = 50.0

    initial_shape_guess = np.array(initial_shape_guess, dtype=float)
    if len(initial_shape_guess) != layer_count:
        log(f"Warning: Initial guess length ({len(initial_shape_guess)}) != layer_count ({layer_count}). Using fallback.", queue_instance=queue_instance)
        initial_shape_guess = get_fallback_profile(layer_count, queue_instance=queue_instance)
        if len(initial_shape_guess) != layer_count:
            log(f"Error: Fallback profile generation failed for L={layer_count}. Cannot optimize.", queue_instance=queue_instance)
            return np.ones(layer_count) / layer_count, HIGH_LOSS, -1.0, -1.0

    # Spline-based initial guess integration
    if SPLINE_AVAILABLE:
        spline_guess = predict_flux_assignments(
            num_layers_target=layer_count,
            known_configs=PRETRAINED_PROFILES,
            target_ppfd=target_ppfd
        )
        log(f"Using spline-based initial guess: {spline_guess}", queue_instance=queue_instance)
        initial_shape_guess = spline_guess

    # Normalize initial guess
    initial_shape_guess = np.abs(initial_shape_guess)
    sum_initial = np.sum(initial_shape_guess)
    if sum_initial <= 1e-9:
        log("Warning: Initial guess sum near zero. Using uniform.", queue_instance=queue_instance)
        initial_shape_guess = np.ones(layer_count) / layer_count
    else:
        initial_shape_guess = initial_shape_guess / sum_initial

    log(f"Starting optimization ({OPTIMIZER_METHOD}, maxiter={OPTIMIZER_MAXITER}, tol={OPTIMIZER_TOLERANCE:.1e})...", queue_instance=queue_instance)
    log(f"Constraints: Each layer's flux is capped at {MAX_FLUX} lumens. PPFD tolerance: ±{PPFD_TOLERANCE} lumens.", queue_instance=queue_instance)
    log(f"Initial Shape (Normalized): {np.array2string(initial_shape_guess, precision=3, max_line_width=120)}", queue_instance=queue_instance)

    # Evaluate initial guess
    initial_loss_raw, initial_ppfd, initial_mdou = -1.0, -1.0, -1.0
    initial_loss_total = HIGH_LOSS
    try:
        log("  Evaluating initial guess...", queue_instance=queue_instance)
        initial_scaled_flux = scale_flux_to_ppfd(initial_shape_guess, layer_count, target_ppfd, height_m, queue_instance=queue_instance)
        # Hard cap each layer to MAX_FLUX
        initial_scaled_flux = np.minimum(initial_scaled_flux, MAX_FLUX)
        if np.sum(initial_scaled_flux) > 1e-9:
            initial_loss_raw, initial_ppfd, initial_mdou = simulate_loss(initial_scaled_flux, layer_count, target_ppfd, height_m, queue_instance=queue_instance)
            raw_ppfd_error = abs(initial_ppfd - target_ppfd)
            adjusted_ppfd_error = max(0, raw_ppfd_error - PPFD_TOLERANCE)
            # Total loss: DOU penalty part remains as given by the simulation loss minus the PPFD error
            initial_loss_total = adjusted_ppfd_error + (initial_loss_raw - raw_ppfd_error)
            log(f"Initial Scaled Flux: {np.array2string(initial_scaled_flux, precision=1, max_line_width=120)}", queue_instance=queue_instance)
            log(f"Initial Eval: RawLoss={initial_loss_raw:.3f} (PPFD error {raw_ppfd_error:.1f} adjusted to {adjusted_ppfd_error:.1f}) -> TotalLoss={initial_loss_total:.3f}, PPFD={initial_ppfd:.1f}, mDOU={initial_mdou:.1f}%", queue_instance=queue_instance)
            if initial_loss_total >= HIGH_LOSS:
                log("Error: Initial guess results in high/invalid total loss. Aborting optimization.", queue_instance=queue_instance)
                return initial_shape_guess, HIGH_LOSS, initial_ppfd, initial_mdou
        else:
            log("Error: Scaling failed for initial guess. Aborting optimization.", queue_instance=queue_instance)
            return initial_shape_guess, HIGH_LOSS, -1.0, -1.0
    except Exception as e:
        log(f"Error during initial guess evaluation: {type(e).__name__}: {e}. Aborting.", queue_instance=queue_instance)
        return initial_shape_guess, HIGH_LOSS, -1.0, -1.0

    # Define loss function
    def loss_fn_opt(shape_vec):
        nonlocal iteration_count
        iteration_count += 1
        log(f"--- Opt Iteration {iteration_count} ---", queue_instance=queue_instance)
        shape_vec = np.array(shape_vec, dtype=float)
        shape_vec = np.abs(shape_vec)
        sum_vec = np.sum(shape_vec)
        if sum_vec <= 1e-9:
            log(f" Iter {iteration_count}: Shape vector sum near zero, returning HIGH_LOSS", queue_instance=queue_instance)
            return HIGH_LOSS
        normed_shape = shape_vec / sum_vec
        scaled_flux_vec = scale_flux_to_ppfd(normed_shape, layer_count, target_ppfd, height_m, queue_instance=queue_instance)
        scaled_flux_vec = np.minimum(scaled_flux_vec, MAX_FLUX)  # enforce maximum flux per layer
        if np.sum(scaled_flux_vec) <= 1e-9:
            log(f" Iter {iteration_count}: Scaling failed, returning HIGH_LOSS", queue_instance=queue_instance)
            return HIGH_LOSS
        loss_val_raw, mean_ppfd, mdou = simulate_loss(scaled_flux_vec, layer_count, target_ppfd, height_m, queue_instance=queue_instance)
        if not np.isfinite(loss_val_raw) or loss_val_raw >= HIGH_LOSS:
            log(f" Iter {iteration_count}: Raw loss invalid or HIGH_LOSS ({loss_val_raw:.3e}), returning HIGH_LOSS", queue_instance=queue_instance)
            flux_str = np.array2string(scaled_flux_vec, precision=1, max_line_width=120, separator=', ')
            log(f"            PPFD={mean_ppfd:.1f}, mDOU={mdou:.1f}%, Flux=[{flux_str}]", queue_instance=queue_instance)
            return HIGH_LOSS
        raw_ppfd_error = abs(mean_ppfd - target_ppfd)
        adjusted_ppfd_error = max(0, raw_ppfd_error - PPFD_TOLERANCE)
        total_loss = adjusted_ppfd_error + (loss_val_raw - raw_ppfd_error)
        flux_str = np.array2string(scaled_flux_vec, precision=1, max_line_width=120, separator=', ')
        log(f" Iter {iteration_count}: Raw PPFD error={raw_ppfd_error:.1f} (adjusted to {adjusted_ppfd_error:.1f}) -> TotalLoss={total_loss:.3f}", queue_instance=queue_instance)
        log(f"            PPFD={mean_ppfd:.1f}, mDOU={mdou:.1f}%, Flux=[{flux_str}]", queue_instance=queue_instance)
        return total_loss

    # Run the optimization
    optimized_shape = initial_shape_guess
    final_loss_total = initial_loss_total
    final_mean_ppfd = initial_ppfd
    final_m_dou = initial_mdou
    final_loss_raw = initial_loss_raw
    opt_result = None
    try:
        opt_start_time = time.time()
        options_dict = {'maxiter': OPTIMIZER_MAXITER, 'disp': False}
        if OPTIMIZER_METHOD == 'COBYLA':
            options_dict['rhobeg'] = 0.1
            options_dict['tol'] = OPTIMIZER_TOLERANCE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            opt_result = minimize(loss_fn_opt, initial_shape_guess, method=OPTIMIZER_METHOD, options=options_dict)
        opt_duration = time.time() - opt_start_time
        log(f"Optimization ended. Duration: {opt_duration:.2f}s", queue_instance=queue_instance)
        log(f"Optimizer Result: {opt_result.message}", queue_instance=queue_instance)
        if hasattr(opt_result, 'nit'):
            log(f" Iterations: {opt_result.nit}", queue_instance=queue_instance)
        if hasattr(opt_result, 'nfev'):
            log(f" Func Evals: {opt_result.nfev}", queue_instance=queue_instance)
        final_opt_loss_total = opt_result.fun if hasattr(opt_result, 'fun') and np.isfinite(opt_result.fun) else HIGH_LOSS
        improved_loss = final_opt_loss_total < initial_loss_total - 1e-6
        if opt_result.success or improved_loss:
            if not opt_result.success:
                log(f"Warning: Optimization did not formally converge but improved total loss ({initial_loss_total:.3f} -> {final_opt_loss_total:.3f}). Using result.", queue_instance=queue_instance)
            optimized_res_shape = np.abs(opt_result.x)
            sum_opt = np.sum(optimized_res_shape)
            if sum_opt <= 1e-9:
                log("Warning: Optimization resulted in near-zero vector. Using initial guess results.", queue_instance=queue_instance)
            else:
                optimized_shape = optimized_res_shape / sum_opt
                log("Optimization successful or improved loss. Performing final evaluation...", queue_instance=queue_instance)
                final_scaled_flux = scale_flux_to_ppfd(optimized_shape, layer_count, target_ppfd, height_m, queue_instance=queue_instance)
                final_scaled_flux = np.minimum(final_scaled_flux, MAX_FLUX)
                if np.sum(final_scaled_flux) > 1e-9:
                    final_loss_raw, final_mean_ppfd, final_m_dou = simulate_loss(final_scaled_flux, layer_count, target_ppfd, height_m, queue_instance=queue_instance)
                    final_loss_total = final_opt_loss_total
                    log(f"Final Scaled Flux: {np.array2string(final_scaled_flux, precision=1, max_line_width=120)}", queue_instance=queue_instance)
                else:
                    log("Error: Scaling failed for optimized shape. Using initial guess results.", queue_instance=queue_instance)
        else:
            log(f"Warning: Optimization failed or did not improve total loss (Final Opt Loss: {final_opt_loss_total:.3f}). Using initial guess results.", queue_instance=queue_instance)
    except LinAlgError as lae:
        log(f"Linear Algebra Error during optimization: {lae}. Using initial guess results.", queue_instance=queue_instance)
    except Exception as e:
        log(f"Error during optimization process: {type(e).__name__}: {e}. Using initial guess results.", queue_instance=queue_instance)
        import traceback
        log(traceback.format_exc(), queue_instance=queue_instance)
    log(f"Final Optimized Shape (Normalized): {np.array2string(optimized_shape, precision=3, max_line_width=120)}", queue_instance=queue_instance)
    log(f"Final Result Eval: RawLoss={final_loss_raw:.3f}, PPFD={final_mean_ppfd:.1f}, mDOU={final_m_dou:.1f}%", queue_instance=queue_instance)
    return optimized_shape, final_loss_raw, final_mean_ppfd, final_m_dou

def iterative_optimization(start_layers=5, max_layers=20, target_ppfd=1250, height_m=3.0, max_iterations_per_layer=10, queue_instance=None):
    """
    Iteratively optimizes the flux shape for increasing layer counts.
    For each layer count (starting at start_layers up to max_layers):
      - Runs the optimization repeatedly (up to max_iterations_per_layer) until the simulated mDOU >= 95%.
      - After each run, adds a training sample (input parameters and optimized flux profile) to the training data file.
      - If mDOU is not met, the optimization for that layer count is repeated.
      - Once mDOU >= 95% is achieved, it moves on to the next layer count.
    """
    current_layer = start_layers
    while current_layer <= max_layers:
        converged = False
        iteration = 0
        log(f"Starting optimization for {current_layer} layers...", queue_instance=queue_instance)
        while not converged and iteration < max_iterations_per_layer:
            iteration += 1
            # Generate an initial guess (using fallback profile here)
            initial_guess = get_fallback_profile(current_layer, queue_instance=queue_instance)
            # Run optimization for the current layer count
            optimized_shape, loss, ppfd, mdou = optimize_flux(initial_guess, current_layer, target_ppfd, height_m, queue_instance=queue_instance)
            log(f"Layer count {current_layer}, Iteration {iteration}: PPFD={ppfd:.1f}, mDOU={mdou:.1f}%", queue_instance=queue_instance)
            # Add training data (features: normalized input parameters, target: optimized flux profile)
            X, Y = load_dataset(queue_instance=queue_instance)
            x_sample = normalize_input(current_layer, height_m, target_ppfd, queue_instance=queue_instance)
            y_sample = optimized_shape  # The optimized flux profile vector
            X.append(x_sample)
            Y.append(y_sample)
            save_dataset(X, Y, queue_instance=queue_instance)
            if mdou >= 95.0:
                log(f"Layer count {current_layer} converged with mDOU {mdou:.1f}% on iteration {iteration}. Moving to next layer count.", queue_instance=queue_instance)
                converged = True
            else:
                log(f"Layer count {current_layer} did not meet target DOU (mDOU {mdou:.1f}%). Retrying...", queue_instance=queue_instance)
        current_layer += 1


# -------------------------
# Main Execution for a Single Scenario
# -------------------------
def solve_single_scenario(layer, height_ft, target_ppfd, log_queue=None):
    """Solves a single scenario: trains model, gets guess, optimizes, saves."""
    global global_log_lines
    global_log_lines = [] # Reset global log lines

    run_start_time = time.time()
    log(f"--- Solving Scenario: Layer={layer}, Height={height_ft:.2f}ft, TargetPPFD={target_ppfd:.0f} ---", queue_instance=log_queue)

    # --- Input Validation ---
    if not isinstance(layer, int) or layer <= 0 or layer > MAX_LAYERS:
        msg = f"Invalid layer count '{layer}'. Must be integer > 0 and <= {MAX_LAYERS}."
        log(f"Error: {msg}", queue_instance=log_queue)
        return {"error": msg, "logs": "\n".join(global_log_lines)}
    if not isinstance(height_ft, (int, float)) or height_ft <= (1/12): # Min height? e.g. 1 inch
        msg = f"Invalid height '{height_ft}'. Must be > ~0.08 ft."
        log(f"Error: {msg}", queue_instance=log_queue)
        return {"error": msg, "logs": "\n".join(global_log_lines)}
    if not isinstance(target_ppfd, (int, float)) or target_ppfd <= 0:
        msg = f"Invalid target PPFD '{target_ppfd}'. Must be > 0."
        log(f"Error: {msg}", queue_instance=log_queue)
        return {"error": msg, "logs": "\n".join(global_log_lines)}

    # Check floor size mapping (uses helper now)
    floor_size_m = get_floor_size(layer, log_queue)
    log(f"Using floor size: {floor_size_m:.2f}m for requested layer {layer}", queue_instance=log_queue)

    height_m = height_ft * 0.3048
    input_vec = normalize_input(layer, height_m, target_ppfd, queue_instance=log_queue)
    input_vec_np = np.array(input_vec)
    log(f"Normalized Input Vector: {np.array2string(input_vec_np, precision=3)}", queue_instance=log_queue)

    # --- Load Data & Train Model ---
    X_data, Y_data = load_dataset(queue_instance=log_queue)
    log(f"Dataset contains {len(X_data)} valid samples.", queue_instance=log_queue)
    model = None # Define model outside the block
    is_model_trained = False
    if len(X_data) >= MIN_SAMPLES_FOR_TRAINING:
        log(f"Sufficient data ({len(X_data)} >= {MIN_SAMPLES_FOR_TRAINING}). Training model...", queue_instance=log_queue)
        train_start_time = time.time()
        try:
            if len(X_data) == len(Y_data): # Check consistency again
                X_np = np.array(X_data)
                Y_padded = []
                for y_vec in Y_data:
                    pad_width = MAX_LAYERS - len(y_vec)
                    if pad_width >= 0:
                        Y_padded.append(np.pad(y_vec, (0, pad_width), constant_values=0))
                    else:
                         Y_padded.append(y_vec[:MAX_LAYERS])
                         log(f"Warning: Y vector longer than MAX_LAYERS encountered during training prep.", queue_instance=log_queue)

                if Y_padded:
                    Y_np = np.array(Y_padded)
                    if X_np.shape[0] == Y_np.shape[0] and Y_np.shape[1] == MAX_LAYERS:
                        model = MultiOutputRegressor(
                            RandomForestRegressor(
                                n_estimators=RF_ESTIMATORS, random_state=RF_RANDOM_STATE,
                                n_jobs=RF_N_JOBS, max_depth=15,
                                min_samples_split=5, min_samples_leaf=3
                            )
                        )
                        model.fit(X_np, Y_np)
                        is_model_trained = True
                        train_duration = time.time() - train_start_time
                        log(f"Model trained successfully. Duration: {train_duration:.2f}s", queue_instance=log_queue)
                    else:
                         log(f"Error: Shape mismatch X({X_np.shape}) vs Y({Y_np.shape}) or Y pad error. Skipping training.", queue_instance=log_queue)
                else:
                    log("Error: No valid Y data available for training.", queue_instance=log_queue)
            else:
                 log("Error: X/Y length mismatch before training. Skipping.", queue_instance=log_queue)
        except Exception as e:
            log(f"Error during model training: {type(e).__name__}: {e}", queue_instance=log_queue)
            is_model_trained = False
    else:
        log("Not enough data to train model.", queue_instance=log_queue)

    # --- Determine Initial Guess ---
    initial_shape = None
    guess_source = "Unknown"
    if is_model_trained and model is not None:
        try:
            log("Attempting prediction with trained model...", queue_instance=log_queue)
            pred_padded = model.predict(input_vec_np.reshape(1, -1))[0]
            pred = pred_padded[:layer]
            pred = np.abs(pred)
            if np.sum(pred) > 1e-9:
                initial_shape = pred / np.sum(pred)
                log("Using model prediction as initial guess.", queue_instance=log_queue)
                guess_source = "Model"
            else:
                 log("Model prediction sum too low. Falling back.", queue_instance=log_queue)
        except Exception as e:
            log(f"Model prediction failed: {type(e).__name__}: {e}. Falling back.", queue_instance=log_queue)

    if initial_shape is None:
        log("Using fallback profile as initial guess.", queue_instance=log_queue)
        initial_shape = get_fallback_profile(layer, queue_instance=log_queue)
        guess_source = "Fallback"
        if len(initial_shape) != layer:
             msg = f"Failed to get initial guess for layer {layer}"
             log(f"Error: {msg}", queue_instance=log_queue)
             return {"error": msg, "logs": "\n".join(global_log_lines)}

    # --- Run Optimization ---
    # <<< FIX: Change variable name from 'loss' to 'loss_raw' >>>
    optimized_shape, loss_raw, final_ppfd, final_mdou = None, HIGH_LOSS, -1.0, -1.0 # Initialize defaults
    try:
        # Call optimize_flux and unpack results into correct variables
        optimized_shape, loss_raw, final_ppfd, final_mdou = optimize_flux(
            initial_shape, layer, target_ppfd, height_m, queue_instance=log_queue
        )

        # Check the RAW loss returned by optimize_flux
        if not np.isfinite(loss_raw) or loss_raw >= HIGH_LOSS:
             log("Optimization resulted in high or invalid raw loss. Scenario considered failed.", queue_instance=log_queue)
             return {
                 "error": "Optimization failed (high/invalid final raw loss)",
                 "final_ppfd": final_ppfd, "mDOU": final_mdou, "loss": loss_raw, # Report raw loss
                 "logs": "\n".join(global_log_lines)
             }
        # <<< FIX: Use loss_raw in the completion log message >>>
        log(f"Optimization Complete: RawLoss={loss_raw:.3f}, PPFD={final_ppfd:.1f}, mDOU={final_mdou:.1f}%", queue_instance=log_queue)

    except Exception as e:
        log(f"--- Unhandled Error during Optimization Call ---", queue_instance=log_queue)
        log(f"{type(e).__name__}: {e}", queue_instance=log_queue)
        import traceback
        log(traceback.format_exc(), queue_instance=log_queue)
        return {
            "error": f"Optimization function crashed: {e}",
            # Use initialized defaults if crash occurred before assignment
            "final_ppfd": final_ppfd, "mDOU": final_mdou, "loss": loss_raw,
            "logs": "\n".join(global_log_lines)
        }

    # --- Process Result (Save/Update Dataset) ---
    action = "Skipped"
    reason = "No suitable action determined"
    should_save_dataset = False

    # Find if existing data point is close enough
    found_idx = -1
    min_dist = float('inf')
    closest_idx = -1
    for i, x in enumerate(X_data):
        dist = np.linalg.norm(np.array(x) - input_vec_np)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
        if np.allclose(np.array(x), input_vec_np, atol=1e-4):
            found_idx = i
            break

    log(f"Searching for existing data point (tolerance 1e-4). Closest dist: {min_dist:.3g} at index {closest_idx}.", queue_instance=log_queue)

    update_tolerance = OPTIMIZER_TOLERANCE

    if found_idx != -1:
        log(f"Found existing data point at index {found_idx}. Comparing results...", queue_instance=log_queue)
        try:
            existing_y_shape = Y_data[found_idx]
            if len(existing_y_shape) == layer:
                 existing_scaled = scale_flux_to_ppfd(existing_y_shape, layer, target_ppfd, height_m, queue_instance=log_queue)
                 if np.sum(existing_scaled) > 1e-9:
                      # <<< FIX: Use loss_raw for comparison >>>
                      old_loss_raw, old_ppfd, old_mdou = simulate_loss(existing_scaled, layer, target_ppfd, height_m, queue_instance=log_queue)
                      log(f" Comparison: New RawLoss={loss_raw:.3f} vs Old RawLoss={old_loss_raw:.3f}", queue_instance=log_queue)

                      # <<< FIX: Use loss_raw for comparison >>>
                      if np.isfinite(old_loss_raw) and loss_raw < old_loss_raw - update_tolerance:
                           Y_data[found_idx] = list(optimized_shape)
                           # <<< FIX: Use loss_raw in reason message >>>
                           action, reason = "Updated", f"Raw loss improved ({old_loss_raw:.3f} -> {loss_raw:.3f})"
                           should_save_dataset = True
                      else:
                           # <<< FIX: Use loss_raw in reason message >>>
                           action, reason = "Skipped", f"Existing result better/similar (RawLoss New: {loss_raw:.3f}, Old: {old_loss_raw:.3f})"
                 else:
                      log("Warning: Could not scale existing shape vector for comparison. Updating entry.", queue_instance=log_queue)
                      Y_data[found_idx] = list(optimized_shape)
                      action, reason = "Updated", "Could not evaluate old raw loss, replaced."
                      should_save_dataset = True
            else:
                 log(f"Warning: Length mismatch for existing Y data index {found_idx} ({len(existing_y_shape)} vs {layer}). Overwriting.", queue_instance=log_queue)
                 Y_data[found_idx] = list(optimized_shape)
                 action, reason = "Updated", "Corrected length mismatch"
                 should_save_dataset = True
        except Exception as e:
             log(f"Error comparing with existing data: {type(e).__name__}: {e}. Skipping comparison.", queue_instance=log_queue)
             action, reason = "Skipped", "Error during comparison"
    else:
        log("No exact match found. Adding as new data point.", queue_instance=log_queue)
        X_data.append(input_vec)
        Y_data.append(list(optimized_shape)) # Store list
        action, reason = "Added", "New scenario result"
        should_save_dataset = True

    # --- Log Outcome & Save ---
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "layer": layer, "height_ft": height_ft, "target_ppfd": target_ppfd,
        "final_ppfd": float(final_ppfd) if np.isfinite(final_ppfd) else -1.0,
        "mDOU": float(final_mdou) if np.isfinite(final_mdou) else -1.0,
        # <<< FIX: Use loss_raw in log entry >>>
        "loss": float(loss_raw) if np.isfinite(loss_raw) else HIGH_LOSS,
        "action": action, "reason": reason,
        "guess_source": guess_source, "optimizer": OPTIMIZER_METHOD
    }
    log_result(log_entry, queue_instance=log_queue)

    if should_save_dataset:
        save_dataset(X_data, Y_data, queue_instance=log_queue)
        log(f"Dataset Action: {action}. Dataset saved.", queue_instance=log_queue)
    else:
        log(f"Dataset Action: {action}. Dataset not modified.", queue_instance=log_queue)

    run_duration = time.time() - run_start_time
    log(f"--- Scenario Finished. Duration: {run_duration:.2f}s ---", queue_instance=log_queue)

    # <<< FIX: Return loss_raw >>>
    return {
        "action": action, "reason": reason,
        "final_ppfd": final_ppfd, "mDOU": final_mdou, "loss": loss_raw,
        "error": None, # Explicitly set error to None on success
        "logs": "\n".join(global_log_lines)
    }

# -------------------------
# Main Execution (for local testing/batch processing)
# -------------------------
if __name__ == "__main__":
    # Use standard print for standalone runs (log function handles this)
    log("--- Running Standalone Batch Processing ---")

    # Initial dataset load
    X_data_main, Y_data_main = load_dataset() # Logs to console via log()

    # --- Parameter Sweep ---
    num_processed = 0
    results_summary = []
    batch_start_time = time.time()

    total_scenarios = len(layer_counts_sweep) * len(heights_ft_sweep) * len(ppfd_targets_sweep)
    log(f"Total scenarios in sweep: {total_scenarios}")
    log(f"Layers: {layer_counts_sweep}")
    log(f"Heights (ft): {heights_ft_sweep}")
    log(f"PPFD Targets: {ppfd_targets_sweep}")

    for layer in layer_counts_sweep:
        for height_ft in heights_ft_sweep:
            for target_ppfd in ppfd_targets_sweep:
                num_processed += 1
                progress = f"({num_processed}/{total_scenarios})"
                log(f"\n{progress} {datetime.now().strftime('%H:%M:%S')} | Processing: L={layer}, H={height_ft:.2f}ft, TgtPPFD={target_ppfd:.1f}")

                try:
                    # Call solve_single_scenario, which handles everything
                    # Pass log_queue=None for console logging
                    result = solve_single_scenario(layer, height_ft, target_ppfd, log_queue=None)
                    results_summary.append(result)

                    # Optional: Reload dataset periodically if ML model relies heavily on recent data
                    # if num_processed % 5 == 0:
                    #    log("--- Checkpoint: Reloading dataset ---")
                    #    X_data_main, Y_data_main = load_dataset()


                except Exception as e:
                    log(f"--- Catastrophic Failure in Batch Loop for L={layer}, H={height_ft}, P={target_ppfd} ---")
                    log(f"{type(e).__name__}: {e}")
                    import traceback
                    log(traceback.format_exc())
                    results_summary.append({"error": f"Crash: {e}", "action": "CRASH"})
                    # Log failure to CSV
                    log_result({
                        "timestamp": datetime.now().isoformat(), "layer": layer,
                        "height_ft": height_ft, "target_ppfd": target_ppfd,
                        "final_ppfd": -1, "mDOU": -1, "loss": -1,
                        "action": "CRASH", "reason": f"Unhandled exception: {e}"
                    })


    # --- Final Summary ---
    batch_duration = time.time() - batch_start_time
    log("\n--- Batch Processing Summary ---")
    log(f"Total Duration: {batch_duration:.2f}s")
    log(f"Total Scenarios Processed: {num_processed}")

    actions = [r.get("action", "Error") for r in results_summary if isinstance(r, dict)]
    from collections import Counter
    action_counts = Counter(actions)

    log(f"Actions:")
    for action, count in action_counts.items():
         log(f"  - {action}: {count}")

    num_errors = action_counts.get("Error", 0) + action_counts.get("CRASH", 0)
    log(f"Total Errors/Crashes: {num_errors}")

    # Final dataset info
    X_data_final, Y_data_final = load_dataset()
    log(f"Final dataset size: {len(X_data_final)}")
    log("\n✅ Batch run completed.")
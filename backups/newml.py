# --- START OF FILE newml_corrected.py ---

import os
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

# Attempt to import simulation; handle if not found
try:
    from lighting_simulation_data_1 import prepare_geometry, simulate_lighting
except ImportError:
    print("ERROR: Could not import 'lighting_simulation_data_1'.")
    print("Make sure 'lighting_simulation_data_1.py' is in the same directory or accessible.")
    exit() # Stop if simulation code isn't available

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.optimize import minimize_scalar, minimize

# Suppress specific warnings if needed (e.g., from scipy optimization)
warnings.filterwarnings("ignore", message=".*Method Powell does not use gradient information.*")

# -------------------------
# CONFIGURATION
# -------------------------
DATASET_FILE = "flux_learning_dataset.npz"
LOG_FILE = "flux_optimization_log.csv"
MAX_LAYERS = 20 # Used for padding Y data for ML model
DOU_TARGET = 95
PPFD_TOLERANCE = 15  # µmol/m²/s (Informational, not directly used in core logic)
FLOOR_SIZES = {
    10: 6.10, 11: 6.74, 12: 7.38, 13: 8.02, 14: 8.66,
    15: 9.30, 16: 9.94, 17: 10.58, 18: 11.22, 19: 11.68, 20: 12.32
}

# Sweeping parameters
layer_counts = list(range(10, 21))
heights_ft = np.linspace(1, 10, 5) # Consider increasing num for denser coverage
ppfd_targets = np.linspace(600, 1500, 5) # Consider increasing num & adding lower values

# Optimization settings
OPTIMIZER_METHOD = 'Powell' # 'Powell' or 'Nelder-Mead' are often good choices here
OPTIMIZER_MAXITER = 200 # Increase if optimization seems to terminate too early
HIGH_LOSS = 1e12 # Define a constant for high loss value <<<--- ADD THIS LINE

# ML Settings
MIN_SAMPLES_FOR_TRAINING = 9 # Increased threshold for more reliable initial model
RF_ESTIMATORS = 100
RF_RANDOM_STATE = 42 # For reproducibility
RF_N_JOBS = -1 # Use all available CPU cores

# -------------------------
# Dataset Handling
# -------------------------
def load_dataset():
    """Loads the dataset (X, Y) from the NPZ file."""
    if os.path.exists(DATASET_FILE):
        try:
            # Allow pickle needed for object arrays (ragged Y)
            data = np.load(DATASET_FILE, allow_pickle=True)
            X_loaded = data["X"]
            Y_loaded = data["Y"]

            # Basic validation
            if len(X_loaded) != len(Y_loaded):
                 print(f"Warning: Mismatch in loaded data lengths. X: {len(X_loaded)}, Y: {len(Y_loaded)}. Resetting dataset.")
                 return [], []

            # Ensure data is in the expected list format
            X = [list(item) for item in X_loaded]
            Y = [list(item) for item in Y_loaded] # Y should be saved as list/array elements in an object array

            return X, Y
        except Exception as e:
            print(f"Error loading dataset '{DATASET_FILE}': {e}")
            print("Starting with an empty dataset.")
            return [], []
    return [], []

def save_dataset(X, Y):
    """Saves the dataset (X, Y) to the NPZ file."""
    try:
        # Save Y as an object array to handle variable lengths
        np.savez(DATASET_FILE, X=np.array(X, dtype=object), Y=np.array(Y, dtype=object))
    except Exception as e:
        print(f"Error saving dataset '{DATASET_FILE}': {e}")

def log_result(entry):
    """Appends a result entry to the CSV log file."""
    df = pd.DataFrame([entry])
    file_exists = os.path.exists(LOG_FILE)
    try:
        df.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)
    except Exception as e:
        print(f"Error writing to log file '{LOG_FILE}': {e}")

# -------------------------
# Helpers
# -------------------------
def normalize_input(layer, height_m, ppfd):
    """Normalizes input parameters to a range (approx 0-1)."""
    max_layer_num = max(layer_counts) if layer_counts else MAX_LAYERS
    max_height_m = 10 * 0.3048 # Max height 10ft in meters
    max_ppfd = 1500

    norm_layer = layer / max_layer_num if max_layer_num > 0 else 0
    norm_height = height_m / max_height_m if max_height_m > 0 else 0
    norm_ppfd = ppfd / max_ppfd if max_ppfd > 0 else 0

    return [norm_layer, norm_height, norm_ppfd]

def scale_flux_to_ppfd(shape_vector, layer_count, target_ppfd, height_m):
    """
    Scales a flux distribution shape to achieve the target average PPFD.

    Args:
        shape_vector (np.ndarray): The relative flux distribution (should sum to 1).
        layer_count (int): Number of active layers.
        target_ppfd (float): The desired average PPFD.
        height_m (float): Height in meters.

    Returns:
        np.ndarray: The scaled flux vector, or the original shape_vector if scaling fails.
    """
    if layer_count not in FLOOR_SIZES:
        print(f"Error: Invalid layer_count {layer_count} not in FLOOR_SIZES.")
        return np.array(shape_vector) # Return original shape on error

    size = FLOOR_SIZES[layer_count]
    try:
        geo = prepare_geometry(size, size, height_m)
    except Exception as e:
        print(f"Error preparing geometry: {e}")
        return np.array(shape_vector)

    shape_vector = np.array(shape_vector) # Ensure numpy array

    # Ensure shape is valid before scaling
    if np.any(shape_vector < 0):
        # print("Warning: Negative values in shape_vector before scaling. Using abs().")
        shape_vector = np.abs(shape_vector)
    sum_shape = np.sum(shape_vector)
    if sum_shape <= 1e-9: # Check for effectively zero sum
        # print("Warning: shape_vector sums to near zero. Cannot scale.")
        # Return a zero vector of the correct size, or handle as appropriate
        return np.zeros_like(shape_vector) # Or maybe return the (useless) input?

    # Normalize shape internally for consistent scaling calculation
    norm_shape = shape_vector / sum_shape

    def ppfd_error(scale):
        if scale <= 0: return 1e12 # Penalize non-positive scales heavily
        scaled = norm_shape * scale # Scale the normalized shape
        # Pad the scaled vector to MAX_LAYERS for the simulation function
        padded = np.pad(scaled, (0, MAX_LAYERS - layer_count))
        try:
            floor_ppfd, _, _, _ = simulate_lighting(padded, geo)
            mean_ppfd = np.mean(floor_ppfd)
            if not np.isfinite(mean_ppfd):
                # print(f"Warning: Simulation returned non-finite PPFD ({mean_ppfd}) for scale {scale}.")
                return 1e12 # High error if simulation yields NaN/inf
            return abs(mean_ppfd - target_ppfd)
        except Exception as e:
            # print(f"Error during simulation in ppfd_error for scaling: {e}")
            return 1e12 # High error if simulation fails

    # Optimize scaling factor
    try:
        res = minimize_scalar(ppfd_error, bounds=(1e-6, target_ppfd * MAX_LAYERS * 2), method='bounded', options={'xatol': 1e-3}) # Wider bounds for scale
        # Scale factor can be roughly proportional to target_ppfd * number_of_layers
        # Upper bound target_ppfd * MAX_LAYERS * 2 is a heuristic guess

        if res.success and res.x > 1e-6:
             final_scale = res.x
             # Apply the found scale to the original (potentially unnormalized) shape_vector
             # Or more consistently, apply to the normalized shape and then scale by the original sum if needed,
             # but since the goal is just the final scaled vector, scaling the normalized shape is fine.
             scaled_result = norm_shape * final_scale
             scaled_result[~np.isfinite(scaled_result)] = 0 # Clean up potential NaN/inf
             scaled_result = np.maximum(scaled_result, 0) # Ensure non-negative
             return scaled_result
        else:
            # print(f"Warning: Scaling optimization failed or resulted in near-zero scale (scale={res.x}).")
            # Fallback: Return the normalized shape scaled by target_ppfd as a rough guess?
            # Or just return the unscaled input shape? Returning unscaled seems safer.
             return shape_vector * 0 # Return zero if scaling fails drastically
    except Exception as e:
        print(f"Error during scaling optimization: {e}")
        return shape_vector * 0 # Return zero vector on error


def simulate_loss(scaled_flux, layer_count, target_ppfd, height_m):
    """
    Simulates lighting for a scaled flux distribution and calculates a loss value.

    Args:
        scaled_flux (np.ndarray): The flux vector already scaled to the target PPFD.
        layer_count (int): Number of active layers.
        target_ppfd (float): The target average PPFD (used for loss calculation).
        height_m (float): Height in meters.

    Returns:
        tuple: (loss, mean_ppfd, m_dou)
               Returns (high_loss, -1, -1) on error.
    """
    HIGH_LOSS = 1e12 # Define a constant for high loss

    if layer_count not in FLOOR_SIZES:
        print(f"Error: Invalid layer_count {layer_count} not in FLOOR_SIZES.")
        return HIGH_LOSS, -1, -1

    # Ensure input is numpy array
    scaled_flux = np.array(scaled_flux)
    if len(scaled_flux) != layer_count:
         print(f"Error: scaled_flux length ({len(scaled_flux)}) != layer_count ({layer_count}).")
         return HIGH_LOSS, -1, -1


    size = FLOOR_SIZES[layer_count]
    try:
        geo = prepare_geometry(size, size, height_m)
    except Exception as e:
        print(f"Error preparing geometry: {e}")
        return HIGH_LOSS, -1, -1

    # Pad the scaled vector for the simulation
    padded_flux = np.pad(scaled_flux, (0, MAX_LAYERS - layer_count))

    try:
        floor_ppfd, _, _, _ = simulate_lighting(padded_flux, geo)

        # Validate simulation output
        if floor_ppfd is None or not hasattr(floor_ppfd, '__len__') or len(floor_ppfd) == 0:
            print("Warning: Simulation returned invalid floor_ppfd.")
            return HIGH_LOSS, -1, -1

        mean_ppfd = np.mean(floor_ppfd)

        # Handle non-finite or non-positive mean_ppfd
        if not np.isfinite(mean_ppfd):
            print(f"Warning: Simulation resulted in non-finite mean_ppfd ({mean_ppfd}).")
            # Try to calculate based on finite values only?
            finite_ppfd = floor_ppfd[np.isfinite(floor_ppfd)]
            if len(finite_ppfd) > 0:
                mean_ppfd = np.mean(finite_ppfd)
                if not np.isfinite(mean_ppfd): # Still non-finite? Give up.
                     return HIGH_LOSS, -1, -1
            else: # All values were non-finite
                 return HIGH_LOSS, -1, -1

        if mean_ppfd <= 1e-9: # Effectively zero or negative PPFD
            # print(f"Warning: Simulation resulted in near-zero or negative mean_ppfd ({mean_ppfd:.2f}).")
            # Loss calculation below will likely fail or be meaningless.
            # Return high loss, but report the (problematic) mean PPFD.
            return HIGH_LOSS, mean_ppfd, 0 # Assign 0 DOU

        # Calculate MAD and mDOU
        mad = np.mean(np.abs(floor_ppfd[np.isfinite(floor_ppfd)] - mean_ppfd)) # Use only finite values for MAD
        m_dou = 100 * (1 - mad / mean_ppfd)
        if not np.isfinite(m_dou):
             # print(f"Warning: Calculated non-finite mDOU (mad={mad}, mean_ppfd={mean_ppfd}). Setting DOU to 0.")
             m_dou = 0 # Assign 0 DOU if calculation fails

        # Calculate Loss = PPFD error + DOU penalty
        ppfd_term = abs(mean_ppfd - target_ppfd)
        dou_penalty = 10 * max(0, DOU_TARGET - m_dou) # Penalize DOU below target
        loss = ppfd_term + dou_penalty

        if not np.isfinite(loss):
             print(f"Warning: Calculated non-finite loss (PPFD Term: {ppfd_term}, DOU Penalty: {dou_penalty}).")
             loss = HIGH_LOSS # Assign high loss

        return loss, mean_ppfd, m_dou

    except Exception as e:
        print(f"Error during simulation or loss calculation: {e}")
        return HIGH_LOSS, -1, -1 # High loss, invalid PPFD/DOU on error


def optimize_flux(initial_shape_guess, layer_count, target_ppfd, height_m):
    """
    Optimizes the flux distribution shape starting from an initial guess.

    Args:
        initial_shape_guess (np.ndarray): Normalized initial guess for the shape.
        layer_count (int): Number of active layers.
        target_ppfd (float): Target average PPFD.
        height_m (float): Height in meters.

    Returns:
        tuple: (optimized_shape, loss, mean_ppfd, m_dou)
               Returns (initial_guess, high_loss, -1, -1) on failure.
    """
    HIGH_LOSS = 1e12
    initial_shape_guess = np.array(initial_shape_guess) # Ensure numpy

    # Validate initial guess
    if len(initial_shape_guess) != layer_count:
        print(f"Warning: Initial guess length ({len(initial_shape_guess)}) != layer_count ({layer_count}). Using uniform.")
        initial_shape_guess = np.ones(layer_count) / layer_count
    else:
        initial_shape_guess = np.abs(initial_shape_guess) # Ensure non-negative
        sum_initial = np.sum(initial_shape_guess)
        if sum_initial <= 1e-9:
            # print("Warning: Initial guess sums to near zero. Using uniform.")
            initial_shape_guess = np.ones(layer_count) / layer_count
        else:
            initial_shape_guess = initial_shape_guess / sum_initial # Normalize

    # --- Loss function for the optimizer ---
    memo = {} # Simple memoization for loss function if needed (optional)
    def loss_fn(vec):
        vec_tuple = tuple(vec)
        # if vec_tuple in memo: return memo[vec_tuple] # Uncomment for memoization

        # Ensure vector is valid within the loss function
        vec = np.abs(vec) # Ensure non-negative during optimization
        sum_vec = np.sum(vec)
        if sum_vec <= 1e-9:
            return HIGH_LOSS # Avoid division by zero, high loss

        normed_shape = vec / sum_vec

        # --- Simulation Core ---
        # 1. Scale the normalized shape to the target PPFD
        scaled_flux_vec = scale_flux_to_ppfd(normed_shape, layer_count, target_ppfd, height_m)

        # 2. Calculate loss based on the simulation of the scaled flux
        loss_val, _, _ = simulate_loss(scaled_flux_vec, layer_count, target_ppfd, height_m)
        # --- End Simulation Core ---

        # memo[vec_tuple] = loss_val # Uncomment for memoization
        return loss_val
    # --- End loss function ---

    # --- Run Minimization ---
    optimized_shape = initial_shape_guess # Default to initial if fails
    final_loss = HIGH_LOSS
    final_mean_ppfd = -1
    final_m_dou = -1

    try:
        # Powell and Nelder-Mead are generally good choices for derivative-free optimization
        res = minimize(loss_fn, initial_shape_guess, method=OPTIMIZER_METHOD,
                       options={'maxiter': OPTIMIZER_MAXITER, 'disp': False}) # disp=True for debug

        if res.success:
            optimized_res_shape = np.abs(res.x)
            sum_opt = np.sum(optimized_res_shape)
            if sum_opt <= 1e-9: # Handle case where optimization results in zero vector
                 print("Warning: Optimization resulted in near-zero vector. Using initial guess.")
                 # Keep optimized_shape as initial_shape_guess
            else:
                 optimized_shape = optimized_res_shape / sum_opt # Final normalization
                 # print("Optimization successful.") # Debug
        else:
            print(f"Warning: Optimization ({OPTIMIZER_METHOD}) failed to converge.")
            # Keep optimized_shape as initial_shape_guess

    except Exception as e:
        print(f"Error during optimization process: {e}")
        # Keep optimized_shape as initial_shape_guess

    # --- Final Evaluation ---
    # Recalculate performance metrics with the *chosen* shape (optimized or fallback)
    try:
        final_scaled_flux = scale_flux_to_ppfd(optimized_shape, layer_count, target_ppfd, height_m)
        final_loss, final_mean_ppfd, final_m_dou = simulate_loss(final_scaled_flux, layer_count, target_ppfd, height_m)
    except Exception as e:
        print(f"Error during final evaluation after optimization: {e}")
        # Values remain at defaults (high loss, -1, -1)

    return optimized_shape, final_loss, final_mean_ppfd, final_m_dou


# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    # Load existing data
    X_data, Y_data = load_dataset()
    print(f"📦 Loaded dataset with {len(X_data)} samples.")

    # Initialize and train the model
    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=RF_ESTIMATORS,
            random_state=RF_RANDOM_STATE,
            n_jobs=RF_N_JOBS
        )
    )
    is_model_trained = False
    if len(X_data) >= MIN_SAMPLES_FOR_TRAINING:
        print(f"🧠 Training model on {len(X_data)} samples...")
        try:
            # Prepare data for scikit-learn
            X_np = np.array(X_data)

            # Pad Y data to MAX_LAYERS for MultiOutputRegressor
            Y_padded = []
            for y_vec in Y_data:
                pad_width = MAX_LAYERS - len(y_vec)
                if pad_width < 0:
                    print(f"Warning: Y vector longer ({len(y_vec)}) than MAX_LAYERS ({MAX_LAYERS}). Truncating.")
                    y_padded = np.array(y_vec[:MAX_LAYERS])
                elif pad_width == 0:
                     y_padded = np.array(y_vec)
                else:
                     y_padded = np.pad(y_vec, (0, pad_width), 'constant', constant_values=0)
                Y_padded.append(y_padded)
            Y_np = np.array(Y_padded) # Now a rectangular array (N_samples x MAX_LAYERS)

            # Check shapes before fitting
            if X_np.shape[0] != Y_np.shape[0]:
                 print(f"Error: Shape mismatch between X ({X_np.shape}) and Y ({Y_np.shape}) before training. Skipping training.")
            elif Y_np.shape[1] != MAX_LAYERS:
                 print(f"Error: Padded Y shape[1] ({Y_np.shape[1]}) does not match MAX_LAYERS ({MAX_LAYERS}). Skipping training.")
            else:
                 model.fit(X_np, Y_np)
                 is_model_trained = True
                 print("✅ Model trained.")

        except Exception as e:
            print(f"❌ Error during model training: {e}")
            is_model_trained = False # Ensure model is not used if training failed
    else:
        print(f"⚠️ Not enough data ({len(X_data)} < {MIN_SAMPLES_FOR_TRAINING}) to train model. Using uniform initial guess.")

    # --- Parameter Sweep ---
    num_processed = 0
    num_updated = 0
    num_added = 0
    num_skipped = 0
    num_errors = 0

    print("\n--- Starting Parameter Sweep ---")
    for layer in layer_counts:
        for height_ft in heights_ft:
            for target_ppfd in ppfd_targets:
                num_processed += 1
                height_m = height_ft * 0.3048
                input_vec = normalize_input(layer, height_m, target_ppfd)
                input_vec_np = np.array(input_vec) # Use numpy array for comparisons

                print(f"\n🔍 {datetime.now().strftime('%H:%M:%S')} | Processing: L={layer}, H={height_ft:.2f}ft, TgtPPFD={target_ppfd:.1f}")

                # --- Get Initial Guess ---
                initial_shape_guess = np.ones(layer) / layer # Default: uniform distribution
                if is_model_trained:
                    try:
                        # Model predicts padded vector (length MAX_LAYERS)
                        pred_padded = model.predict([input_vec])[0]
                        # Take only the first 'layer' elements
                        pred_shape = pred_padded[:layer]
                        pred_shape = np.abs(pred_shape) # Ensure non-negative
                        sum_pred = np.sum(pred_shape)
                        if sum_pred > 1e-9:
                           initial_shape_guess = pred_shape / sum_pred # Normalize valid prediction
                           # print("ℹ️ Using model prediction as initial guess.") # Debug
                        # else: print("⚠️ Model prediction sum is near zero, using uniform guess.") # Debug
                    except Exception as e:
                         print(f"⚠️ Error during model prediction: {e}. Using uniform guess.")
                # else: print("ℹ️ Model not trained, using uniform guess.") # Debug

                # --- Optimization ---
                try:
                    optimized_shape, loss, final_ppfd, final_mdou = optimize_flux(
                        initial_shape_guess, layer, target_ppfd, height_m
                    )

                    # Check if optimization produced valid results
                    if not np.isfinite(loss) or loss >= HIGH_LOSS:
                         print(f"❌ Error or High Loss ({loss:.2f}) after optimization. Skipping saving.")
                         num_errors += 1
                         log_result({
                            "timestamp": datetime.now().isoformat(), "layer": layer,
                            "height_ft": height_ft, "target_ppfd": target_ppfd,
                            "final_ppfd": final_ppfd, "mDOU": final_mdou, "loss": loss,
                            "action": "Error", "reason": "Optimization failed or resulted in very high loss"
                         })
                         continue # Skip to next iteration

                    # --- Check if exists and update/add ---
                    found_existing_idx = -1
                    for i, x_existing in enumerate(X_data):
                        # Compare normalized input vectors using numpy's allclose
                        if np.allclose(np.array(x_existing), input_vec_np, atol=1e-6):
                           found_existing_idx = i
                           break

                    should_save_or_update = False
                    action = "Skipped"
                    reason = ""
                    update_tolerance = 1e-3 # Only update if new loss is significantly better

                    if found_existing_idx != -1:
                        # Calculate loss of the *existing* solution for fair comparison
                        existing_y_shape = Y_data[found_existing_idx]
                        if len(existing_y_shape) != layer:
                             print(f"Warning: Length mismatch for existing data point {found_existing_idx}. Skipping comparison.")
                             reason = "Existing data length mismatch"
                             num_skipped += 1
                        else:
                            # Need to scale the existing solution to the *current* target_ppfd and recalculate its loss
                            scaled_existing = scale_flux_to_ppfd(existing_y_shape, layer, target_ppfd, height_m)
                            prev_loss, prev_ppfd, prev_mdou = simulate_loss(scaled_existing, layer, target_ppfd, height_m)

                            if loss < prev_loss - update_tolerance:
                               # Update with the new, better shape (store as list)
                               Y_data[found_existing_idx] = list(optimized_shape)
                               should_save_or_update = True
                               action = "Updated"
                               reason = f"Improved (Loss: {prev_loss:.2f} -> {loss:.2f})"
                               num_updated += 1
                            else:
                               should_save_or_update = False # Keep existing
                               action = "Skipped"
                               reason = f"Existing better/equal (Current Loss: {loss:.2f}, Prev Loss: {prev_loss:.2f})"
                               num_skipped += 1
                    else:
                        # Add new entry (store shape as list)
                        X_data.append(input_vec)
                        Y_data.append(list(optimized_shape))
                        should_save_or_update = True
                        action = "Added"
                        reason = "New entry"
                        num_added += 1

                    # --- Log result ---
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "layer": layer,
                        "height_ft": height_ft,
                        "target_ppfd": target_ppfd,
                        "final_ppfd": final_ppfd,
                        "mDOU": final_mdou,
                        "loss": loss,
                        "action": action,
                        "reason": reason
                    }
                    log_result(log_entry)

                    if should_save_or_update:
                        print(f"✅ {action}. {reason}. Saved. PPFD={final_ppfd:.1f}, mDOU={final_mdou:.2f}%")
                    else:
                        print(f"⏭ {action}. {reason}. PPFD={final_ppfd:.1f}, mDOU={final_mdou:.2f}%")

                except Exception as e:
                    num_errors += 1
                    print(f"❌❌❌ Unhandled error during processing L={layer}, H={height_ft}, P={target_ppfd}: {e}")
                    # Log the error
                    log_result({
                        "timestamp": datetime.now().isoformat(), "layer": layer,
                        "height_ft": height_ft, "target_ppfd": target_ppfd,
                        "final_ppfd": -1, "mDOU": -1, "loss": -1,
                        "action": "Error", "reason": f"Unhandled exception: {e}"
                    })
                    # Potentially add a traceback here if needed for debugging
                    # import traceback
                    # traceback.print_exc()

    # --- End of Parameter Sweep ---
    print("\n--- Parameter Sweep Summary ---")
    print(f"Processed: {num_processed}, Added: {num_added}, Updated: {num_updated}, Skipped: {num_skipped}, Errors: {num_errors}")
    print(f"Total dataset size: {len(X_data)}")

    # Save the final dataset
    print(f"\n💾 Saving final dataset ({len(X_data)} entries)...")
    save_dataset(X_data, Y_data)
    if num_processed % 10 == 0:
        print(f"Checkpoint: Saving dataset at {num_processed} iterations...")
        save_dataset(X_data, Y_data)
    print("✅ Dataset saved.")
    print("\n✅ Batch run completed.")

# --- END OF FILE newml_corrected.py ---
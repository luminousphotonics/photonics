print(">>> Script started", flush=True)

import sys
try:
    print(">>> sys.argv:", sys.argv, flush=True)
except Exception as e:
    print(f">>> Early crash: {e}", flush=True)

import os
import numpy as np
import argparse
from datetime import datetime
from newml import (
    load_dataset, save_dataset, log_result, normalize_input,
    optimize_flux, scale_flux_to_ppfd, simulate_loss, MAX_LAYERS,
    MIN_SAMPLES_FOR_TRAINING, MultiOutputRegressor, RandomForestRegressor,
    RF_ESTIMATORS, RF_RANDOM_STATE, RF_N_JOBS
)

def run_single_scenario(layer, height_ft, target_ppfd):
    height_m = height_ft * 0.3048
    input_vec = normalize_input(layer, height_m, target_ppfd)
    input_vec_np = np.array(input_vec)

    # Load dataset
    X_data, Y_data = load_dataset()
    print(f"Loaded dataset with {len(X_data)} samples.")

    # Train model if enough data
    is_model_trained = False
    if len(X_data) >= MIN_SAMPLES_FOR_TRAINING:
        model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=RF_ESTIMATORS,
                random_state=RF_RANDOM_STATE,
                n_jobs=RF_N_JOBS
            )
        )
        X_np = np.array(X_data)
        Y_padded = [
            np.pad(y, (0, MAX_LAYERS - len(y)), constant_values=0)
            for y in Y_data
        ]
        Y_np = np.array(Y_padded)
        model.fit(X_np, Y_np)
        is_model_trained = True
    else:
        model = None

    # Initial guess
    initial_shape = np.ones(layer) / layer
    if is_model_trained:
        try:
            pred = model.predict([input_vec])[0][:layer]
            pred = np.abs(pred)
            if np.sum(pred) > 1e-9:
                initial_shape = pred / np.sum(pred)
        except Exception as e:
            print(f"Model prediction failed: {e}")

    # Optimize
    optimized_shape, loss, final_ppfd, final_mdou = optimize_flux(
        initial_shape, layer, target_ppfd, height_m
    )

    # Determine save/update
    action, reason = "Skipped", "No update necessary"
    should_save = False
    found_idx = -1
    for i, x in enumerate(X_data):
        if np.allclose(np.array(x), input_vec_np, atol=1e-6):
            found_idx = i
            break

    if found_idx != -1:
        old_scaled = scale_flux_to_ppfd(Y_data[found_idx], layer, target_ppfd, height_m)
        old_loss, _, _ = simulate_loss(old_scaled, layer, target_ppfd, height_m)
        if loss < old_loss - 1e-3:
            Y_data[found_idx] = list(optimized_shape)
            action, reason = "Updated", f"Loss improved ({old_loss:.2f} → {loss:.2f})"
            should_save = True
    else:
        X_data.append(input_vec)
        Y_data.append(list(optimized_shape))
        action, reason = "Added", "New entry"
        should_save = True

    log_result({
        "timestamp": datetime.now().isoformat(),
        "layer": layer,
        "height_ft": height_ft,
        "target_ppfd": target_ppfd,
        "final_ppfd": final_ppfd,
        "mDOU": final_mdou,
        "loss": loss,
        "action": action,
        "reason": reason
    })

    if should_save:
        save_dataset(X_data, Y_data)

    return {
        "action": action,
        "reason": reason,
        "final_ppfd": final_ppfd,
        "mDOU": final_mdou,
        "loss": loss
    }
print(">>> Entering __main__", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--ppfd", type=float, required=True)
    args = parser.parse_args()

    result = run_single_scenario(args.layer, args.height, args.ppfd)

    # Output key:value pairs (no extra formatting)
    for k, v in result.items():
        print(f"{k}: {v}")


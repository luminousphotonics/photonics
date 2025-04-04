# grow_builder/light_calculator.py

import math
import numpy as np
import logging
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)

# --- Constants ---
# You can still set a default N here if desired, but it can be passed in.
# Defaulting to 6 as per the previous focus.
DEFAULT_NUM_LAYERS = 6

# --- Core Calculation Logic ---

def _calculate_cob_and_strip_data(W: float, L: float, target_H: float, num_total_layers: int) -> Dict[str, Any]:
    """
    Generates COB positions and ordered strip coordinates for visualization.

    Args:
        W: Room width.
        L: Room length.
        target_H: Desired height for COBs/Strips.
        num_total_layers: The number of layers (N value, e.g., 6 for 61 points).

    Returns:
        A dictionary containing:
        - 'cob_positions': List of {'x','y','z','id','layer'} for each COB.
        - 'strip_layer_coordinates': List of lists, where each inner list
          contains {'x','y','z'} points for one layer's strip in order.
        - 'verification': Basic verification info.
    """
    logger.info(f"Calculating COB/Strip data for N={num_total_layers}, W={W:.2f}, L={L:.2f}, H={target_H:.2f}")

    cob_positions = []
    strip_layer_coordinates = [] # List[List[Dict]]
    verification = {
        'requested_N_layers': num_total_layers,
        'generated_N_cobs': 0,
        'expected_N_cobs': 0,
        'match_status': 'ERROR', # Default status
        'error': None
    }

    n = num_total_layers - 1 # Max layer index (0 to n)

    # --- Calculate Expected COB Count ---
    # Formula for centered square numbers variant: 2*n*(n+1)+1
    if n >= 0:
        expected_cobs = 2 * n * (n + 1) + 1
        verification['expected_N_cobs'] = expected_cobs
    else:
        logger.error("Number of layers must be at least 1.")
        verification['error'] = "Invalid number of layers (must be >= 1)."
        return {'cob_positions': [], 'strip_layer_coordinates': [], 'verification': verification}

    # --- Generate Base Diamond Coordinates ---
    base_positions_with_layer = []
    if n >= 0:
        base_positions_with_layer.append({'x': 0, 'y': 0, 'layer': 0}) # Center point
        for i in range(1, n + 1): # Layers 1 to n
            for x in range(-i, i + 1):
                y_abs = i - abs(x)
                # Only add points on the exact perimeter diamond
                if abs(x) + y_abs == i:
                    if y_abs == 0:
                        base_positions_with_layer.append({'x': x, 'y': 0, 'layer': i})
                    else:
                        base_positions_with_layer.append({'x': x, 'y': y_abs, 'layer': i})
                        base_positions_with_layer.append({'x': x, 'y': -y_abs, 'layer': i})

    # --- Deduplicate and Sort Base Positions (Ensures correct count) ---
    seen_base_coords = set()
    unique_base_positions = []
    for pos in base_positions_with_layer:
        coord_tuple = (pos['x'], pos['y'])
        if coord_tuple not in seen_base_coords:
            unique_base_positions.append(pos)
            seen_base_coords.add(coord_tuple)
    unique_base_positions.sort(key=lambda p: (p['layer'], p['x'], p['y'])) # Consistent sort

    generated_base_count = len(unique_base_positions)
    if generated_base_count != expected_cobs:
        logger.error(f"Mismatch in generated base points ({generated_base_count}) vs expected ({expected_cobs}). Check diamond logic.")
        verification['error'] = f"Internal error: Base point count mismatch ({generated_base_count} vs {expected_cobs})."
        # Continue calculation but verification will fail

    # --- Calculate Transformation Parameters ---
    theta = math.radians(45)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    center_x, center_y = W / 2.0, L / 2.0
    # Avoid division by zero if n=0 (num_total_layers=1)
    scale_x = W / (n * math.sqrt(2)) if n > 0 else W / 2.0
    scale_y = L / (n * math.sqrt(2)) if n > 0 else L / 2.0

    # --- Transform COB Positions ---
    for idx, pos in enumerate(unique_base_positions):
        bx, by = pos['x'], pos['y']
        rx = bx * cos_t - by * sin_t
        ry = bx * sin_t + by * cos_t
        px = center_x + rx * scale_x
        py = center_y + ry * scale_y
        cob_positions.append({
            'x': px, 'y': py, 'z': target_H, # Use target_H directly
            'id': idx, 'layer': pos['layer']
        })

    verification['generated_N_cobs'] = len(cob_positions)

    # --- Generate Ordered Strip Coordinates for each layer > 0 ---
    for layer_idx in range(1, num_total_layers): # Strips for layers 1 to n
        i = layer_idx
        ordered_abstract_coords = []

        # Generate abstract coordinates in clockwise order (same as visualization script)
        for x in range(i, 0, -1): ordered_abstract_coords.append((x, i - x)) # Top-Right -> Q1 -> Q2
        for x in range(0, -i, -1): ordered_abstract_coords.append((x, i + x)) # Top-Left -> Q2 -> Q3
        for x in range(-i, 0, 1): ordered_abstract_coords.append((x, -i - x)) # Bottom-Left -> Q3 -> Q4
        for x in range(0, i, 1): ordered_abstract_coords.append((x, -i + x)) # Bottom-Right -> Q4 -> Q1

        # Apply the same transformation
        current_layer_strip_coords = []
        for ax, ay in ordered_abstract_coords:
            rx = ax * cos_t - ay * sin_t
            ry = ax * sin_t + ay * cos_t
            px = center_x + rx * scale_x
            py = center_y + ry * scale_y
            current_layer_strip_coords.append({'x': px, 'y': py, 'z': target_H})

        strip_layer_coordinates.append(current_layer_strip_coords)

    # --- Final Verification Status ---
    if verification['generated_N_cobs'] == verification['expected_N_cobs']:
        verification['match_status'] = 'PASSED'
        logger.info(f"Calculation PASSED: Generated {verification['generated_N_cobs']} COBs as expected for N={num_total_layers}.")
    else:
        verification['match_status'] = 'ERROR'
        if not verification['error']: # Add error if not already set
             verification['error'] = "Generated COB count does not match expected count."
        logger.error(f"Calculation ERROR: Generated {verification['generated_N_cobs']} COBs, expected {verification['expected_N_cobs']}.")

    return {
        'cob_positions': cob_positions,
        'strip_layer_coordinates': strip_layer_coordinates,
        'verification': verification
    }


# --- Main Entry Point ---
def get_layout_and_targets(W: float, L: float, H_ceiling: float, light_h: float) -> Dict[str, Any]:
    """
    Calculates COB positions and strip coordinates based on dimensions and layer count.
    Modules are no longer calculated.
    """
    # Use the globally set default number of layers
    num_layers_to_calc = DEFAULT_NUM_LAYERS

    # Call the core calculation function
    calc_result = _calculate_cob_and_strip_data(W, L, light_h, num_layers_to_calc)

    # Structure the output for the view/frontend
    cob_positions = calc_result['cob_positions']
    verification_info = calc_result['verification']

    # For compatibility, make target_cob_positions just the x,y,z of placed ones
    target_cobs_output = [{'x': p['x'], 'y': p['y'], 'z': p['z']} for p in cob_positions]

    return {
        'modules': [], # No modules calculated anymore
        'placed_cob_positions': cob_positions, # Contains full info {'x','y','z','id','layer'}
        'target_cob_positions': target_cobs_output, # Simple list of {'x','y','z'}
        'strip_layer_coordinates': calc_result['strip_layer_coordinates'], # New key for strip data
        'verification': verification_info
        # 'plant_positions': calculate_plant_positions_grid(W, L) # Keep if needed by view
    }


# --- Plant Position Calculation (Keep as is) ---
def calculate_plant_positions_grid(W, L, spacing=1.0):
    """Calculates plant positions on a simple grid."""
    positions = []
    if W <= 0 or L <= 0 or spacing <= 0: return positions
    nx = max(1, int(W / spacing)); ny = max(1, int(L / spacing))
    actual_spacing_x = W / (nx + 1); actual_spacing_y = L / (ny + 1)
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            positions.append([i * actual_spacing_x, j * actual_spacing_y, 0.0])
    logger.info(f"Generated {len(positions)} plant positions with ~{spacing} spacing.")
    return positions

# --- Example Usage ---
if __name__ == "__main__":
    print(f"Testing light_calculator (COBs + Strips, N={DEFAULT_NUM_LAYERS})...")
    test_W, test_L, test_H_ceil, test_light_h = 10.0, 10.0, 8.0, 7.0

    layout_result = get_layout_and_targets(test_W, test_L, test_H_ceil, test_light_h)

    print("\n--- Layout Result ---")
    print(f"Modules Placed: {len(layout_result.get('modules', []))}") # Should be 0
    print(f"Target COBs (Count): {len(layout_result.get('target_cob_positions', []))}")
    print(f"Placed COBs (Count): {len(layout_result.get('placed_cob_positions', []))}")
    print(f"Strip Layers Generated: {len(layout_result.get('strip_layer_coordinates', []))}")
    if layout_result.get('strip_layer_coordinates'):
        print(f"  Points in first strip layer: {len(layout_result['strip_layer_coordinates'][0])}")

    verification = layout_result.get('verification', {})
    print(f"Verification: Status='{verification.get('match_status')}', Error='{verification.get('error')}', "
          f"ExpectedN={verification.get('expected_N_cobs')}, GeneratedN={verification.get('generated_N_cobs')}")

    if verification.get('match_status') == 'PASSED':
        print("Calculation successful.")
    else:
        print("Calculation encountered an issue.")
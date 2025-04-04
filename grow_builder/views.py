# grow_builder/views.py

from django.http import JsonResponse
from django.shortcuts import render
import logging
import math # <-- Add import
import numpy as np # <-- Add import

# Assuming light_calculator is in the same app directory or accessible via Python path
# from .light_calculator import get_layout_and_targets # <-- Comment out or remove if not used elsewhere
from .light_calculator import (
    calculate_plant_positions_grid # Keep if using plants
)

logger = logging.getLogger(__name__)

def build_cob_positions(W, L, light_h, num_layers):
    """
    Calculates COB positions and returns them along with transformation parameters.

    Args:
        W (float): Room width in meters.
        L (float): Room length in meters.
        light_h (float): Mounting height of the COBs in meters.
        num_layers (int): The number of concentric layers (>= 1).

    Returns:
        tuple: (cob_positions_np, transform_params)
               - cob_positions_np: numpy.ndarray of shape (N, 4) [x, y, z, layer].
               - transform_params: dict containing parameters needed for strips.
    """
    if num_layers < 1:
        logger.warning("build_cob_positions called with num_layers < 1. Returning empty.")
        empty_params = {'center_x': W/2, 'center_y': L/2, 'scale_x': 1, 'scale_y': 1, 'cos_t': 1, 'sin_t': 0, 'light_h': light_h}
        return np.array([], dtype=np.float64).reshape(0, 4), empty_params

    n = num_layers - 1 # Max layer index (0 to n)
    positions = []

    # Abstract diamond grid points (layer by layer)
    positions.append((0, 0, light_h, 0)) # Center COB, layer 0
    for i in range(1, n + 1): # Layers 1 to n
        for x_layer in range(-i, i + 1):
            y_abs = i - abs(x_layer)
            # Add point if it's on the diamond perimeter for layer i
            # (Ensures we only get points defining the diamond shape)
            if abs(x_layer) + y_abs == i:
                if y_abs == 0:
                    positions.append((x_layer, 0, light_h, i))
                else:
                    positions.append((x_layer, y_abs, light_h, i))
                    positions.append((x_layer, -y_abs, light_h, i))

    # Transformation parameters (matching the script)
    theta = math.radians(45)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    center_x, center_y = W / 2.0, L / 2.0

    # Scale factors to fit the diamond within the room boundaries
    # Denominator is max extent (n/sqrt(2)) *before* scaling to fit W/2 or L/2
    # Use the script's scaling logic for consistency
    denominator = n * math.sqrt(2)
    scale_x = W / denominator if n > 0 else 1.0 # Avoid division by zero if only layer 0 exists
    scale_y = L / denominator if n > 0 else 1.0

    transform_params = {
        'center_x': center_x, 'center_y': center_y,
        'scale_x': scale_x, 'scale_y': scale_y,
        'cos_t': cos_t, 'sin_t': sin_t,
        'light_h': light_h # Store the height used
    }

    transformed_positions = []
    for (ax, ay, _, layer) in positions: # Abstract x, y; use light_h from params for consistency
        # Rotate
        rx = ax * cos_t - ay * sin_t
        ry = ax * sin_t + ay * cos_t
        # Scale and Translate
        px = center_x + rx * scale_x
        py = center_y + ry * scale_y
        pz = light_h # Use the consistent light height
        transformed_positions.append((px, py, pz, layer))

    cob_positions_np = np.array(transformed_positions, dtype=np.float64)
    if cob_positions_np.shape[0] == 0: # Handle case where input was invalid
       cob_positions_np = cob_positions_np.reshape(0,4)


    logger.info(f"Built {len(transformed_positions)} COB positions using {num_layers} layers.")
    logger.debug(f"Transform params: {transform_params}")

    return cob_positions_np, transform_params
# --- Layer Calculation Helper ---
def calculate_num_layers(W_m, L_m):
    """
    Calculates the number of COB layers based purely on a linear relationship
    with the effective side length of the floor area.

    Rule: 10 layers @ 6.10m effective side, +/- 1 layer per +/- 0.64m step.

    Args:
        W_m (float): Room width in meters.
        L_m (float): Room length in meters.

    Returns:
        int: The calculated number of COB layers (minimum 1).
    """
    # Define the reference point and step size
    reference_layers = 10
    reference_effective_side_m = 6.10
    step_m = 0.64

    # Handle invalid dimensions - minimum 1 layer (center COB)
    if W_m <= 0 or L_m <= 0:
        logger.warning("Invalid dimensions (<=0) for layer calculation. Returning minimum layers: 1.")
        return 1

    # Calculate the side length of a square with equivalent area
    area_m2 = W_m * L_m
    effective_side_m = math.sqrt(area_m2)
    logger.info(f"Layer Calc: W={W_m:.2f}m, L={L_m:.2f}m -> Area={area_m2:.2f}m², Effective Side={effective_side_m:.2f}m")

    # Calculate the difference in side length from the reference
    delta_side_m = effective_side_m - reference_effective_side_m

    # Calculate the corresponding change in layers (using floor for steps completed)
    # floor works correctly for both positive and negative delta_side_m
    # e.g., floor(0.63 / 0.64) = 0; floor(0.64 / 0.64) = 1
    # e.g., floor(-0.01 / 0.64) = -1; floor(-0.64 / 0.64) = -1; floor(-0.65 / 0.64) = -2
    layer_delta = math.floor(delta_side_m / step_m)

    # Calculate the theoretical number of layers
    calculated_layers = reference_layers + layer_delta
    logger.info(f"Effective side delta: {delta_side_m:.2f}m. Layer delta (steps): {layer_delta}. Theoretical layers: {calculated_layers}")

    # Enforce a practical minimum of 1 layer
    num_layers = max(1, calculated_layers)
    if num_layers != calculated_layers:
         logger.info(f"Adjusted layers from {calculated_layers} to practical minimum: {num_layers}")

    return int(num_layers)

# --- API View ---
def calculate_lights_api(request):
    """ API endpoint for COB/Strip layout """
    try:
        W_str = request.GET.get('W'); L_str = request.GET.get('L')
        H_str = request.GET.get('H'); light_h_str = request.GET.get('light_h')

        # --- Parameter Validation (Expecting METERS here) ---
        if not all([W_str, L_str, H_str, light_h_str]):
             missing = [k for k, v in {'W': W_str, 'L': L_str, 'H': H_str, 'light_h': light_h_str}.items() if not v]
             logger.warning(f"API Bad Request: Missing params: {', '.join(missing)}")
             return JsonResponse({'error': f'Missing required parameters: {", ".join(missing)}'}, status=400)
        try:
            W = float(W_str); L = float(L_str); H = float(H_str); light_h = float(light_h_str)
            if not (W > 0 and L > 0 and H > 0 and light_h > 0): raise ValueError("Dimensions and heights must be positive.")
            if light_h >= H: raise ValueError("Light height must be less than ceiling height.")
        except ValueError as ve:
            logger.warning(f"API Bad Request: Invalid params (expecting meters): {ve}. Input: W='{W_str}', L='{L_str}', H='{H_str}', light_h='{light_h_str}'")
            return JsonResponse({'error': f'Invalid parameter value: {ve}'}, status=400)

        logger.info(f"API: Request for W={W:.2f}m, L={L:.2f}m, H={H:.2f}m, light_h={light_h:.2f}m")

        # --- Calculate Dynamic Number of Layers for COBs ---
        num_layers = calculate_num_layers(W, L)

        # --- Call the COB calculator (expects METERS) - GET PARAMS TOO ---
        cob_positions_np, transform_params = build_cob_positions(W, L, light_h, num_layers)

        # --- Format COB positions for JSON ---
        placed_cobs_list = [
            {'x': float(row[0]), 'y': float(row[1]), 'z': float(row[2])} # x=px, y=py, z=pz (light_h)
            for row in cob_positions_np
        ]

        # --- Calculate Strip Layout using Script Logic --- ### UPDATED STRIP LOGIC ###
        strip_layer_coordinates = [] # This will be the final list of lists for JSON
        logger.info(f"Calculating strips for layers 1 to {num_layers - 1}...")

        for layer_idx in range(1, num_layers): # Strips start from layer 1
            # Get ordered points (tuples) for this layer using the transform params
            ordered_points_tuples = get_ordered_strip_layer_coords(layer_idx, transform_params)

            if ordered_points_tuples:
                # Convert list of tuples (px, py, pz) to list of dicts {'x':px, 'y':py, 'z':pz}
                layer_points_list = [
                    {'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])}
                    for p in ordered_points_tuples
                ]
                strip_layer_coordinates.append(layer_points_list)
                # logger.debug(f" > Added layer {layer_idx} with {len(layer_points_list)} points.")
            # else:
                # logger.debug(f" > No points generated for strip layer {layer_idx}.")


        # --- Calculate Plant Positions (expects METERS) ---
        plant_positions = calculate_plant_positions_grid(W, L)
        logger.info(f"API: Calculated {len(plant_positions)} plant positions.")

        # --- Prepare Success Response (All coords are in METERS) ---
        response_data = {
            'modules': [],
            'placed_cob_positions': placed_cobs_list,
            'target_cob_positions': [], # Not calculated
            'strip_layer_coordinates': strip_layer_coordinates, # USE THE NEW DATA
            'plant_positions': plant_positions,
            'dimensions': {'W_m': W, 'L_m': L, 'H_m': H, 'light_h_m': light_h},
            'verification': {
                 'match_status': 'N/A',
                 'error': None,
                 'num_layers_calculated': num_layers,
                 'num_cobs_placed': len(placed_cobs_list),
                 'num_strip_layers': len(strip_layer_coordinates) # Count actual layers generated
            }
        }
        logger.info(f"API: Sending successful response. N_PlacedCOBs={len(placed_cobs_list)}, "
                    f"N_StripLayers={len(strip_layer_coordinates)}, N_LayersUsedForCOBs={num_layers}, N_Plants={len(plant_positions)}")
        return JsonResponse(response_data, status=200)

    except Exception as e:
        logger.exception(f"API FATAL UNHANDLED ERROR in View: {type(e).__name__}: {e}")
        return JsonResponse({'error': 'An internal server error occurred.'}, status=500)

def get_ordered_strip_layer_coords(layer_index, transform_params):
    """
    Generates the transformed coordinates for a specific layer's perimeter IN ORDER,
    suitable for drawing connecting strips. Returns list of (px, py, pz) tuples.
    Uses transformation parameters provided.
    """
    if layer_index == 0:
        return [] # No strips for the center layer 0

    # --- Retrieve parameters ---
    center_x = transform_params['center_x']
    center_y = transform_params['center_y']
    scale_x = transform_params['scale_x']
    scale_y = transform_params['scale_y']
    cos_t = transform_params['cos_t']
    sin_t = transform_params['sin_t']
    # Use the consistent light height passed in params
    light_h = transform_params['light_h']

    i = layer_index # Current layer index (1 or greater)
    ordered_abstract_coords = []

    # Generate abstract diamond coordinates in clockwise order
    # Note: Using light_h here for the abstract 'h' is just a placeholder,
    # the final pz comes from transform_params['light_h'] after transformation.

    # Top-Right vertex to Top vertex (Quadrant 1 edge)
    for x in range(i, 0, -1): # x from i down to 1
        ordered_abstract_coords.append((x, i - x, light_h, i)) # y = i - abs(x) = i - x
    # Top vertex to Left vertex (Quadrant 2 edge)
    for x in range(0, -i, -1): # x from 0 down to -(i-1)
         ordered_abstract_coords.append((x, i + x, light_h, i)) # y = i - abs(x) = i - (-x) = i + x
    # Left vertex to Bottom vertex (Quadrant 3 edge)
    for x in range(-i, 0, 1): # x from -i up to -1
         ordered_abstract_coords.append((x, -i - x, light_h, i)) # y = -(i - abs(x)) = -(i - (-x)) = -i - x
    # Bottom vertex to Right vertex (Quadrant 4 edge)
    for x in range(0, i + 1, 1): # x from 0 up to i (include the starting point x=i again to close loop visually if needed, although frontend connects last to first)
         ordered_abstract_coords.append((x, -i + x, light_h, i)) # y = -(i - abs(x)) = -(i - x) = -i + x

    # Apply the same transformation as COBs
    ordered_transformed_coords = []
    unique_points = set() # Avoid duplicate points if generation logic overlaps start/end

    for (ax, ay, _, layer) in ordered_abstract_coords:
        # Rotate
        rx = ax * cos_t - ay * sin_t
        ry = ax * sin_t + ay * cos_t
        # Scale and Translate
        px = center_x + rx * scale_x
        py = center_y + ry * scale_y
        pz = light_h # Use the consistent light height

        point_tuple = (round(px, 5), round(py, 5), round(pz, 5)) # Round for stability if needed
        if point_tuple not in unique_points:
            ordered_transformed_coords.append(point_tuple)
            unique_points.add(point_tuple)

    # Check if the first point was added again at the end and remove if so,
    # as the frontend will connect the last point back to the first.
    if len(ordered_transformed_coords) > 1 and ordered_transformed_coords[0] == ordered_transformed_coords[-1]:
        ordered_transformed_coords.pop()


    # logger.debug(f"Generated {len(ordered_transformed_coords)} ordered points for strip layer {layer_index}")
    return ordered_transformed_coords

# --- Update initial values to FEET ---
def grow_room_builder_view(request):
    # Set initial values in FEET for the HTML form defaults
    initial_W_ft = 12.0
    initial_L_ft = 12.0
    initial_H_ft = 10.0 # Changed default to 10ft for more typical rooms
    # Calculate default light height relative to ceiling height in FEET
    # Place lights ~1ft below ceiling, minimum 0.5ft
    initial_light_h_ft = max(0.5, round(initial_H_ft - 1.0, 1)) # e.g. 9.0 ft for 10ft ceiling

    # Ensure light height is less than ceiling height initially
    if initial_light_h_ft >= initial_H_ft:
        initial_light_h_ft = max(0.5, round(initial_H_ft * 0.9, 1)) # Fallback

    context = {
        'initial_W': initial_W_ft,
        'initial_L': initial_L_ft,
        'initial_H': initial_H_ft,
        'initial_light_h': initial_light_h_ft,
     }
    return render(request, 'grow_builder/builder.html', context)
#!/usr/bin/env python3
import math
import matplotlib.pyplot as plt
import argparse

def build_diamond_61_pattern(width_ft, length_ft):
    # Raw pattern: each sub‐list is one layer.
    layers_raw = [
        [(0, 0)],  # Layer 1 (center)
        [(-1, 0), (1, 0), (0, -1), (0, 1)],  # Layer 2
        [(-1, -1), (1, -1), (-1, 1), (1, 1),
         (-2, 0), (2, 0), (0, -2), (0, 2)],  # Layer 3
        [(-2, -1), (2, -1), (-2, 1), (2, 1),
         (-1, -2), (1, -2), (-1, 2), (1, 2),
         (-3, 0), (3, 0), (0, -3), (0, 3)],  # Layer 4
        [(-2, -2), (2, -2), (-2, 2), (2, 2),
         (-3, -1), (3, -1), (-3, 1), (3, 1),
         (-1, -3), (1, -3), (-1, 3), (1, 3),
         (-4, 0), (4, 0), (0, -4), (0, 4)],  # Layer 5
        [(-3, -2), (3, -2), (-3, 2), (3, 2),
         (-2, -3), (2, -3), (-2, 3), (2, 3),
         (-4, -1), (4, -1), (-4, 1), (4, 1),
         (-1, -4), (1, -4), (-1, 4), (1, 4),
         (-5, 0), (5, 0), (0, -5), (0, 5)]   # Layer 6
    ]
    # Define spacing based on the floor dimensions.
    spacing_x = width_ft / 3.5
    spacing_y = length_ft / 3.5
    theta = math.radians(45)
    layers_scaled = []
    for layer in layers_raw:
        coords_layer = []
        for (ix, iy) in layer:
            x_offset = ix * spacing_x
            y_offset = iy * spacing_y
            rotated_x = x_offset * math.cos(theta) - y_offset * math.sin(theta)
            rotated_y = x_offset * math.sin(theta) + y_offset * math.cos(theta)
            coords_layer.append((rotated_x, rotated_y))
        layers_scaled.append(coords_layer)
    return layers_scaled

def generate_light_sources(floor_width_ft, floor_length_ft):
    # Build the left diamond arrangement.
    diamond_layers = build_diamond_61_pattern(floor_width_ft, floor_length_ft)
    
    # Determine the overall horizontal extent of the left diamond.
    all_points = [pt for layer in diamond_layers for pt in layer]
    xs = [pt[0] for pt in all_points]
    min_x = min(xs)
    max_x = max(xs)
    width_arr = max_x - min_x

    # Use a one-unit gap between the arrangements.
    spacing_x = floor_width_ft / 3.5  # one “unit”
    gap = spacing_x

    # Build the right diamond by shifting the left one to the right.
    offset = width_arr + gap
    second_diamond_layers = []
    for layer in diamond_layers:
        shifted_layer = [(x + offset, y) for (x, y) in layer]
        second_diamond_layers.append(shifted_layer)

    # Merge the two diamonds, grouping layer 1 and 2 together and then each subsequent layer.
    merged_first = diamond_layers[0] + diamond_layers[1] + second_diamond_layers[0] + second_diamond_layers[1]
    remaining_layers = []
    for i in range(2, len(diamond_layers)):
        remaining_layers.append(diamond_layers[i] + second_diamond_layers[i])
    # Combined layers: group 1 is merged layers 1&2; group 2 is layer 3; group 3 is layer 4; 
    # group 4 is layer 5; group 5 is layer 6.
    combined_layers = [merged_first] + remaining_layers

    # --- Insert extra gap COBs into group 4 (which comes from diamond_layers[4]) ---
    # We want the extra COBs to share rows with group 4.
    # For each of a selected set of rows (from the left diamond's layer 5), we will find
    # the rightmost left COB and the leftmost right COB, then insert a COB at the midpoint.
    left_group4 = diamond_layers[4]
    right_group4 = second_diamond_layers[4]

    # Get unique y-values from the left diamond’s group 4.
    tol = 1e-3
    unique_y = sorted({round(y, 3) for (x, y) in left_group4})
    # Choose the central 5 rows (if there are at least 5).
    if len(unique_y) >= 5:
        start = (len(unique_y) - 5) // 2
        central_rows = unique_y[start:start+5]
    else:
        central_rows = unique_y

    extra_gap_cobs = []
    for target_y in central_rows:
        # For a given row, take the left diamond’s points with y ~ target_y.
        left_candidates = [x for (x, y) in left_group4 if abs(y - target_y) < tol]
        right_candidates = [x for (x, y) in right_group4 if abs(y - target_y) < tol]
        if left_candidates and right_candidates:
            x_left = max(left_candidates)    # rightmost left diamond COB in that row
            x_right = min(right_candidates)   # leftmost right diamond COB in that row
            mid_x = (x_left + x_right) / 2
            extra_gap_cobs.append((mid_x, target_y))
    # Insert these extra COBs into group 4 (which is combined_layers[3],
    # because group 1 is merged layers 0&1, group 2 comes from layer 3, group 3 from layer 4, and group 4 from layer 5)
    group4_index = 1 + 2  # merged_first is group 1; remaining_layers[0] is group 2; remaining_layers[1] is group 3; remaining_layers[2] is group 4.
    combined_layers[group4_index].extend(extra_gap_cobs)
    
    return combined_layers

def visualize_light_sources(floor_width_ft, floor_length_ft):
    light_layers = generate_light_sources(floor_width_ft, floor_length_ft)
    
    plt.figure(figsize=(8, 8))
    # Define colors for up to 6 groups.
    layer_colors = ["red", "blue", "green", "orange", "purple", "brown"]

    for idx, layer in enumerate(light_layers):
        if layer:
            xs = [pt[0] for pt in layer]
            ys = [pt[1] for pt in layer]
            # Label group 4 to indicate it now contains the extra gap COBs.
            label = f"Group {idx+1}"
            if idx == 3:
                label = "Group 4 (with gap COBs)"
            plt.scatter(xs, ys, color=layer_colors[idx % len(layer_colors)],
                        label=label, s=80, alpha=0.8)
    
    plt.title(f"Mirrored Diamond with Extra Gap COBs in Group 4\nFloor: {floor_width_ft}ft x {floor_length_ft}ft")
    plt.xlabel("X Position (ft)")
    plt.ylabel("Y Position (ft)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize two mirrored COB arrangements with extra gap COBs sharing rows with group 4.")
    parser.add_argument('--floor_width', type=float, default=12.0, help='Floor width in feet.')
    parser.add_argument('--floor_length', type=float, default=12.0, help='Floor length in feet.')
    args = parser.parse_args()
    visualize_light_sources(args.floor_width, args.floor_length)

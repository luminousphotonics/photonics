#!/usr/bin/env python3
import math
import matplotlib.pyplot as plt
import argparse

def build_diamond_61_pattern(width_ft, length_ft):
    """
    Builds the full mirrored diamond pattern with gap COBs.
    
    Returns a list of six groups:
      - Group 1: Merged Layers 1 & 2 (both left and right diamonds)
      - Group 2: Layer 3 (both)
      - Group 3: Layer 4 (both)
      - Group 4: Layer 5 (both, without gap COBs)
      - Group 5: Layer 6 (both)
      - Group 6: Extra Gap COBs (calculated from layer 5)
    """
    # Raw diamond pattern layers (6 layers)
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
    
    # Determine spacing based on floor dimensions.
    spacing_x = width_ft / 3.5
    spacing_y = length_ft / 3.5
    theta = math.radians(45)

    # Build left diamond (scaled and rotated)
    diamond_layers = []
    for layer in layers_raw:
        coords_layer = []
        for (ix, iy) in layer:
            x_offset = ix * spacing_x
            y_offset = iy * spacing_y
            rotated_x = x_offset * math.cos(theta) - y_offset * math.sin(theta)
            rotated_y = x_offset * math.sin(theta) + y_offset * math.cos(theta)
            coords_layer.append((rotated_x, rotated_y))
        diamond_layers.append(coords_layer)

    # Determine horizontal extent of the left diamond.
    all_points = [pt for layer in diamond_layers for pt in layer]
    xs = [pt[0] for pt in all_points]
    min_x = min(xs)
    max_x = max(xs)
    width_arr = max_x - min_x

    # Define one-unit gap between the two arrangements.
    gap = spacing_x

    # Build the right diamond by shifting the left diamond.
    offset = width_arr + gap
    second_diamond_layers = []
    for layer in diamond_layers:
        shifted_layer = [(x + offset, y) for (x, y) in layer]
        second_diamond_layers.append(shifted_layer)

    # Form the groups.
    group1 = diamond_layers[0] + diamond_layers[1] + second_diamond_layers[0] + second_diamond_layers[1]
    group2 = diamond_layers[2] + second_diamond_layers[2]
    group3 = diamond_layers[3] + second_diamond_layers[3]
    group4 = diamond_layers[4] + second_diamond_layers[4]
    group5 = diamond_layers[5] + second_diamond_layers[5]

    # Compute extra gap COBs from layer 5.
    left_group4 = diamond_layers[4]
    right_group4 = second_diamond_layers[4]
    tol = 1e-3
    unique_y = sorted({round(y, 3) for (x, y) in left_group4})
    if len(unique_y) >= 5:
        start = (len(unique_y) - 5) // 2
        central_rows = unique_y[start:start+5]
    else:
        central_rows = unique_y

    gap_group = []
    for target_y in central_rows:
        left_candidates = [x for (x, y) in left_group4 if abs(y - target_y) < tol]
        right_candidates = [x for (x, y) in right_group4 if abs(y - target_y) < tol]
        if left_candidates and right_candidates:
            x_left = max(left_candidates)
            x_right = min(right_candidates)
            mid_x = (x_left + x_right) / 2
            gap_group.append((mid_x, target_y))

    # Return all six groups.
    return [group1, group2, group3, group4, group5, gap_group]

def visualize_light_sources(floor_width_ft, floor_length_ft):
    groups = build_diamond_61_pattern(floor_width_ft, floor_length_ft)
    group_labels = [
        "Group 1 (Merged Layers 1 & 2)",
        "Group 2 (Layer 3)",
        "Group 3 (Layer 4)",
        "Group 4 (Layer 5)",
        "Group 5 (Layer 6)",
        "Gap COBs"
    ]
    colors = ["red", "blue", "green", "orange", "purple", "brown"]

    plt.figure(figsize=(8, 8))
    for idx, group in enumerate(groups):
        if group:
            xs = [pt[0] for pt in group]
            ys = [pt[1] for pt in group]
            plt.scatter(xs, ys, color=colors[idx % len(colors)],
                        label=group_labels[idx], s=80, alpha=0.8)
    plt.title("Mirrored Diamond Arrangement with Gap COBs\nFloor: {}ft x {}ft".format(floor_width_ft, floor_length_ft))
    plt.xlabel("X Position (ft)")
    plt.ylabel("Y Position (ft)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize full diamond pattern with groups and gap COBs.")
    parser.add_argument('--floor_width', type=float, default=12.0, help='Floor width in feet.')
    parser.add_argument('--floor_length', type=float, default=12.0, help='Floor length in feet.')
    args = parser.parse_args()
    visualize_light_sources(args.floor_width, args.floor_length)

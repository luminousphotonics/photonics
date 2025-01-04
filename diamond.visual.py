import math
import matplotlib.pyplot as plt

def build_diamond_61_pattern(width_ft, length_ft):
    """
    Builds a Diamond: 61 arrangement (6 layers, total 61 emitters).
    This scales the arrangement so the outermost layer roughly
    fits within width_ft x length_ft, then rotates 45 degrees
    to form a diamond shape.
    Returns a list of layer-coordinates, e.g.:
        [
          [(x0,y0)],   # Layer 0
          [(x1_1,y1_1), (x1_2,y1_2), ...],  # Layer 1
          ...
          [(...), ...] # Layer 5
        ]
    """
    # Raw “Diamond: 61” pattern in integer steps
    layers_raw = [
        [(0, 0)],  # Layer 0: 1 emitter
        [(-1, 0), (1, 0), (0, -1), (0, 1)],  # Layer 1: 4 emitters
        [(-1, -1), (1, -1), (-1, 1), (1, 1),
         (-2, 0), (2, 0), (0, -2), (0, 2)],  # Layer 2: 8 emitters
        [(-2, -1), (2, -1), (-2, 1), (2, 1),
         (-1, -2), (1, -2), (-1, 2), (1, 2),
         (-3, 0), (3, 0), (0, -3), (0, 3)],  # Layer 3: 12 emitters
        [(-2, -2), (2, -2), (-2, 2), (2, 2),
         (-3, -1), (3, -1), (-3, 1), (3, 1),
         (-1, -3), (1, -3), (-1, 3), (1, 3),
         (-4, 0), (4, 0), (0, -4), (0, 4)],  # Layer 4: 16 emitters
        [(-3, -2), (3, -2), (-3, 2), (3, 2),
         (-2, -3), (2, -3), (-2, 3), (2, 3),
         (-4, -1), (4, -1), (-4, 1), (4, 1),
         (-1, -4), (1, -4), (-1, 4), (1, 4),
         (-5, 0), (5, 0), (0, -5), (0, 5)]   # Layer 5: 20 emitters
    ]

    # Scale and rotate them to fit ~ width_ft x length_ft,
    # then do a 45° rotation to form a “diamond” shape.
    spacing_x = width_ft / 7.2
    spacing_y = length_ft / 7.2
    theta = math.radians(45)

    layers_scaled = []
    for layer in layers_raw:
        coords_layer = []
        for (ix, iy) in layer:
            x_offset = ix * spacing_x
            y_offset = iy * spacing_y
            # rotate 45° around origin
            rotated_x = x_offset * math.cos(theta) - y_offset * math.sin(theta)
            rotated_y = x_offset * math.sin(theta) + y_offset * math.cos(theta)
            coords_layer.append((rotated_x, rotated_y))
        layers_scaled.append(coords_layer)

    return layers_scaled


def main():
    # Adjust these if you want a different overall size
    width_ft = 12.0
    length_ft = 12.0

    diamond_positions = build_diamond_61_pattern(width_ft, length_ft)

    # Let's plot each layer with a different color
    layer_colors = ["red", "blue", "green", "orange", "purple", "cyan"]
    # (6 layers total, so we can pick 6 colors)

    plt.figure(figsize=(6, 6))
    for idx, coords in enumerate(diamond_positions):
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        color = layer_colors[idx % len(layer_colors)]
        plt.scatter(xs, ys, color=color, label=f"Layer {idx}", s=80, alpha=0.8)

    plt.title(f"Diamond: 61 Layout ({width_ft} ft x {length_ft} ft nominal)")
    plt.xlabel("X (feet)")
    plt.ylabel("Y (feet)")
    plt.legend(loc="best")
    plt.grid(True)
    # Make the axes have equal scaling so circles, squares, etc. aren’t skewed
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()

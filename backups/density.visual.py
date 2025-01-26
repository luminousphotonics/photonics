import matplotlib.pyplot as plt
import numpy as np

def visualize_centered_square_vs_grid(layers, grid_size):
    """
    Visualizes a rotated centered square number sequence (with colored layers and squares)
    against a centered regular grid (with concentric square layers outlined, starting
    from a central square formed by 4 elements), both with the same cross-section limits.

    Args:
        layers: List of lists representing the coordinates of elements in each
                layer of the centered square pattern.
        grid_size: Tuple (rows, cols) representing the dimensions of the grid.
    """

    # --- Rotated Centered Square Visualization ---
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)

    # Rotation angle (45 degrees)
    theta = np.radians(45)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])

    all_coords_rotated = []
    layer_colors = ["blue", "green", "orange", "yellow", "purple", "cyan"]

    for i, layer in enumerate(layers):
        rotated_layer = []
        for x, y in layer:
            rotated_coord = np.dot(rotation_matrix, np.array([x, y]))
            rotated_layer.append(rotated_coord)
            all_coords_rotated.append(rotated_coord)

        xs = [coord[0] for coord in rotated_layer]
        ys = [coord[1] for coord in rotated_layer]
        plt.scatter(xs, ys, s=50, color=layer_colors[i % len(layer_colors)], label=f"Layer {i+1}")

        # Draw squares around each layer
        if len(xs) >= 2:
            max_x = max(xs)
            min_x = min(xs)
            max_y = max(ys)
            min_y = min(ys)
            width = max(max_x - min_x, max_y - min_y)
            center_x = (max_x + min_x) / 2
            center_y = (max_y + min_y) / 2
            square = plt.Rectangle((center_x - width / 2, center_y - width / 2), width, width,
                                   linewidth=1.5, edgecolor=layer_colors[i % len(layer_colors)],
                                   facecolor='none')
            plt.gca().add_patch(square)

    # Find the maximum coordinate for the centered square pattern
    max_coord_centered = max(max(abs(x), abs(y)) for x, y in all_coords_rotated)

    plt.title("Rotated Centered Square Number Sequence")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.xlim(-max_coord_centered - 1, max_coord_centered + 1)
    plt.ylim(-max_coord_centered - 1, max_coord_centered + 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='upper right', fontsize='small')

    # --- Grid Visualization (Centered, Concentric Squares from 4 Central Elements) ---
    plt.subplot(1, 2, 2)

    rows, cols = grid_size
    grid_coords = []

    # Create centered grid coordinates
    for r in range(rows):
        for c in range(cols):
            x = c - cols // 2 + (0.5 if cols % 2 == 0 else 0)
            y = r - rows // 2 + (0.5 if rows % 2 == 0 else 0)
            grid_coords.append((x, y))

    # Assign each point to a layer based on its distance from the center
    layers_grid = {}
    for coord in grid_coords:
        x, y = coord
        # Layer number: 1 for central layer, 2 for first ring, etc.
        layer_num = int(max(abs(x), abs(y)) + 0.5)
        if layer_num == 0:
            layer_num = 1  # Assign central points to Layer 1
        if layer_num not in layers_grid:
            layers_grid[layer_num] = []
        layers_grid[layer_num].append(coord)

    # Define colors for up to 6 layers
    grid_layer_colors = ["blue", "green", "orange", "yellow", "purple", "cyan"]

    # Plot all grid points in light gray first
    xs_all = [coord[0] for coord in grid_coords]
    ys_all = [coord[1] for coord in grid_coords]
    plt.scatter(xs_all, ys_all, s=50, color='lightgray')

    # Function to sort points in a layer clockwise based on angle from center
    def sort_layer_points(layer_coords):
        # Calculate angles from center (0,0)
        angles = [np.arctan2(y, x) for x, y in layer_coords]
        # Sort points by angle
        sorted_coords = [coord for _, coord in sorted(zip(angles, layer_coords))]
        return sorted_coords

    # Iterate through each layer and plot
    for layer_num in sorted(layers_grid.keys()):
        layer_coords = layers_grid[layer_num]
        sorted_coords = sort_layer_points(layer_coords)
        layer_color = grid_layer_colors[(layer_num - 1) % len(grid_layer_colors)]

        # Extract sorted x and y coordinates
        layer_xs = [coord[0] for coord in sorted_coords]
        layer_ys = [coord[1] for coord in sorted_coords]

        # Scatter the layer points with distinct color
        plt.scatter(layer_xs, layer_ys, s=50, color=layer_color, label=f"Layer {layer_num}")

        # Connect the perimeter points to form a square
        for i in range(len(sorted_coords)):
            start = sorted_coords[i]
            end = sorted_coords[(i + 1) % len(sorted_coords)]
            plt.plot([start[0], end[0]], [start[1], end[1]], color=layer_color, linewidth=1.5)

    # Find the maximum coordinate for setting plot limits
    max_coord_grid = max(max(abs(x), abs(y)) for x, y in grid_coords)

    plt.title("Regular Grid (Centered)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.xlim(-max_coord_grid - 1, max_coord_grid + 1)
    plt.ylim(-max_coord_grid - 1, max_coord_grid + 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='upper right', fontsize='small')

    # --- Density Comparison ---
    num_elements_centered = len(all_coords_rotated)
    num_elements_grid = len(grid_coords)

    area_centered = (2 * max_coord_centered)**2
    area_grid = (2 * max_coord_centered)**2  # Assuming both plots have the same limits

    density_centered = num_elements_centered / area_centered if area_centered > 0 else 0
    density_grid = num_elements_grid / area_grid if area_grid > 0 else 0

    #plt.figtext(0.25, 0.95, f"Centered Density: {density_centered:.2f} elements/unit²",
    #            ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    #plt.figtext(0.75, 0.95, f"Grid Density: {density_grid:.2f} elements/unit²",
    #            ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    layers = [
        [(0, 0)],
        [(-1, 0), (1, 0), (0, -1), (0, 1)],
        [(-1, -1), (1, -1), (-1, 1), (1, 1),
         (-2, 0), (2, 0), (0, -2), (0, 2)],
        [(-2, -1), (2, -1), (-2, 1), (2, 1),
         (-1, -2), (1, -2), (-1, 2), (1, 2),
         (-3, 0), (3, 0), (0, -3), (0, 3)],
        [(-2, -2), (2, -2), (-2, 2), (2, 2),
         (-3, -1), (3, -1), (-3, 1), (3, 1),
         (-1, -3), (1, -3), (-1, 3), (1, 3),
         (-4, 0), (4, 0), (0, -4), (0, 4)],
        [(-3, -2), (3, -2), (-3, 2), (3, 2),
         (-2, -3), (2, -3), (-2, 3), (2, 3),
         (-4, -1), (4, -1), (-4, 1), (4, 1),
         (-1, -4), (1, -4), (-1, 4), (1, 4),
         (-5, 0), (5, 0), (0, -5), (0, 5)]
    ]

    grid_size = (8, 8)

    visualize_centered_square_vs_grid(layers, grid_size)

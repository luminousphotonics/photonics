import matplotlib.pyplot as plt
import numpy as np
import math

def visualize_cob_positions(cob_positions, W, L):
    """
    Visualizes COB positions in a 2D plane.

    Args:
        cob_positions: A NumPy array of (x, y) coordinates.
        W: Width of the room.
        L: Length of the room.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(cob_positions[:, 0], cob_positions[:, 1], marker='o', color='blue', label='COB Positions')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Visualization of COB Positions')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()

    plt.gca().add_patch(plt.Rectangle((0, 0), W, L, linewidth=1, edgecolor='r', facecolor='none'))

    plt.show()

def build_cob_positions(W, L, H):
    """
    Calculates COB positions with a 45-degree rotation.

    Args:
        W: Width of the room.
        L: Length of the room.
        H: Height of the COBs (not used in the 2D visualization).

    Returns:
        A NumPy array of (x, y) coordinates.
    """
    center = (W / 2, L / 2)
    scaling_factor = W / 7.2
    theta = math.radians(45)  # 45-degree rotation

    # Simplified relative coordinates (no need for separate layers list)
    relative_coords = [
        (0, 0),  # Center
        (-1, 0), (1, 0), (0, -1), (0, 1),  # Layer 1
        (-1, -1), (1, -1), (-1, 1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2),  # Layer 2
        (-2, -1), (2, -1), (-2, 1), (2, 1), (-1, -2), (1, -2), (-1, 2), (1, 2), (-3, 0), (3, 0), (0, -3), (0, 3),  # Layer 3
        (-2, -2), (2, -2), (-2, 2), (2, 2), (-3, -1), (3, -1), (-3, 1), (3, 1), (-1, -3), (1, -3), (-1, 3), (1, 3), (-4, 0), (4, 0), (0, -4), (0, 4),  # Layer 4
        (-3, -2), (3, -2), (-3, 2), (3, 2), (-2, -3), (2, -3), (-2, 3), (2, 3), (-4, -1), (4, -1), (-4, 1), (4, 1), (-1, -4), (1, -4), (-1, 4), (1, 4), (-5, 0), (5, 0), (0, -5), (0, 5)  # Layer 5
    ]

    light_positions = []
    for dx, dy in relative_coords:
        # Apply rotation
        rx = dx * math.cos(theta) - dy * math.sin(theta)
        ry = dx * math.sin(theta) + dy * math.cos(theta)

        # Apply scaling and center offset
        px = center[0] + rx * scaling_factor
        py = center[1] + ry * scaling_factor
        light_positions.append((px, py))

    return np.array(light_positions)

if __name__ == '__main__':
    W = 10
    L = 10
    H = 3

    cob_positions = build_cob_positions(W, L, H)
    print(cob_positions)  # Print the calculated coordinates

    visualize_cob_positions(cob_positions, W, L)
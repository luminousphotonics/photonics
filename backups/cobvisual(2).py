import matplotlib.pyplot as plt
import numpy as np

def visualize_cob_positions_precalculated(cob_positions, W, L):
    """
    Visualizes COB positions (pre-calculated) in a 2D plane.

    Args:
        cob_positions: A NumPy array of (x, y) coordinates.
        W: Width of the room (used for the bounding box).
        L: Length of the room (used for the bounding box).
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(cob_positions[:, 0], cob_positions[:, 1], marker='o', color='blue', label='COB Positions')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Visualization of COB Positions (Pre-calculated)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()

    plt.gca().add_patch(plt.Rectangle((0, 0), W, L, linewidth=1, edgecolor='r', facecolor='none'))

    plt.show()


if __name__ == '__main__':
    W = 10  # Width
    L = 10  # Length
    # H = 3  # Height is not needed for visualization, but we keep W and L for the room outline.

    # Your pre-calculated coordinates:
    cob_positions_precalculated = np.array([
        [5.        , 5.        ],
        [4.01790725, 4.01790725],
        [5.98209275, 5.98209275],
        [5.98209275, 4.01790725],
        [4.01790725, 5.98209275],
        [5.        , 3.0358145 ],
        [6.9641855 , 5.        ],
        [3.0358145 , 5.        ],
        [5.        , 6.9641855 ],
        [3.0358145 , 3.0358145 ],
        [6.9641855 , 6.9641855 ],
        [6.9641855 , 3.0358145 ],
        [3.0358145 , 6.9641855 ],
        [4.01790725, 2.05372175],
        [7.94627825, 5.98209275],
        [2.05372175, 4.01790725],
        [5.98209275, 7.94627825],
        [5.98209275, 2.05372175],
        [7.94627825, 4.01790725],
        [2.05372175, 5.98209275],
        [4.01790725, 7.94627825],
        [2.05372175, 2.05372175],
        [7.94627825, 7.94627825],
        [7.94627825, 2.05372175],
        [2.05372175, 7.94627825],
        [5.        , 1.07162899],
        [8.92837101, 5.        ],
        [1.07162899, 5.        ],
        [5.        , 8.92837101],
        [3.0358145 , 1.07162899],
        [8.92837101, 6.9641855 ],
        [1.07162899, 3.0358145 ],
        [6.9641855 , 8.92837101],
        [6.9641855 , 1.07162899],
        [8.92837101, 3.0358145 ],
        [1.07162899, 6.9641855 ],
        [3.0358145 , 8.92837101],
        [1.07162899, 1.07162899],
        [8.92837101, 8.92837101],
        [8.92837101, 1.07162899],
        [1.07162899, 8.92837101],
        [4.01790725, 0.08953624],
        [9.91046376, 5.98209275],
        [0.08953624, 4.01790725],
        [5.98209275, 9.91046376],
        [5.98209275, 0.08953624],
        [9.91046376, 4.01790725],
        [0.08953624, 5.98209275],
        [4.01790725, 9.91046376],
        [2.05372175, 0.08953624],
        [9.91046376, 7.94627825],
        [0.08953624, 2.05372175],
        [7.94627825, 9.91046376],
        [7.94627825, 0.08953624],
        [9.91046376, 2.05372175],
        [0.08953624, 7.94627825],
        [2.05372175, 9.91046376],
        [0.08953624, 0.08953624],
        [9.91046376, 9.91046376],
        [9.91046376, 0.08953624],
        [0.08953624, 9.91046376]
    ])

    visualize_cob_positions_precalculated(cob_positions_precalculated, W, L)
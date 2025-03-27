import csv
import numpy as np

layer_ppfds = {}
with open("ppfd_data.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    for row in reader:
        layer = int(row[0])
        ppfd = float(row[1])
        if layer not in layer_ppfds:
            layer_ppfds[layer] = []
        layer_ppfds[layer].append(ppfd)

average_ppfds = {}
num_points = {} # NEW: to count the grid points
for layer, ppfds in layer_ppfds.items():
    average_ppfds[layer] = np.mean(ppfds)
    num_points[layer] = len(ppfds) # Count the elements.

for layer, avg_ppfd in sorted(average_ppfds.items()):
    print(f"Layer {layer}: Average PPFD = {avg_ppfd:.2f}, Number of points: {num_points[layer]}")
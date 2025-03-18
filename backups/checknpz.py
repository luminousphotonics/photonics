import numpy as np

data = np.load("./backups/training_data.npz", allow_pickle=True)
print("Arrays found in training_data.npz:", data.files)

if "X_train" in data:
    X_train = data["X_train"]
    print("X_train shape:", X_train.shape)
else:
    print("X_train is missing.")

if "y_train" in data:
    y_train = data["y_train"]
    print("y_train shape:", y_train.shape)
else:
    print("y_train is missing.")

if "G_train" in data:
    G_train = data["G_train"]
    print("G_train shape:", G_train.shape)
else:
    print("G_train is missing (this is optional).")

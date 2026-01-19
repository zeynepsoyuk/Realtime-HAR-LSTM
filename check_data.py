import os
import numpy as np

DATA_DIR = "sequences"

print("Checking data directory:", os.path.abspath(DATA_DIR))

if not os.path.exists(DATA_DIR):
    print("ERROR: data folder not found")
    exit()

files = [f for f in os.listdir(DATA_DIR) if f.endswith(".npy")]

print("Found", len(files), "npy files")

for f in files:
    path = os.path.join(DATA_DIR, f)
    arr = np.load(path)
    print(f, "->", arr.shape)

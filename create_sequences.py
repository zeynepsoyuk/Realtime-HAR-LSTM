import pandas as pd
import numpy as np
import os

INPUT_DIR = "dataset"
OUTPUT_DIR = "sequences"
SEQUENCE_LENGTH = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):
    if not file.endswith(".csv"):
        continue

    activity = file.replace(".csv", "")
    df = pd.read_csv(os.path.join(INPUT_DIR, file))
    df = df.dropna()

    data = df.values
    num_sequences = len(data) // SEQUENCE_LENGTH

    sequences = []

    for i in range(num_sequences):
        seq = data[i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]
        sequences.append(seq)

    sequences = np.array(sequences)

    np.save(os.path.join(OUTPUT_DIR, f"{activity}.npy"), sequences)
    print(f"{activity}: {sequences.shape}")

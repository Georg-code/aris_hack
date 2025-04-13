import numpy as np
import matplotlib.pyplot as plt
import torch
from perlin_numpy import generate_perlin_noise_2d

# === Parameters ===
array_size = 100         # Output noise shape
res = 5              # Number of periods in each direction

# === Generate Perlin Noise ===
np.random.seed(0)
noise = generate_perlin_noise_2d((array_size, array_size), (res, res))

# Save raw noise to CSV
np.savetxt("perlin_noise_100x100_raw.csv", noise, delimiter=",")

# === Visualize Perlin Noise as 2D heatmap ===
plt.imshow(noise, cmap='gray', interpolation='lanczos')
plt.title("Raw Perlin Noise")
plt.colorbar()
plt.show()

# === Normalize noise to [0, 1] ===
normalized = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))

# Save normalized data to CSV
np.savetxt("heightmap100.csv", normalized, delimiter=",")

# === Load normalized data ===
data = np.loadtxt("heightmap100.csv", delimiter=",")

# === Plot middle row of the heightmap ===
middle_row = data[data.shape[0] // 2, :]  # Use correct slicing

plt.plot(middle_row)
plt.title("Middle Row of Normalized Perlin Noise")
plt.xlabel("X Index")
plt.ylabel("Height (Normalized)")
plt.grid(True)
plt.show()

# === Convert numpy array to PyTorch tensor ===
torch_array = torch.tensor(normalized, dtype=torch.float32)



import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import os

# --- Use same device as training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Match DQN architecture used during training ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- One-hot encoding ---
def one_hot(state, n):
    return torch.eye(n, device=device)[state]

# --- Convert flat position to (row, col) in 4×4 grid ---
def to_grid_coords(pos):
    return (pos // 4, pos % 4)

# --- Use exact map as in training ---
custom_map = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n

# --- Load trained model ---
model = DQN(n_states, n_actions).to(device)
model_path = os.path.abspath("../dqn_frozenlake.pth")  # match training output
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f"\n✅ Model loaded from: {model_path}")

# --- Run one episode ---
state, _ = env.reset()
done = False
positions = [state]

while not done:
    state_tensor = one_hot(state, n_states).unsqueeze(0)
    with torch.no_grad():
        action = torch.argmax(model(state_tensor)).item()

    state, reward, terminated, truncated, _ = env.step(action)
    positions.append(state)
    done = terminated or truncated

# --- Output visited positions as (row, col) ---
grid_positions = [to_grid_coords(pos) for pos in positions]
print("\n📍 Visited grid positions:")
for i, coord in enumerate(grid_positions):
    print(f"Step {i}: {coord}")

# --- Final outcome ---
if reward == 1:
    print("\n✅ Reached the goal!")
else:
    print("\n❌ Failed (fell in hole or ended early)")

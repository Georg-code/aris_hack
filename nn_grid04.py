import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Constants
BOARD_ROWS, BOARD_COLS = 5, 5
START = (4, 0)
CHECKPOINTS = [(0, 0), (0, 4)]
CHECKPOINT_REWARD = 5
STEP_PENALTY = -2

# Initial terrain matrix
initial_matrix = np.array([
    [5, -2, -2, -2, 5],
    [-2, -2, -2, -2, -2],
    [-2, -2, -2, -2, -2],
    [-2, -2, -2, -2, -2],
    [0, -2, -2, -2, -2]
])

# Action mapping
ACTIONS = ["up", "down", "left", "right"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}

# Define the Neural Network policy
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=27, hidden_dim=32, output_dim=4):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Encode state as a flat tensor with checkpoint progress (length 25 + 2)
def encode_state(position, checkpoints):
    vec = np.zeros((BOARD_ROWS, BOARD_COLS))
    vec[position] = 1.0  # agent position
    flat = vec.flatten()
    # Add checkpoint flags
    cp_status = [1.0 if cp in checkpoints else 0.0 for cp in CHECKPOINTS]
    return np.concatenate([flat, cp_status])

class RLAgent:
    def __init__(self, lr=0.01):
        self.model = PolicyNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state_vec, epsilon):
        if random.random() < epsilon:
            return random.choice(ACTIONS)
        else:
            with torch.no_grad():
                q_values = self.model(torch.tensor(state_vec, dtype=torch.float32))
                return IDX_TO_ACTION[torch.argmax(q_values).item()]

    def train(self, episodes=200, gamma=0.9, epsilon=0.1):
        for ep in range(episodes):
            pos = START
            checkpoints = set()
            visited = set()
            total_reward = 0
            for t in range(30):
                state_vec = encode_state(pos, checkpoints)
                action = self.choose_action(state_vec, epsilon)

                # Compute next position
                delta = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[action]
                new_pos = (pos[0] + delta[0], pos[1] + delta[1])
                if not (0 <= new_pos[0] < BOARD_ROWS and 0 <= new_pos[1] < BOARD_COLS):
                    new_pos = pos

                # Base reward from terrain
                reward = initial_matrix[new_pos]

                # Penalize loops
                if new_pos == pos or (new_pos in visited):
                    reward -= 3
                visited.add(new_pos)

                # Reward checkpoint visits
                if new_pos in CHECKPOINTS and new_pos not in checkpoints:
                    checkpoints.add(new_pos)
                    reward += CHECKPOINT_REWARD

                # Reward for returning home after all checkpoints
                if new_pos == START and len(checkpoints) == len(CHECKPOINTS):
                    reward += CHECKPOINT_REWARD

                if ep < 3:
                    print(f"EP{ep+1} STEP{t+1}: pos={pos}, action={action}, new_pos={new_pos}, reward={reward}")

                next_state_vec = encode_state(new_pos, checkpoints)
                target = reward + gamma * torch.max(self.model(torch.tensor(next_state_vec, dtype=torch.float32))).item()

                self.optimizer.zero_grad()
                q_values = self.model(torch.tensor(state_vec, dtype=torch.float32))
                target_f = q_values.clone().detach()
                target_f[ACTION_TO_IDX[action]] = target
                loss = self.loss_fn(q_values, target_f)
                loss.backward()
                self.optimizer.step()

                pos = new_pos
                total_reward += reward

                if pos == START and len(checkpoints) == len(CHECKPOINTS):
                    break

            if (ep + 1) % 50 == 0:
                print(f"Episode {ep+1}, Total Reward: {total_reward}")

    def test_run(self, max_steps=50):
        pos = START
        checkpoints = set()
        path = []
        for _ in range(max_steps):
            state_vec = encode_state(pos, checkpoints)
            action = self.choose_action(state_vec, epsilon=0.0)
            path.append((pos, action))

            delta = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[action]
            new_pos = (pos[0] + delta[0], pos[1] + delta[1])
            if not (0 <= new_pos[0] < BOARD_ROWS and 0 <= new_pos[1] < BOARD_COLS):
                new_pos = pos

            if new_pos in CHECKPOINTS:
                checkpoints.add(new_pos)

            if new_pos == START and len(checkpoints) == len(CHECKPOINTS):
                path.append((new_pos, "done"))
                break

            pos = new_pos

        return path

if __name__ == "__main__":
    agent = RLAgent(lr=0.01)
    agent.train(episodes=200, gamma=0.9, epsilon=0.1)
    path = agent.test_run()
    print("Test path:")
    for step in path:
        print(step)

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Create FrozenLake environment
custom_map = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]
env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="human")

# Neural network model (input is one-hot encoded discrete state)
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

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
target_update_freq = 100
memory_size = 1000
episodes = 200

# Dimensions
input_dim = env.observation_space.n
output_dim = env.action_space.n

# Initialize networks
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)

def one_hot(state):
    return torch.eye(input_dim)[state]

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state_tensor = one_hot(state).unsqueeze(0)
        q_values = policy_net(state_tensor)
        return torch.argmax(q_values).item()

def optimize_model():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.stack([one_hot(s) for s in states])
    next_states = torch.stack([one_hot(s) for s in next_states])
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)

    q_values = policy_net(states).gather(1, actions).squeeze()
    with torch.no_grad():
        max_next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training loop
rewards_per_episode = []
steps_done = 0

for episode in range(episodes):
    state, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.append((state, action, reward, next_state, done))
        state = next_state
        episode_reward += reward

        optimize_model()

        if steps_done % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        steps_done += 1

    epsilon = max(epsilon_min, epsilon_decay * epsilon)
    rewards_per_episode.append(episode_reward)

# Plot results
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DQN on FrozenLake')
plt.show()

# Save the trained policy network
torch.save(policy_net.state_dict(), "dqn_frozenlake2.pth")

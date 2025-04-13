import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from env import Mars

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

def preprocess_state(state, size):
    # Convert agent and target positions into a single flat normalized vector
    agent = state["agent"] / (size - 1)
    target = state["target"] / (size - 1)
    return torch.tensor(np.concatenate([agent, target]), dtype=torch.float32).to(device)

def select_action(policy_net, state_tensor):
    probs = policy_net(state_tensor)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)

def train():
    env = Mars(size=5)
    size = env.size
    n_states = 4  # agent(x, y), target(x, y)
    n_actions = env.action_space.n

    policy_net = PolicyNet(n_states, n_actions).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
    gamma = 0.99

    for episode in range(1000):
        log_probs = []
        rewards = []
        state, _ = env.reset()
        done = False

        while not done:
            state_tensor = preprocess_state(state, size)
            action, log_prob = select_action(policy_net, state_tensor)
            next_state, reward, done, _, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        # Compute discounted rewards
        discounted = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            discounted.insert(0, G)
        discounted = torch.tensor(discounted).to(device)
        if len(discounted) > 1:
            discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)


        # Update policy
        loss = -torch.stack(log_probs) @ discounted
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode}: total reward = {sum(rewards):.2f}")

    torch.save(policy_net.state_dict(), "mars_policy.pt")
    print("Model saved as mars_policy.pt")

if __name__ == "__main__":
    train()

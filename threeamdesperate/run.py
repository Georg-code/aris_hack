
import torch
import torch.nn as nn
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

def run():
    env = Mars(size=5, is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    policy_net = PolicyNet(n_states, n_actions).to(device)
    policy_net.load_state_dict(torch.load("frozenlake_policy.pt"))
    policy_net.eval()

    for episode in range(5):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            state_tensor = torch.eye(n_states)[state].to(device)
            with torch.no_grad():
                probs = policy_net(state_tensor)
            action = torch.argmax(probs).item()
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        print(f"Episode {episode+1} total reward: {total_reward}")

if __name__ == "__main__":
    run()

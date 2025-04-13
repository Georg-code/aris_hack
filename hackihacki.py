import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
from gymnasium import spaces
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

GRID_SIZE = 100

# Load and normalize grayscale height map
img = Image.open("height.png").convert("L")
img_resized = img.resize((GRID_SIZE, GRID_SIZE), Image.BILINEAR)
height_data = np.array(img_resized, dtype=np.float32) / 255.0

class GridWorldEnv(gym.Env):
    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        })
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),   # RIGHT
            1: np.array([0, 1]),   # UP
            2: np.array([-1, 0]),  # LEFT
            3: np.array([0, -1]),  # DOWN
        }

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # self._agent_location = np.random.randint(1, self.size - 1, size=2)
        # self._target_location = np.random.randint(1, self.size - 1, size=2)
        self._agent_location = np.array([int(self.size / 2), int(self.size / 2)])
        self._target_location = np.array([self.size - 1, self.size - 1])
        while np.array_equal(self._agent_location, self._target_location):
            self._target_location = np.random.randint(1, self.size - 1, size=2)
        return self._get_obs(), self._get_info()

    def step(self, action):
        direction = self._action_to_direction[action]
        new_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        # if np.array_equal(new_location, self._agent_location):
        #     reward = -1.0  # Edge penalty
        # else:
        prev_location = self._agent_location.copy()
        self._agent_location = new_location
        terminated = np.array_equal(self._agent_location, self._target_location)

        reward = 10.0 if terminated else 0.0
        if not terminated:
            dist = np.linalg.norm(self._agent_location - self._target_location, ord=1)
            # reward -= 0.2 

            prev_height = height_data[prev_location[1], prev_location[0]]
            curr_height = height_data[self._agent_location[1], self._agent_location[0]]
            reward -= 0.2 * max(0, abs(curr_height - prev_height))

        return self._get_obs(), reward, terminated, False, self._get_info()

class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.out(x)

class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class GridWorldDQL:
    learning_rate_a = 0.001
    discount_factor_g = 0.9
    network_sync_rate = 10
    replay_memory_size = 1000
    mini_batch_size = 32
    loss_fn = nn.MSELoss()
    model_path = "gridworld_dql.pt"

    def train(self, episodes=3000):
        env = GridWorldEnv(size=GRID_SIZE)
        self.grid_size = env.size
        num_states = 4
        num_actions = env.action_space.n

        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)

        policy_dqn = DQN(in_states=num_states, h1_nodes=32, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=32, out_actions=num_actions)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []
        step_count = 0

        for ep in range(episodes):
            state = env.reset()[0]
            terminated = False
            ep_reward = 0

            while not terminated:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        s = self.state_to_dqn_input(state)
                        action = policy_dqn(s).argmax().item()

                new_state, reward, terminated, _, _ = env.step(action)
                memory.append((state, action, new_state, reward, terminated))
                state = new_state
                step_count += 1
                ep_reward += reward

            rewards_per_episode[ep] = ep_reward

            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                if ep > 400:
                    epsilon = max(epsilon - 1 / episodes, 0.01)
                epsilon_history.append(epsilon)

                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

            avg_reward = np.mean(rewards_per_episode[max(0, ep - 99):ep + 1])
            print(f"Episode {ep+1}/{episodes} | Reward: {ep_reward:.2f} | Avg(100): {avg_reward:.2f} | Epsilon: {epsilon:.2f}")

        env.close()
        torch.save(policy_dqn.state_dict(), self.model_path)

        sum_rewards = np.convolve(rewards_per_episode, np.ones(100)/100, mode='valid')
        plt.figure(figsize=(15, 4))

        plt.subplot(131)
        plt.plot(sum_rewards)
        plt.title("Avg reward (100-episode window)")
        plt.xlabel("Episode")
        plt.ylabel("Avg reward")

        plt.subplot(132)
        plt.plot(epsilon_history)
        plt.title("Epsilon decay")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")

        plt.subplot(133)
        self.plot_final_path(policy_dqn)
        plt.title("Final Path to Goal")

        plt.tight_layout()
        plt.savefig("gridworld_results.png")
        plt.show()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:
            target = torch.FloatTensor([reward]) if terminated else \
                torch.FloatTensor([reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state)).max().item()])

            current_q = policy_dqn(self.state_to_dqn_input(state))
            target_q = current_q.clone().detach()
            target_q[action] = target

            current_q_list.append(current_q)
            target_q_list.append(target_q)

        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def state_to_dqn_input(self, state: dict) -> torch.Tensor:
        size = self.grid_size - 1
        agent = state["agent"] / size
        target = state["target"] / size
        return torch.FloatTensor(np.concatenate([agent, target]))

    def plot_final_path(self, policy_dqn):
        env = GridWorldEnv(size=self.grid_size)
        state, _ = env.reset()
        path = [state["agent"].copy()]

        for _ in range(100):
            s = self.state_to_dqn_input(state)
            action = policy_dqn(s).argmax().item()
            state, _, done, _, _ = env.step(action)
            path.append(state["agent"].copy())
            if done:
                break

        path = np.array(path)
        x_path, y_path = path[:, 0], path[:, 1]
        z_path = [height_data[y, x] for x, y in path]

        X, Y = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE))
        Z = height_data

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap="Reds_r", alpha=0.8, edgecolor='none')

        ax.plot(x_path, y_path, z_path, color='green', linewidth=3, label='Agent Path')
        ax.scatter(x_path[0], y_path[0], z_path[0], color='green', s=50, label='Start')
        ax.scatter(x_path[-1], y_path[-1], z_path[-1], color='blue', s=50, label='Goal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Height')
        ax.legend()

    def load_model(self):
        model = DQN(in_states=4, h1_nodes=32, out_actions=4)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model

if __name__ == "__main__":
    dql = GridWorldDQL()
    dql.train(episodes=3000)

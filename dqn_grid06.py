from typing import Optional
import numpy as np
import pandas as pd
import gymnasium as gym


class Mars(gym.Env):

    def __init__(self, size: int = 10, heightmap_path: str = "heightmap.csv", height_penalty_factor: float = 1.0):
        self.size = size
        self.height_penalty_factor = height_penalty_factor

        # Load height map from CSV
        self.height_map = pd.read_csv(heightmap_path, header=None).values
        assert self.height_map.shape == (size, size), "Heightmap must match environment size"

        # Initialize positions
        self._agent_location = np.array([5, 2], dtype=np.int32)
        self._target_location = np.array([0, 0], dtype=np.int32)

        # Define observation and action spaces
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
        })

        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, 1]),   # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "target": self._target_location,
        }

    def _get_info(self):
        return {}

    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False

        # Get elevation-based penalty
        height_penalty = self.height_map[tuple(self._agent_location)]

        # Reward: 1 for reaching the goal, negative based on elevation otherwise
        reward = 1.0 if terminated else -self.height_penalty_factor * height_penalty

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def reset(self) -> tuple[dict[str, np.ndarray], dict[str, Optional[float]]]:
        self._agent_location = np.random.randint(0, self.size, size=2)
        self._target_location = np.random.randint(0, self.size, size=2)

        return self._get_obs(), {}

    def render(self, mode="human"):
        grid = np.full((self.size, self.size), ".", dtype=str)
        grid[tuple(self._target_location)] = "T"
        grid[tuple(self._agent_location)] = "A"
        print("\n".join(" ".join(row) for row in grid))

    def close(self):
        pass

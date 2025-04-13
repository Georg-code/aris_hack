from typing import Optional
import numpy as np
import gymnasium as gym


class Mars(gym.Env):

    def __init__(self, size: int = 5):
        # The size of the square grid
        self.size = size

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
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
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid bounds
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # An environment is completed if and only if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False
        reward = 1 if terminated else 0  # the agent is only reached at the end of the episode
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

        


    def reset(self,) -> tuple[dict[str, np.ndarray], dict[str, Optional[float]]]:
        # Randomly choose the agent and target locations
        self._agent_location = np.random.randint(0, self.size, size=2)
        self._target_location = np.random.randint(0, self.size, size=2)

        # Return the initial observation and an empty info dictionary
        return (
            {
                "agent": self._agent_location,
                "target": self._target_location,
            },
            {},
        )
    def render(self, mode="human"):
        # Create a grid representation of the environment
        grid = np.full((self.size, self.size), ".", dtype=str)

        # Mark the agent and target locations
        grid[tuple(self._agent_location)] = "A"
        grid[tuple(self._target_location)] = "T"

        # Print the grid to the console
        print("\n".join(" ".join(row) for row in grid))


    def close(self):
        pass



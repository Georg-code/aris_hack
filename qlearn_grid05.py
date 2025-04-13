import numpy as np
import random

# Define the gridworld environment
class GridWorld:
    def __init__(self):
        self.grid = np.array([
            [0, 0, 0, 1],  # Goal at (0, 3)
            [0, -1, 0, 0],  # Wall with reward -1
            [0, 0, 0, 0],
            [0, 0, 0, 0]  # Start at (3, 0)
        ])
        self.start_state = (3, 0)
        self.state = self.start_state

    def reset(self):
        self.state = self.start_state
        return self.state

    def is_terminal(self, state):
        return self.grid[state] == 1 or self.grid[state] == -1

    def get_next_state(self, state, action):
        next_state = list(state)
        if action == 0:  # Move up
            next_state[0] = max(0, state[0] - 1)
        elif action == 1:  # Move right
            next_state[1] = min(3, state[1] + 1)
        elif action == 2:  # Move down
            next_state[0] = min(3, state[0] + 1)
        elif action == 3:  # Move left
            next_state[1] = max(0, state[1] - 1)
        return tuple(next_state)

    def step(self, action):
        next_state = self.get_next_state(self.state, action)
        reward = self.grid[next_state]
        self.state = next_state
        done = self.is_terminal(next_state)
        return next_state, reward, done
    
class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.q_table = np.zeros((4, 4, 4))  # Q-values for each state-action pair
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, 3)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_value(self, state, action, reward, next_state):
        max_future_q = np.max(self.q_table[next_state])  # Best Q-value for next state
        current_q = self.q_table[state][action]
        # Q-learning formula
        self.q_table[state][action] = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )

env = GridWorld()
agent = QLearningAgent()

episodes = 100  # Number of training episodes

for episode in range(episodes):
    state = env.reset()  # Reset the environment at the start of each episode
    done = False

    while not done:
        action = agent.choose_action(state)  # Choose an action
        next_state, reward, done = env.step(action)  # Take the action and observe next state, reward
        agent.update_q_value(state, action, reward, next_state)  # Update Q-values
        state = next_state  # Move to the next state
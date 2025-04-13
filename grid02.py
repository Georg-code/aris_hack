import numpy as np

# Constants
BOARD_ROWS, BOARD_COLS = 5, 5
START = (4, 0)
CHECKPOINTS = [(0, 0), (0, 4)]
CHECKPOINT_REWARD = 10
STEP_PENALTY = -1

# Initial terrain matrix with rewards or penalties
initial_matrix = np.array([
    [10, -1, -1, -1, 10],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [0,  -1, -1, -1, -1]
])


class State:
    def __init__(self, state=START, checkpoints=None):
        self.state = state
        self.checkpoints = checkpoints if checkpoints else set()

    def nxtPosition(self, action):
        # Define possible moves based on action
        moves = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        next_state = (self.state[0] + moves[action][0], self.state[1] + moves[action][1])

        # Check if the move is within bounds
        if 0 <= next_state[0] < BOARD_ROWS and 0 <= next_state[1] < BOARD_COLS:
            return next_state
        return self.state

    def giveReward(self):
        # Provide rewards based on the current state's terrain and checkpoints visited
        reward = initial_matrix[self.state]
        if self.state in CHECKPOINTS and self.state not in self.checkpoints:
            reward += CHECKPOINT_REWARD
        if self.state == START and len(self.checkpoints) == len(CHECKPOINTS):
            reward += CHECKPOINT_REWARD
        return reward

    def isEndFunc(self):
        # Check if all checkpoints are visited and agent is back at the start
        return self.state == START and len(self.checkpoints) == len(CHECKPOINTS)


class Agent:
    def __init__(self, lr=0.2, exp_rate=0.3):
        self.actions = ["up", "down", "left", "right"]
        self.lr = lr  # Learning rate, adjustable
        self.exp_rate = exp_rate  # Exploration rate, adjustable
        self.state_values = {}
        self.reset()

    def reset(self):
        # Reset the state and checkpoints visited
        self.state = State()
        self.checkpoints = set()
        self.states_history = []

    def chooseAction(self):
        # Decide whether to explore randomly or exploit known information
        if np.random.uniform(0, 1) <= self.exp_rate:
            return np.random.choice(self.actions)
        else:
            values = {}
            for action in self.actions:
                next_pos = self.state.nxtPosition(action)
                key = (next_pos, tuple(sorted(self.checkpoints)))
                values[action] = self.state_values.get(key, 0)
            return max(values, key=values.get)

    def takeAction(self, action):
        # Execute chosen action and update checkpoints if needed
        next_pos = self.state.nxtPosition(action)
        if next_pos in CHECKPOINTS:
            self.checkpoints.add(next_pos)
        return State(next_pos, self.checkpoints.copy())

    def play(self, rounds=100):
        # Run training for specified rounds
        for _ in range(rounds):
            while True:
                action = self.chooseAction()
                next_state = self.takeAction(action)
                self.states_history.append((self.state.state, tuple(sorted(self.checkpoints))))

                reward = next_state.giveReward()
                key = (self.state.state, tuple(sorted(self.checkpoints)))

                # Update state values based on new reward
                next_key = (next_state.state, tuple(sorted(next_state.checkpoints)))
                self.state_values[key] = self.state_values.get(key, 0) + self.lr * (
                    reward + self.state_values.get(next_key, 0) - self.state_values.get(key, 0)
                )

                if next_state.isEndFunc():
                    self.reset()
                    break
                self.state = next_state

    def showValues(self):
        # Display state values in a grid
        for i in range(BOARD_ROWS):
            print('-------------------------')
            out = '| '
            for j in range(BOARD_COLS):
                key = ((i, j), tuple(sorted(CHECKPOINTS)))
                val = round(self.state_values.get(key, 0), 2)
                out += f'{val:6} | '
            print(out)
        print('-------------------------')


if __name__ == "__main__":
    # Example usage with adjustable parameters
    agent = Agent(lr=0.3, exp_rate=0.4)
    agent.play(10)
    agent.showValues()

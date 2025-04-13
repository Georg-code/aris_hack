import numpy as np

# Constants
BOARD_ROWS, BOARD_COLS = 5, 5
START = (4, 0)
CHECKPOINTS = [(0, 0), (0, 4)]
CHECKPOINT_REWARD = 5   # Adjusted reward
STEP_PENALTY = -2       # Adjusted penalty

# Initial terrain matrix with rewards or penalties
initial_matrix = np.array([
    [5, -2, -2, -2, 5],
    [-2, -2, -2, -2, -2],
    [-2, -2, -2, -2, -2],
    [-2, -2, -2, -2, -2],
    [0, -2, -2, -2, -2]
])


class State:
    def __init__(self, state=START, checkpoints=None):
        self.state = state
        self.checkpoints = checkpoints if checkpoints else set()

    def nxtPosition(self, action):
        moves = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        next_state = (self.state[0] + moves[action][0], self.state[1] + moves[action][1])

        if 0 <= next_state[0] < BOARD_ROWS and 0 <= next_state[1] < BOARD_COLS:
            return next_state
        return self.state

    def giveReward(self):
        reward = initial_matrix[self.state]
        if self.state in CHECKPOINTS and self.state not in self.checkpoints:
            reward += CHECKPOINT_REWARD
        if self.state == START and len(self.checkpoints) == len(CHECKPOINTS):
            reward += CHECKPOINT_REWARD
        return reward

    def isEndFunc(self):
        return self.state == START and len(self.checkpoints) == len(CHECKPOINTS)


class Agent:
    def __init__(self, lr=0.2, exp_rate=0.3):
        self.actions = ["up", "down", "left", "right"]
        self.lr = lr
        self.exp_rate = exp_rate
        self.state_values = {}
        self.reset()

    def reset(self):
        self.state = State()
        self.checkpoints = set()
        self.states_history = []

    def chooseAction(self):
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
        next_pos = self.state.nxtPosition(action)
        if next_pos in CHECKPOINTS:
            self.checkpoints.add(next_pos)
        return State(next_pos, self.checkpoints.copy())

    def play(self, rounds=100):
        for _ in range(rounds):
            while True:
                action = self.chooseAction()
                next_state = self.takeAction(action)
                self.states_history.append((self.state.state, tuple(sorted(self.checkpoints))))

                reward = next_state.giveReward()
                key = (self.state.state, tuple(sorted(self.checkpoints)))

                next_key = (next_state.state, tuple(sorted(next_state.checkpoints)))
                self.state_values[key] = self.state_values.get(key, 0) + self.lr * (
                    reward + self.state_values.get(next_key, 0) - self.state_values.get(key, 0)
                )

                if next_state.isEndFunc():
                    self.reset()
                    break
                self.state = next_state

    def showValues(self):
        for i in range(BOARD_ROWS):
            print('-------------------------')
            out = '| '
            for j in range(BOARD_COLS):
                key = ((i, j), tuple(sorted(CHECKPOINTS)))
                val = round(self.state_values.get(key, 0), 2)
                out += f'{val:6} | '
            print(out)
        print('-------------------------')

    def getOptimalPath(self, max_steps=50):
        current_state = State()
        checkpoints_visited = set()
        path = []
        visited_states = set()
        steps = 0

        while not current_state.isEndFunc() and steps < max_steps:
            state_key = (current_state.state, tuple(sorted(checkpoints_visited)))
            if state_key in visited_states:
                print("Detected loop, exiting.")
                break
            visited_states.add(state_key)

            best_action = None
            best_value = float('-inf')

            for action in self.actions:
                next_pos = current_state.nxtPosition(action)
                key = (next_pos, tuple(sorted(checkpoints_visited)))
                value = self.state_values.get(key, float('-inf'))
                if value > best_value:
                    best_value = value
                    best_action = action

            if best_action is None:
                print("No viable moves left, exiting.")
                break

            path.append(best_action)
            current_pos = current_state.nxtPosition(best_action)

            if current_pos in CHECKPOINTS:
                checkpoints_visited.add(current_pos)

            current_state = State(current_pos, checkpoints_visited.copy())
            steps += 1

        if steps >= max_steps:
            print("Maximum steps reached without completing path.")

        return path


if __name__ == "__main__":
    agent = Agent(lr=0.3, exp_rate=0.4)
    agent.play(50)
    agent.showValues()
    optimal_path = agent.getOptimalPath()
    print("Optimal Path:", optimal_path)

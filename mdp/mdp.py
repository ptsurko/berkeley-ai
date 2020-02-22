from collections import defaultdict

def expected_utility(mdp, values, state, action):
    return sum(prob * values[new_state] for (new_state, prob) in mdp.transitions(state, action))

argmax = max


class Agent:
    def __init__(self, gamma):
        self.gamma = gamma
        self.values = defaultdict(int)

    def get_q_value(self, mdp, state, action):
        transitions = mdp.transitions(state, action)
        if not len(transitions):
            return 0

        return sum(prob * (mdp.get_reward(state) + self.gamma * self.values[new_state]) for (new_state, prob) in mdp.transitions(state, action))

    def get_value(self, mdp, state):
        actions = mdp.actions(state)
        if not len(actions):
            return 0

        return max(self.get_q_value(mdp, state, action) for action in actions)


    def learn(self, mdp):
        raise NotImplementedError("Please Implement this method")

class ValueIterationAgent(Agent):
    def __init__(self, epsilon=0.001, max_iterations=None, gamma=0.9):
        super().__init__(gamma) 
        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def learn(self, mdp):
        convergence = False
        self.values = defaultdict(int)

        iteration = 0
        while not convergence:
            new_values = defaultdict(int)

            delta = 0
            for state in mdp.states():
                new_values[state] = self.get_value(mdp, state)
                delta = max(delta, abs(new_values[state] - new_values[state]))

            self.values = new_values
            iteration += 1

            if self.max_iterations is not None and iteration >= self.max_iterations:
                convergence = True
            if self.max_iterations is None and delta <= self.epsilon * (1 - self.gamma) / self.gamma:
                convergence = True


class PolicyIterationAgent(Agent):
    def __init__(self, initial_action, gamma=0.9):
        super().__init__(gamma) 
        self.initial_action = initial_action

    def policy_evaluation(self, mdp, policy, max_k=9):
        values = defaultdict(int)

        for _ in range(max_k):
            new_values = values.copy()

            for state in mdp.states():
                new_values[state] = self.get_value(mdp, state)

            values = new_values

        return values

    def learn(self, mdp):
        policy = {state: self.initial_action for state in mdp.states()}

        while True:
            changed = False

            self.values = self.policy_evaluation(mdp, policy)
            new_policy = policy.copy()
            for state in mdp.states():
                actions = mdp.actions(state)
                max_action = None
                if len(actions):
                    max_action = argmax(
                        mdp.actions(state),
                        key=lambda action: self.get_q_value(mdp, state, action)
                    )

                new_policy[state] = max_action
                if policy[state] != new_policy[state]:
                    changed = True

            policy = new_policy

            if not changed:
                break


class ReinforcementAgent(Agent):
    def __init__(self, initial_state=None, samples=[], alpha=0.9, gamma=1):
        super().__init__(gamma) 

        self.samples = samples
        self.initial_state = initial_state
        self.alpha = alpha


    def get_q_value(self, mdp, state, action):
        transitions = mdp.transitions(state, action)
        if not len(transitions):
            return 0

        return sum(prob * (mdp.get_reward(state) + self.gamma * self.values[new_state]) for (new_state, prob) in mdp.transitions(state, action))

    def get_value(self, mdp, state):
        actions = mdp.actions(state)
        if not len(actions):
            return 0

        return max(self.get_q_value(mdp, state, action) for action in actions)

    def learn(self, mdp):
        self.values = {state: 0 for state in mdp.states()}
        self.q_values = {state: [0] * len(mdp.actions(state)) for state in mdp.states()}
        
        for sample in self.samples:
            state = self.initial_state
            new_values = {}
            new_q_values = {key: value.copy() for key, value in self.q_values.items()}

            for action in sample:
                next_state = mdp.next_state(state, action)
                actions = mdp.actions(state)
                action_index = actions.index(action)

                sample = mdp.get_reward(state) + self.gamma * (max(self.q_values[next_state]) if len(self.q_values[next_state]) else 0)
                q_value = (1 - self.alpha) * self.q_values[state][action_index] + self.alpha * sample
                
                new_q_values[state][action_index] = q_value
                new_values[state] = max(new_q_values[state])
                state = next_state

            self.values = new_values
            self.q_values = new_q_values
    

# Rewrite using q-values
def best_policy(mdp, values):
    return [
        (state, argmax(
            mdp.actions(state),
            key=lambda action: sum(t * values[new_state] for (new_state, t) in mdp.transitions(state, action))))
        for state in mdp.states() if state is not Mdp.TERMINAL
    ]


class Mdp:
    EXIT = 'Exit'
    TERMINAL = (None,)

    def states(self):
        raise NotImplementedError("Please Implement this method")

    def transitions(self, state, action):
        raise NotImplementedError("Please Implement this method")

    def get_reward(self, state):
        raise NotImplementedError("Please Implement this method")

    def isTerminal(self, state):
        return state is Mdp.TERMINAL

    def actions(self, state):
        raise NotImplementedError("Please Implement this method")

    def next_state(self, state, action):
        raise NotImplementedError("Please Implement this method")

class CarRacingMdp(Mdp):
    pass


class GridMove:
    TOP = 'Top'
    RIGHT = 'Right'
    BOTTOM = 'Bottom'
    LEFT = 'Left'
    EXIT = Mdp.EXIT


class GridMdp(Mdp):
    def __init__(self, grid, living_reward=0, noise=0.2):
        self.grid = grid
        self.living_reward = living_reward
        self.noise = noise

    def states(self):
        states = [Mdp.TERMINAL]
        for row in range(len(self.grid)):
            for col in range(len(self.grid[0])):
                if self.grid[row][col] is not None:
                    states.append((row, col))
        
        return states

    def transitions(self, state, action):
        outcomes = []
        if state is not Mdp.TERMINAL:
            for possible_action, action_probability in self.get_possible_actions(action, self.noise):
                new_state = self.next_state(state, possible_action)
                outcomes.append((new_state, action_probability))

        return outcomes

    def get_reward(self, state):
        if state == Mdp.TERMINAL:
            return 0.0

        row, col = state
        cell = self.grid[row][col]
        if (type(cell) == int or type(cell) == float) and cell != 0:
            return cell
        return self.living_reward

    def actions(self, state):
        if state == Mdp.TERMINAL:
            return []
        row, col = state
        cell = self.grid[row][col]
        if (type(cell) == int or type(cell) == float) and cell != 0:
            return [Mdp.EXIT]
        return [GridMove.TOP, GridMove.RIGHT, GridMove.BOTTOM, GridMove.LEFT]

    def get_possible_actions(self, move, noise):
        if move == GridMove.TOP:
            return [(GridMove.TOP, 1 - noise), (GridMove.RIGHT, noise / 2), (GridMove.LEFT, noise / 2)]
        if move == GridMove.RIGHT:
            return [(GridMove.RIGHT, 1 - noise), (GridMove.TOP, noise / 2), (GridMove.BOTTOM, noise / 2)]
        if move == GridMove.BOTTOM:
            return [(GridMove.BOTTOM, 1 - noise), (GridMove.RIGHT, noise / 2), (GridMove.LEFT, noise / 2)]
        if move == GridMove.LEFT:
            return [(GridMove.LEFT, 1 - noise), (GridMove.TOP, noise / 2), (GridMove.BOTTOM, noise / 2)]
        if move == GridMove.EXIT:
            return [(GridMove.EXIT, 1)]

        raise Exception("{} move is not supported".format(move))
        

    def next_state(self, state, action):
        row, col = state
        max_row = len(self.grid) - 1
        max_col = len(self.grid[0]) - 1
        if action == GridMove.TOP:
            if row - 1 >= 0 and self.grid[row - 1][col] is not None:
                return (row - 1, col)

        if action == GridMove.RIGHT:
            if col + 1 <= max_col and self.grid[row][col + 1] is not None:
                return (row, col + 1)

        if action == GridMove.BOTTOM:
            if row + 1 <= max_row and self.grid[row + 1][col] is not None:
                return (row + 1, col)

        if action == GridMove.LEFT:
            if col - 1 >= 0 and self.grid[row][col - 1] is not None:
                return (row, col - 1)

        if action == GridMove.EXIT:
            return Mdp.TERMINAL

        return state

    def to_action_grid(self, actions):
        chars = {
            GridMove.TOP: '^',
            GridMove.RIGHT: '>',
            GridMove.BOTTOM: 'v',
            GridMove.LEFT: '<',
            GridMove.EXIT: 'X',
        }
        result = []
        for row in range(len(self.grid)):
            row_moves = []
            for col in range(len(self.grid[0])):
                for (state, action) in actions:
                    if state == (row, col):
                        row_moves.append(chars[action])
                        break
                else:
                    row_moves.append('.')

            result.append(row_moves)
        return result

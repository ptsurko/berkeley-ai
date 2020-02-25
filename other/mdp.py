from collections import defaultdict

def expected_utility(mdp, values, state, action):
    return sum(prob * values[new_state] for (new_state, prob) in mdp.transitions(state, action))

argmax = max


class Agent:
    def get_q_value(self, state, action):
        raise NotImplementedError("Please Implement this method")

    def get_value(self, state):
        raise NotImplementedError("Please Implement this method")

    def get_action(self, state):
        raise NotImplementedError("Please Implement this method")

class ValueIterationAgent(Agent):
    def __init__(self, mdp, max_iterations=None, gamma=0.9):
        self.mdp = mdp
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.values = defaultdict(int)

        for _ in range(max_iterations):
            new_values = defaultdict(int)

            for state in self.mdp.states():
                new_values[state] = self.compute_value(state)

            self.values = new_values

    def get_q_value(self, state, action):
        transitions = list(self.mdp.transitions(state, action))
        if not len(transitions):
            return 0

        return sum(prob * (self.mdp.get_reward(state) + self.gamma * self.get_value(new_state)) for (new_state, prob) in self.mdp.transitions(state, action))

    def get_value(self, state):
        return self.values[state]

    def get_action(self, state):
        actions = self.mdp.actions(state)
        if not len(actions):
            return None

        return max(actions, key=lambda action: self.get_q_value(state, action))

    def compute_value(self, state):
        actions = self.mdp.actions(state)
        if not len(actions):
            return 0

        return max(self.get_q_value(state, action) for action in actions)

class PolicyIterationAgent(Agent):
    def __init__(self, mdp, initial_action, gamma=0.9):
        self.mdp = mdp
        self.initial_action = initial_action
        self.gamma = gamma
        self.values = defaultdict(int)

        self.policy_iteration()

    def policy_evaluation(self, policy, max_k=9):
        for _ in range(max_k):
            new_values = defaultdict(int)

            for state in self.mdp.states():
                new_values[state] = self.compute_value(state)

            self.values = new_values

    def policy_iteration(self):
        policy = {state: self.initial_action for state in self.mdp.states()}

        while True:
            changed = False

            self.policy_evaluation(policy)
            new_policy = policy.copy()
            for state in self.mdp.states():
                new_policy[state] = self.compute_action(state)
                changed = changed or (policy[state] != new_policy[state])

            policy = new_policy

            if not changed:
                break

    def get_q_value(self, state, action):
        transitions = self.mdp.transitions(state, action)
        if not len(transitions):
            return 0

        return sum(prob * (self.mdp.get_reward(state) + self.gamma * self.get_value(new_state)) for (new_state, prob) in self.mdp.transitions(state, action))

    def get_value(self, state):
        return self.values[state]

    def get_action(self, state):
        return self.compute_action(state)

    def compute_value(self, state):
        actions = self.mdp.actions(state)
        if not len(actions):
            return 0

        return max(self.get_q_value(state, action) for action in actions)

    def compute_action(self, state):
        actions = self.mdp.actions(state)
        if not len(actions):
            return None

        return max(actions, key=lambda action: self.get_q_value(state, action))

class SampleQLearningAgent(Agent):
    def __init__(self, mdp, initial_state=None, samples=[], alpha=0.9, gamma=1):
        self.mdp = mdp
        self.gamma = gamma
        self.samples = samples
        self.initial_state = initial_state
        self.alpha = alpha
        self.q_values = defaultdict(int)

        for actions in samples:
            self.update(actions)

    def get_q_value(self, state, action):
        return self.q_values[(state, action)]

    def get_value(self, state):
        actions = self.mdp.actions(state)
        if not len(actions):
            return 0

        return max(self.get_q_value(state, action) for action in actions)

    def update(self, actions):
        state = self.initial_state
        new_q_values = self.q_values.copy()

        for action in actions:
            next_state = self.mdp.next_state(state, action)

            sample = self.mdp.get_reward(state) + self.gamma * self.get_value(next_state)
            
            q_value = (1 - self.alpha) * self.get_q_value(state, action) + self.alpha * sample
            
            new_q_values[(state, action)] = q_value
            state = next_state

        self.q_values = new_q_values

class ExploreQLearningAgent(Agent):
    pass


# Rewrite using q-values
def best_policy(agent, state):
    return agent.get_action(state)


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
        transitions = []
        for possible_action, action_probability in self.get_possible_actions(action, self.noise):
            new_state = self.next_state(state, possible_action)
            transitions.append((new_state, action_probability))
        return transitions

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
        if cell is None:
            return []
        elif (type(cell) == int or type(cell) == float) and cell != 0:
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

    def to_action_grid(self, agent):
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
                action = agent.get_action((row, col))
                if action and action in chars:
                    row_moves.append(chars[action])
                else:
                    row_moves.append('.')

            result.append(row_moves)
        return result

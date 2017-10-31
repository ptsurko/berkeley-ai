

def expected_utility(mdp, values, state, action):
    return sum(t * values[new_state] for (t, new_state) in mdp.T(state, action))


argmax = max


def value_iteration(mdp, epsilon=0.001):
    convergence = False
    values = {state: 0 for state in mdp.states()}

    gamma = mdp.gamma

    while not convergence:
        new_values = values.copy()
        delta = 0
        for state in mdp.states():
            actions = mdp.actions(state)
            max_q_value = None

            for action in actions:
                q_value = sum(t * values[new_state] for (t, new_state) in mdp.T(state, action))

                max_q_value = max(q_value, max_q_value)

            value = mdp.R(state) + gamma * (max_q_value or 0)
            new_values[state] = value

            delta = max(delta, abs(values[state] - new_values[state]))

        values = new_values
        if delta <= epsilon * (1 - gamma) / gamma:
            return new_values


def policy_evaluation(mdp, policy, max_k=20):
    values = {state: 0 for state in mdp.states()}

    for k in xrange(max_k):
        new_values = values.copy()

        for state in mdp.states():
            q_value = expected_utility(mdp, values, state, policy[state])

            value = mdp.R(state) + mdp.gamma * q_value
            new_values[state] = value

        values = new_values

    return values


def policy_iteration(mdp, initial_action):
    policy = {state: initial_action for state in mdp.states()}
    while True:
        changed = False

        values = policy_evaluation(mdp, policy)
        new_policy = policy.copy()
        for state in mdp.states():
            max_action = argmax(
                mdp.actions(state),
                key=lambda action: sum(t * values[new_state] for (t, new_state) in mdp.T(state, action))
            )

            new_policy[state] = max_action
            if policy[state] != new_policy[state]:
                changed = True

        policy = new_policy

        if not changed:
            return values


def best_policy(mdp, values):
    return [
        (state, argmax(
            mdp.actions(state),
            key=lambda action: sum(t * values[new_state] for (t, new_state) in mdp.T(state, action))))
        for state in mdp.states()
    ]


class Mdp:
    def __init__(self, transitions, rewards, terminals=[], gamma=1):
        self._transitions = transitions
        self._rewards = rewards
        self._terminals = terminals
        self.gamma = gamma

    def states(self):
        return self._transitions.keys()

    def T(self, state, action):
        if state in self._terminals or action is None:
            return [(0, state)]

        return [(t, probable_state) for probable_state, t in self._transitions[state][action]]

    def R(self, state):
        return self._rewards[state]

    def actions(self, state):
        if state in self._terminals:
            return [None]

        return self._transitions[state].keys()


class CarRacingMdp(Mdp):
    pass


class GridMove:
    TOP = 'Top'
    RIGHT = 'Right'
    BOTTOM = 'Bottom'
    LEFT = 'Left'


class GridMdp(Mdp):
    def __init__(self, grid, terminals=[], gamma=1, probability=0.8, side_probability=0.1):
        self._grid = grid
        transitions = {}
        rewards = {}
        actions = [GridMove.TOP, GridMove.RIGHT, GridMove.BOTTOM, GridMove.LEFT]

        for row in xrange(len(grid)):
            for col in xrange(len(grid[0])):
                state = (row, col)
                if grid[row][col] is None:
                    continue

                transition = {}
                if (row, col) not in terminals:
                    for action in actions:
                        outcomes = []

                        for possible_action, action_probability in GridMdp.get_possible_actions(action, probability, side_probability):
                            new_state = GridMdp.move_to(grid, (row, col), possible_action)
                            outcomes.append((new_state, action_probability))

                        transition[action] = outcomes

                transitions[state] = transition
                rewards[state] = grid[row][col]

        Mdp.__init__(self, transitions, rewards, terminals=terminals, gamma=gamma)

    @staticmethod
    def get_possible_actions(move, probability, side_probability):
        if move == GridMove.TOP:
            return [(GridMove.TOP, probability), (GridMove.RIGHT, side_probability), (GridMove.LEFT, side_probability)]
        if move == GridMove.RIGHT:
            return [(GridMove.RIGHT, probability), (GridMove.TOP, side_probability), (GridMove.BOTTOM, side_probability)]
        if move == GridMove.BOTTOM:
            return [(GridMove.BOTTOM, probability), (GridMove.RIGHT, side_probability), (GridMove.LEFT, side_probability)]

        return [(GridMove.LEFT, probability), (GridMove.TOP, side_probability), (GridMove.BOTTOM, side_probability)]


    @staticmethod
    def move_to(grid, state, move):
        row, col = state
        max_row = len(grid) - 1
        max_col = len(grid[0]) - 1
        if move == GridMove.TOP:
            if row - 1 >= 0 and grid[row - 1][col] is not None:
                return (row - 1, col)

        if move == GridMove.RIGHT:
            if col + 1 <= max_col and grid[row][col + 1] is not None:
                return (row, col + 1)

        if move == GridMove.BOTTOM:
            if row + 1 <= max_row and grid[row + 1][col] is not None:
                return (row + 1, col)

        if move == GridMove.LEFT:
            if col - 1 >= 0 and grid[row][col - 1] is not None:
                return (row, col - 1)

        return state

    def to_action_grid(self, actions):
        chars = {
            'Top': '^',
            'Right': '>',
            'Bottom': 'v',
            'Left': '<',
            None: '.',
        }
        result = []
        for row in xrange(len(self._grid)):
            row_moves = []
            for col in xrange(len(self._grid[0])):
                for (state, action) in actions:
                    if state == (row, col):
                        row_moves.append(chars[action])
                        break
                else:
                    row_moves.append('.')

            result.append(row_moves)
        return result

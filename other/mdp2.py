import copy
import sys

def build_initial_state(grid):
    state = []
    for grid_row in grid:
        state_row = []
        state_col = None
        for grid_col in grid_row:
            if grid_col == 'X':
                state_col = None
            elif grid_col == ' ':
                state_col = {"V": 0, "R": 0, "isFinal": False, "Qs": [0,0,0,0], "policy": None}
            else:
                state_col = {"V": 0, "R": grid_col, "isFinal": True, "Qs": None, "policy": None}
            state_row.append(state_col)
        state.append(state_row)
    return state


def get_transitions(action, noise=0.2):
    if action == 't':
        return [('t', 1-noise), ('r', noise / 2), ('l', noise / 2)]
    elif action == 'r':
        return [('r', 1-noise), ('t', noise / 2), ('b', noise / 2)]
    elif action == 'b':
        return [('b', 1-noise), ('r', noise / 2), ('l', noise / 2)]

    return [('l', 1-noise), ('t', noise / 2), ('b', noise / 2)]


def get_action_next_pos(row, col, action):
    row_num = len(grid)
    col_num = len(grid[0])

    if action == 't':
        nr, nc = (max(0, row-1), col)
    elif action == 'r':
        nr, nc = (row, min(col_num-1, col+1))
    elif action == 'b':
        nr, nc = (min(row_num-1, row+1), col)
    else:
        nr, nc = (row, max(0, col-1))

    if grid[nr][nc] == 'X':
        return row, col
    return nr, nc


def get_max_len_of_V(state, decimal_points=3):
    max_len = 0
    format = "{0:." + str(decimal_points) + "f}"

    def get_max_len(val):
        return len(format.format(val))

    for r in state:
        for c in r:
            if c is None:
                continue
            elif c["isFinal"]:
                max_len = max(max_len, get_max_len(c["R"]))
            else:
                # print("c = ", c)
                # print("c[Qs] = ", c["Qs"])
                lengths = map(lambda q_val: get_max_len(q_val), c["Qs"])
                max_len = max(max_len, *lengths)

    return max_len


def format_num(number, max_len, decimal_points):
    value = ("{0:." + str(decimal_points) + "f}").format(round(number, decimal_points))
    value = ' ' * (max_len - len(value)) + value
    return value


def get_state_str(state, decimal_points=3):
    max_len = get_max_len_of_V(state, decimal_points)
    s = ''
    space = ' '
    empty = ' ' * max_len
    for r in state:
        line1 = ''
        line2 = ''
        line3 = ''
        for c in r:
            if c is None:
                line1 += empty + space + empty + space + empty + '|'
                line2 += empty + space + empty + space + empty + '|'
                line3 += empty + space + empty + space + empty + '|'
            elif c["isFinal"] is True:
                line1 += empty + space + empty + space + empty + '|'
                line2 += empty + space + format_num(c["R"], max_len, decimal_points) + space + empty + '|'
                line3 += empty + space + empty + space + empty + '|'
            else:
                top_val, right_val, bottom_val, left_val = c["Qs"]
                line1 += empty + space + format_num(top_val, max_len, decimal_points) + space + empty + '|'
                line2 += format_num(left_val, max_len, decimal_points) + space + format_num(c["V"], max_len, decimal_points) + space + format_num(right_val, max_len, decimal_points) + '|'
                line3 += empty + space + format_num(bottom_val, max_len, decimal_points) + space + empty + '|'

        s += line1 + '\n'
        s += line2 + '\n'
        s += line3 + '\n'
        s += '-' * len(line3) + '\n'

    return s

arrows = {
    't': '↑',
    'r': '→',
    'b': '↓',
    'l': '←'
}

def get_poicy_str(p):
    s = ''
    for row in p:
        row_str = ''
        for col in row:
            if col in arrows:
                row_str += arrows[col]
            else:
                row_str += col
            row_str += '|'
        s += row_str + '\n'
        s += '-' * len(row_str) + '\n'
    return s

def get_policy_from_state(state):
    policy = []
    for row in state:
        policy_row = []
        for col in row:
            if col is None:
                policy_row.append(' ')
            elif col["isFinal"] is True:
                policy_row.append('X')
            else:
                policy_row.append(col["policy"])
        policy.append(policy_row)

    return policy


grid = [
    # [' ',   1]
    [' ', ' ', ' ',   1],
    [' ', 'X', ' ',  -1],
    [' ', ' ', ' ', ' '],
]

noise = 0.2
discount = 0.9

actions = ['t', 'r', 'b', 'l']
state = build_initial_state(grid)

iterations = 100
convergence_delta = 0.001

def value_iteration(state, iterations=100, convergence_delta=0.001, verbose=False):
    for iter in range(iterations):
        new_state = []
        isConverged = True
        for r, row in enumerate(state):
            new_state_row = []
            for c, col in enumerate(row):
                new_state_col = copy.copy(col)
                if new_state_col is None:
                    pass
                elif new_state_col["isFinal"] is True:
                    new_state_col["V"] = col["R"]
                else:
                    v = -sys.maxsize - 1
                    Q_values = []
                    for action in actions:
                        q = 0
                        for (transition_action, noise) in get_transitions(action):
                            tr, tc = get_action_next_pos(r, c, transition_action)
                            q += noise * (discount * state[tr][tc]["V"])

                        Q_values.append(q)
                        v = max(v, q)

                        new_state_col[action] = q  # remove
                    new_state_col["V"] = v
                    new_state_col["Qs"] = Q_values

                    max_action_index = Q_values.index(v)
                    if max_action_index >= 0:
                        new_state_col["policy"] = actions[max_action_index]
                    isConverged = isConverged and new_state_col["V"] != 0 and (abs(new_state_col["V"] - col["V"]) <= convergence_delta)
                new_state_row.append(new_state_col)
            new_state.append(new_state_row)
        state = new_state

        if verbose:
            print("iteration: %s" % (iter + 1))
            print(get_state_str(state, 2))
            print(get_poicy_str(get_policy_from_state(state)))

        if isConverged:
            print("Converged on iteration: ", (iter + 1))
            break

    policy = get_policy_from_state(state)
    return state, policy

def init_random_policy(state, policy='t'):
    for row in state:
        for col in row:
            if col is not None and col["isFinal"] != True:
                col["policy"] = policy

def policy_iteration(state, iterations=100, sub_iterations=10, verbose=False):
    init_random_policy(state)

    for iter in range(iterations):
        # policy evaluation
        for _ in range(sub_iterations):
            new_state = []
            for r, row in enumerate(state):
                new_state_row = []
                for c, col in enumerate(row):
                    new_state_col = copy.copy(col)
                    if new_state_col is None:
                        pass
                    elif new_state_col["isFinal"] is True:
                        new_state_col["V"] = col["R"]
                    else:
                        v = 0
                        for (transition_action, noise) in get_transitions(col["policy"]):
                            tr, tc = get_action_next_pos(r, c, transition_action)
                            v += noise * (discount * state[tr][tc]["V"])
                        new_state_col["V"] = v
                    new_state_row.append(new_state_col)
                new_state.append(new_state_row)

        # policy extraction
        new_state = []
        isConverged = True
        for r, row in enumerate(state):
            new_state_row = []
            for c, col in enumerate(row):
                new_state_col = copy.copy(col)
                if new_state_col is None:
                    pass
                elif new_state_col["isFinal"] is True:
                    new_state_col["V"] = col["R"]
                else:
                    v = -sys.maxsize - 1
                    Q_values = []
                    for action in actions:
                        q = 0
                        for (transition_action, noise) in get_transitions(action):
                            tr, tc = get_action_next_pos(r, c, transition_action)
                            q += noise * (discount * state[tr][tc]["V"])

                        Q_values.append(q)
                        v = max(v, q)

                        new_state_col[action] = q  # remove
                    new_state_col["V"] = v
                    new_state_col["Qs"] = Q_values

                    max_action_index = Q_values.index(v)
                    if max_action_index >= 0:
                        new_state_col["policy"] = actions[max_action_index]
                    isConverged = isConverged and new_state_col["policy"] == col["policy"]
                new_state_row.append(new_state_col)
            new_state.append(new_state_row)
        state = new_state

        if verbose:
            print("iteration: %s" % (iter + 1))
            print(get_state_str(state, 2))
            print(get_poicy_str(get_policy_from_state(state)))

        if iter > 0 and isConverged:
            print("Converged on iteration: ", (iter + 1))
            break

    policy = get_policy_from_state(state)
    return state, policy

value_iteration(state, verbose=False)
policy_iteration(state, iterations=20, verbose=False)
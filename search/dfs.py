from util import Stack


def dfs_search(problem):
    frontier = Stack()
    start_state = problem.getStartState()

    frontier.append((start_state, [start_state]))
    exploring_states = [start_state]
    while True:
        if len(frontier) == 0:
            return None

        current_state, actions = frontier.pop()

        if problem.isGoalState(current_state):
            return actions

        for successor_state, _ in reversed(list(problem.getSuccessors(current_state))):
            if successor_state not in exploring_states:
                exploring_states.append(successor_state)
                frontier.append((successor_state, actions + [successor_state]))


def _dfs_search_recursive_internal(problem, state, actions, explored):
    if problem.isGoalState(state):
        return actions

    for successor_state, _ in problem.getSuccessors(state):
        if successor_state not in explored:
            explored.add(successor_state)
            res = _dfs_search_recursive_internal(problem, successor_state, actions + [successor_state], explored)
            if res is not None:
                return res


def dfs_search_recursive(problem):
    start_state = problem.getStartState()
    return _dfs_search_recursive_internal(problem, start_state, [start_state], set([start_state]))

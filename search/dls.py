from util import Stack


def dls_search_internal(problem, cutoff=100):
    frontier = Stack()
    start_state = problem.getStartState()

    frontier.append((start_state, [start_state], 0))
    exploring_states = [start_state]
    cutoff_occurred = False
    while True:
        if len(frontier) == 0:
            return None, cutoff_occurred

        current_state, actions, depth = frontier.pop()

        if problem.isGoalState(current_state):
            return actions, cutoff_occurred

        if depth == cutoff:
            cutoff_occurred = True
            continue

        for successor_state, _ in reversed(list(problem.getSuccessors(current_state))):
            if successor_state not in exploring_states:
                exploring_states.append(successor_state)
                frontier.append((successor_state, actions + [successor_state], depth + 1))


def dls_search(problem, cutoff=100):
    return dls_search_internal(problem, cutoff)[0]
from util import PriorityQueue


def null_heuristic(_):
    return 0


def ucs_search(problem, heuristic=null_heuristic):
    frontier = PriorityQueue()
    state_info = {}
    start_state = problem.getStartState()

    frontier.push(heuristic(start_state), start_state)
    state_info[start_state] = (0, [start_state])

    while True:
        if frontier.empty():
            return None

        current_state = frontier.pop()
        current_cost, actions = state_info[current_state]

        if problem.isGoalState(current_state):
            return actions

        for successor_state, successor_cost in problem.getSuccessors(current_state):
            successor_full_cost = current_cost + successor_cost
            if successor_state not in state_info:
                frontier.push(successor_full_cost + heuristic(successor_state), successor_state)
                state_info[successor_state] = (successor_full_cost, actions + [successor_state])
            elif state_info[successor_state][0] > successor_full_cost:
                frontier.update(successor_full_cost + heuristic(successor_state), successor_state)
                state_info[successor_state] = (successor_full_cost, actions + [successor_state])

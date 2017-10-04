from util import Queue


def bfs_search(problem):
    frontier = Queue()
    start_state = problem.getStartState()

    frontier.put((start_state, [start_state]))
    exploring_states = [start_state]
    while True:
        if frontier.empty():
            return None

        current_state, actions = frontier.get()

        if problem.isGoalState(current_state):
            return actions

        for successor_state, _ in problem.getSuccessors(current_state):
            if successor_state not in exploring_states:
                exploring_states.append(successor_state)
                frontier.put((successor_state, actions + [successor_state]))


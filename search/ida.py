from util import Stack


def null_heuristic(_):
    return 0


def min_number(number1, number2):
    return number1 if number1 < number2 else number2


# TODO: implement recursive version of this algorithm
def ida_search(problem, heuristic=null_heuristic):
    cutoff = heuristic(problem.getStartState())

    while True:
        min_cutoff = float("inf")

        frontier = Stack()
        start_state = problem.getStartState()

        frontier.append((start_state, [start_state], 0))
        while True:
            if len(frontier) == 0:
                break

            current_state, actions, current_cost = frontier.pop()
            current_estimated_cost = current_cost + heuristic(current_state)

            if current_estimated_cost > cutoff:
                min_cutoff = min_number(current_estimated_cost, min_cutoff)

                continue

            if problem.isGoalState(current_state):
                return actions

            for successor_state, successor_cost in reversed(list(problem.getSuccessors(current_state))):
                successor_full_cost = current_cost + successor_cost
                if successor_state not in actions:
                    frontier.append((successor_state, actions + [successor_state], successor_full_cost))

        if min_cutoff == float("inf"):
            return None

        cutoff = min_cutoff

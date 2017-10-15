import random
from chess_problem import attacking_cost


def hill_climbing(problem, cost_func=attacking_cost, max_attempts=10):
    current = problem.getStartState()
    while True:
        if problem.isGoalState(current):
            return current

        current_cost = cost_func(current)
        next_currents = []
        for successor in problem.getSuccessors(current):
            successor_cost = cost_func(successor)
            if successor_cost <= current_cost:
                if len(next_currents) > 0:
                    if successor_cost < next_currents[0][1]:
                        next_currents = []
                    elif successor_cost > next_currents[0][1]:
                        continue

                next_currents.append((successor, successor_cost))
        if len(next_currents) == 0:
            return None

        current = next_currents[random.randrange(0, len(next_currents))][0]

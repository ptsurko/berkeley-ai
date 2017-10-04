import math
from romania import map, locations
from search_problem import SearchProblem
from stats import NullStats


def create_heuristic_to(goal):
    return lambda state: math.sqrt((locations[goal][0] - locations[state][0]) ** 2 + (locations[goal][1] - locations[state][1]) ** 2)


def create_manhattan_heuristic_to(goal):
    return lambda state: abs(locations[goal][0] - locations[state][0]) + abs(locations[goal][1] - locations[state][1])


class RomaniaSearchProblem(SearchProblem):
    def __init__(self, start, end, stats=NullStats()):
        self._start = start
        self._end = end
        self._stats = stats

    def getStartState(self):
        return self._start

    def isGoalState(self, state):
        return state == self._end

    def getSuccessors(self, state):
        self._stats.expand(state)
        return map[state].iteritems()


class RomaniaMultipleCitiesSearchProblem(SearchProblem):
    def __init__(self, start, cities, stats=NullStats()):
        self._start = start
        self._cities = cities
        self._stats = stats

    def getStartState(self):
        return self._start, ()

    def isGoalState(self, state):
        return len(state[1]) == len(self._cities) and all((city in self._cities for city in state[1]))

    def getSuccessors(self, state):
        self._stats.expand(state[0])
        return [((city, (state[1] + (city,) if (city in self._cities and city not in state[1]) else state[1])), cost) for city, cost in map[state[0]].iteritems()]

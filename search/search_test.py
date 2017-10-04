import unittest

from stats import Stats
from bfs import bfs_search
from dfs import dfs_search, dfs_search_recursive
from dls import dls_search
from ids import ids_search
from ucs import ucs_search
from astar import astar_search
from ida import ida_search
from romania_search_problem import RomaniaSearchProblem, RomaniaMultipleCitiesSearchProblem, create_heuristic_to, create_manhattan_heuristic_to


class TestSearch(unittest.TestCase):
    def test_breadth_first_search(self):
        stats = Stats()
        problem = RomaniaSearchProblem('Arad', 'Bucharest', stats)

        actions = bfs_search(problem)

        self.assertEqual(actions, ['Arad', 'Sibiu', 'Fagaras', 'Bucharest'])
        self.assertEqual(stats.expanded, 11)

    def test_depth_first_search(self):
        stats = Stats()
        problem = RomaniaSearchProblem('Arad', 'Craiova', stats)

        actions = dfs_search(problem)

        self.assertEqual(actions, ['Arad', 'Timisoara', 'Lugoj', 'Mehadia', 'Drobeta', 'Craiova'])
        self.assertEqual(stats.expanded, 5)

    def test_depth_first_search_recursive(self):
        stats = Stats()
        problem = RomaniaSearchProblem('Arad', 'Craiova', stats)

        actions = dfs_search_recursive(problem)

        self.assertEqual(actions, ['Arad', 'Timisoara', 'Lugoj', 'Mehadia', 'Drobeta', 'Craiova'])
        self.assertEqual(stats.expanded, 5)


    def test_depth_limited_search1(self):
        stats = Stats()
        problem = RomaniaSearchProblem('Arad', 'Craiova', stats)

        actions = dls_search(problem, 2)

        self.assertEqual(actions, None)
        self.assertEqual(stats.expanded, 4)

    def test_depth_limited_search2(self):
        stats = Stats()
        problem = RomaniaSearchProblem('Arad', 'Craiova', stats)

        actions = dls_search(problem, 100)

        self.assertEqual(actions, ['Arad', 'Timisoara', 'Lugoj', 'Mehadia', 'Drobeta', 'Craiova'])
        self.assertEqual(stats.expanded, 5)

    def test_iterative_deepening_search1(self):
        stats = Stats()
        problem = RomaniaSearchProblem('Arad', 'Arad', stats)

        actions = ids_search(problem)

        self.assertEqual(actions, ['Arad'])
        self.assertEqual(stats.expanded, 0)
        self.assertEqual(stats.expanded_states, [])

    def test_iterative_deepening_search2(self):
        stats = Stats()
        problem = RomaniaSearchProblem('Arad', 'Timisoara', stats)

        actions = ids_search(problem)

        self.assertEqual(actions, ['Arad', 'Timisoara'])
        self.assertEqual(stats.expanded_states, ['Arad'])
        self.assertEqual(stats.expanded, 1)

    def test_iterative_deepening_search3(self):
        stats = Stats()
        problem = RomaniaSearchProblem('Arad', 'Lugoj', stats)

        actions = ids_search(problem)

        self.assertEqual(actions, ['Arad', 'Timisoara', 'Lugoj'])
        self.assertEqual(stats.expanded_states, ['Arad', 'Arad', 'Timisoara'])
        self.assertEqual(stats.expanded, 3)

    def test_iterative_deepening_search4(self):
        stats = Stats()
        problem = RomaniaSearchProblem('Arad', 'Bucharest', stats)

        actions = ids_search(problem)

        self.assertEqual(actions, ['Arad', 'Sibiu', 'Fagaras', 'Bucharest'])
        self.assertEqual(stats.expanded_states, ['Arad', 'Arad', 'Timisoara', 'Zerind', 'Sibiu', 'Arad', 'Timisoara', 'Lugoj', 'Zerind', 'Oradea', 'Sibiu', 'Rimnicu', 'Fagaras'])
        self.assertEqual(stats.expanded, 13)

    def test_iterative_deepening_search5(self):
        stats = Stats()
        problem = RomaniaSearchProblem('Arad', 'Craiova1', stats)

        actions = ids_search(problem)

        self.assertEqual(actions, None)
        self.assertEqual(stats.expanded, 146)

    def test_uniform_cost_search(self):
        stats = Stats()
        problem = RomaniaSearchProblem('Arad', 'Bucharest', stats)

        actions = ucs_search(problem)

        self.assertEqual(actions, ['Arad', 'Sibiu', 'Rimnicu', 'Pitesti', 'Bucharest'])
        self.assertEqual(stats.expanded_states, ['Arad', 'Zerind', 'Timisoara', 'Sibiu', 'Oradea', 'Rimnicu', 'Lugoj', 'Fagaras', 'Mehadia', 'Pitesti', 'Craiova', 'Drobeta'])
        self.assertEqual(stats.expanded, 12)

    def test_astar_search(self):
        stats = Stats()
        problem = RomaniaSearchProblem('Arad', 'Bucharest', stats)

        actions = astar_search(problem, create_heuristic_to('Bucharest'))

        self.assertEqual(actions, ['Arad', 'Sibiu', 'Rimnicu', 'Pitesti', 'Bucharest'])
        self.assertEqual(stats.expanded_states, ['Arad', 'Sibiu', 'Fagaras', 'Rimnicu', 'Pitesti'])
        self.assertEqual(stats.expanded, 5)

    def test_ida_search(self):
        stats = Stats()
        problem = RomaniaSearchProblem('Sibiu', 'Bucharest', stats)

        actions = ida_search(problem, create_heuristic_to('Bucharest'))

        self.assertEqual(actions, ['Sibiu', 'Rimnicu', 'Pitesti', 'Bucharest'])
        self.assertEqual(stats.expanded_states, ['Sibiu', 'Sibiu', 'Fagaras', 'Sibiu', 'Rimnicu', 'Fagaras', 'Sibiu', 'Rimnicu', 'Pitesti', 'Fagaras', 'Sibiu', 'Rimnicu', 'Pitesti'])
        self.assertEqual(stats.expanded, 13)

    def test_ida_search(self):
        stats = Stats()
        problem = RomaniaSearchProblem('Arad', 'Bucharest', stats)

        actions = ida_search(problem, create_heuristic_to('Bucharest'))

        self.assertEqual(actions, ['Arad', 'Sibiu', 'Rimnicu', 'Pitesti', 'Bucharest'])

    def test_ucs_multiple_cities1(self):
        stats = Stats()
        problem = RomaniaMultipleCitiesSearchProblem('Arad', ['Rimnicu', 'Fagaras'], stats)

        actions = ucs_search(problem)

        self.assertEqual(map(lambda a: a[0], actions), ['Arad', 'Sibiu', 'Rimnicu', 'Sibiu', 'Fagaras'])

    def test_ucs_multiple_cities2(self):
        stats = Stats()
        problem = RomaniaMultipleCitiesSearchProblem('Arad', ['Sibiu', 'Oradea'], stats)

        actions = ucs_search(problem)

        self.assertEqual(map(lambda a: a[0], actions), ['Arad', 'Sibiu', 'Oradea'])


if __name__ == '__main__':
    unittest.main()
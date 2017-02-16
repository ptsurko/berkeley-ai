import unittest
from stats import Stats
from dfs import Dfs
from board import Board


class TestDfs(unittest.TestCase):
  def setUp(self):
    self.stats = Stats()

  def test_1(self):
    board = Board([0, 8, 7, 6, 5, 4, 3, 2, 1])
    dfs = Dfs(board, self.stats)
    dfs.search()
    self.assertEqual(self.stats.cost_of_path, 41910)

  def test_2(self):
    board = Board([3, 1, 2, 0, 4, 5, 6, 7, 8])
    dfs = Dfs(board, self.stats)
    dfs.search()
    self.assertEqual(self.stats.cost_of_path, 1)

  def test_3(self):
    board = Board([1, 2, 5, 3, 4, 0, 6, 7, 8])
    dfs = Dfs(board, self.stats)
    dfs.search()
    self.assertEqual(self.stats.cost_of_path, 3)

  def test_4(self):
    board = Board([1, 7, 8, 2, 3, 4, 5, 6, 0])
    dfs = Dfs(board, self.stats)
    dfs.search()
    self.assertEqual(self.stats.cost_of_path, 10774)

  def test_5(self):
    board = Board([8, 6, 4, 2, 1, 3, 5, 7, 0])
    dfs = Dfs(board, self.stats)
    dfs.search()
    self.assertEqual(self.stats.cost_of_path, 9612)


if __name__ == '__main__':
    unittest.main()
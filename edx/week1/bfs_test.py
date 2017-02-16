import unittest
from stats import Stats
from bfs import Bfs
from board import Board


class TestBfs(unittest.TestCase):
  def setUp(self):
    self.stats = Stats()

  def test_1(self):
    board = Board([0, 8, 7, 6, 5, 4, 3, 2, 1])
    bfs = Bfs(board, self.stats)
    bfs.search()
    self.assertEqual(self.stats.cost_of_path, 30)

  def test_2(self):
    board = Board([3, 1, 2, 0, 4, 5, 6, 7, 8])
    bfs = Bfs(board, self.stats)
    bfs.search()
    self.assertEqual(self.stats.cost_of_path, 1)

  def test_3(self):
    board = Board([1, 2, 5, 3, 4, 0, 6, 7, 8])
    bfs = Bfs(board, self.stats)
    bfs.search()
    self.assertEqual(self.stats.cost_of_path, 3)

  def test_4(self):
    board = Board([1, 7, 8, 2, 3, 4, 5, 6, 0])
    bfs = Bfs(board, self.stats)
    bfs.search()
    self.assertEqual(self.stats.cost_of_path, 24)

  def test_5(self):
    board = Board([4, 1, 2, 3, 5, 6, 10, 7, 8, 9, 0, 11, 12, 13, 14, 15])
    bfs = Bfs(board, self.stats)
    bfs.search()
    self.assertEqual(self.stats.cost_of_path, 4)


if __name__ == '__main__':
    unittest.main()
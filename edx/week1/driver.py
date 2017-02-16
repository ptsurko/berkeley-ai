import sys
import string
from board import Board
from stats import Stats
from bfs import Bfs
from dfs import Dfs

# stats = Stats()
# initial_board = Board([1, 2, 5, 3, 4, 0, 6, 7, 8])
# initial_board = Board([0, 1, 2, 3, 4, 5, 6, 7, 8])
# initial_board = Board([0, 8, 7, 6, 5, 4, 3, 2, 1])
# # initial_board = Board([3, 1, 2, 0, 4, 5, 6, 7, 8])
# # initial_board = Board([1, 2, 5, 3, 4, 0, 6, 7, 8])
# # initial_board = Board([1, 7, 8, 2, 3, 4, 5, 6, 0])
# # initial_board = Board([4,1,2,3,5,6,10,7,8,9,0,11,12,13,14,15])
#
# bfs = Bfs(initial_board, stats)
# bfs.search()
#
# # dfs = Dfs(initial_board, stats)
# # dfs.search()
#
# print stats


def main(argv):
  stats = Stats()
  search_alg = argv[0]
  board = Board([int(i) for i in string.split(argv[1], ',')])

  if search_alg == 'bfs':
    search_strategy = Bfs(board, stats)
  elif search_alg == 'dfs':
    search_strategy = Dfs(board, stats)

  search_strategy.search()
  f = open('output.txt', 'w')
  f.write(str(stats))
  f.close()

if __name__ == "__main__":
  main(sys.argv[1:])

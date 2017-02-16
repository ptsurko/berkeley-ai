import math


class Board:
  __slots__ = ('_board', '_board_size', '_blank_index')

  def __init__(self, board, board_size=None, blank_index=None):
    self._board = board
    self._board_size = board_size or (int)(math.sqrt(len(board)))
    self._blank_index = blank_index or board.index(0)

  @property
  def board(self):
    return self._board

  def __eq__(self, other):
    return self._board == other.board

  def can_move_up(self):
    return self._blank_index >= self._board_size

  def can_move_down(self):
    return self._blank_index < len(self._board) - self._board_size

  def can_move_right(self):
    return (self._blank_index + 1) % self._board_size != 0

  def can_move_left(self):
    return self._blank_index % self._board_size != 0

  def _move(self, new_blank):
    new_board = self._board[:]
    new_board[self._blank_index], new_board[new_blank] = new_board[new_blank], new_board[self._blank_index]
    return Board(new_board, board_size=self._board_size, blank_index=new_blank)

  def move_up(self):
    return self._move(self._blank_index - self._board_size)

  def move_down(self):
    return self._move(self._blank_index + self._board_size)

  def move_left(self):
    return self._move(self._blank_index - 1)

  def move_right(self):
    return self._move(self._blank_index + 1)

  def is_solved(self):
    for i in xrange(1, len(self._board)):
      if self._board[i - 1] > self._board[i]:
        return False
    return True

  def __str__(self):
    return str(self._board)

  def __hash__(self):
    return hash(tuple(self._board))

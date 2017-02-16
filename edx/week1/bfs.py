from move import Move
from collections import namedtuple

_Step = namedtuple('Step', ['parent_step', 'move', 'board', 'depth'])


class Bfs(object):
  def __init__(self, initial_board, stats):
    self._stats = stats
    self._initial_board = initial_board

  @classmethod
  def _get_step_path(cls, step):
    steps = []
    while step.parent_step is not None:
      steps.append(step.move)
      step = step.parent_step

    steps.reverse()
    return steps

  def search(self):
    steps = [_Step(None, Move.NONE, self._initial_board, 0)]
    visited_boards = set(self._initial_board.board)
    step_index = 0
    while True:
      if step_index == len(steps):
        print 'NO SOLUTION'
        break

      step = steps[step_index]
      board = step.board

      if board.is_solved():
        self._stats.set_search_depth(step.depth)
        self._stats.set_max_search_depth(step.depth)
        self._stats.set_fringe_size(len(steps))
        self._stats.end()
        self._stats.set_path_to_goal(self._get_step_path(step))
        break

      board = step.board

      if board.can_move_up():
        up_board = board.move_up()
        if up_board not in visited_boards:
          steps.append(_Step(step, Move.UP, up_board, step.depth + 1))
          self._stats.set_max_search_depth(step.depth + 1)
          visited_boards.add(up_board)

      if board.can_move_down():
        down_board = board.move_down()
        if down_board not in visited_boards:
          steps.append(_Step(step, Move.DOWN, down_board, step.depth + 1))
          self._stats.set_max_search_depth(step.depth + 1)
          visited_boards.add(down_board)

      if board.can_move_left():
        left_board = board.move_left()
        if left_board not in visited_boards:
          steps.append(_Step(step, Move.LEFT, left_board, step.depth + 1))
          self._stats.set_max_search_depth(step.depth + 1)
          visited_boards.add(left_board)

      if board.can_move_right():
        right_board = board.move_right()
        if right_board not in visited_boards:
          steps.append(_Step(step, Move.RIGHT, right_board, step.depth + 1))
          self._stats.set_max_search_depth(step.depth + 1)
          visited_boards.add(right_board)

      self._stats.set_search_depth(step.depth)
      self._stats.set_max_search_depth(step.depth)
      self._stats.set_fringe_size(len(steps))
      self._stats.expand_node()
      step_index += 1
      # visited_boards.add(board)
      # print stats
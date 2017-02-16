from board import Board


assert Board([1, 2, 5, 3, 4, 0, 6, 7, 8]).can_move_up()
assert Board([1, 2, 5, 3, 4, 8, 6, 7, 0]).can_move_up()
assert Board([1, 2, 5, 3, 4, 8, 0, 7, 6]).can_move_up()
assert not Board([1, 2, 0, 3, 4, 5, 6, 7, 8]).can_move_up()
assert not Board([0, 2, 1, 3, 4, 5, 6, 7, 8]).can_move_up()


assert Board([1, 2, 5, 3, 4, 0, 6, 7, 8]).can_move_down()
assert Board([0, 2, 5, 3, 4, 1, 6, 7, 8]).can_move_down()
assert Board([1, 2, 0, 3, 4, 5, 6, 7, 8]).can_move_down()
assert not Board([1, 2, 6, 3, 4, 5, 0, 7, 8]).can_move_down()
assert not Board([1, 2, 6, 3, 4, 5, 8, 7, 0]).can_move_down()

assert Board([1, 0, 5, 3, 4, 2, 6, 7, 8]).can_move_left()
assert Board([1, 2, 0, 3, 4, 8, 6, 7, 3]).can_move_left()
assert Board([1, 2, 5, 3, 0, 8, 4, 7, 6]).can_move_left()
assert Board([1, 2, 5, 3, 4, 6, 8, 7, 0]).can_move_left()
assert not Board([1, 2, 3, 0, 4, 5, 6, 7, 8]).can_move_left()
assert not Board([0, 2, 1, 3, 4, 5, 6, 7, 8]).can_move_left()

assert Board([1, 0, 5, 3, 4, 2, 6, 7, 8]).can_move_right()
assert Board([0, 2, 1, 3, 4, 8, 6, 7, 3]).can_move_right()
assert Board([1, 2, 5, 3, 0, 8, 4, 7, 6]).can_move_right()
assert Board([1, 2, 5, 3, 4, 6, 8, 0, 7]).can_move_right()
assert not Board([1, 2, 3, 8, 4, 5, 6, 7, 0]).can_move_right()
assert not Board([1, 2, 0, 3, 4, 5, 6, 7, 8]).can_move_right()


assert Board([1, 2, 5, 3, 4, 0, 6, 7, 8]) == Board([1, 2, 5, 3, 4, 0, 6, 7, 8])
assert not (Board([1, 2, 5, 3, 4, 6, 0, 7, 8]) == Board([1, 2, 5, 3, 4, 0, 6, 7, 8]))

assert Board([1, 2, 5, 3, 4, 0, 6, 7, 8]).move_up() == Board([1, 2, 0, 3, 4, 5, 6, 7, 8])
assert Board([1, 2, 5, 3, 4, 0, 6, 7, 8]).move_down() == Board([1, 2, 5, 3, 4, 8, 6, 7, 0])
assert Board([1, 2, 5, 3, 0, 4, 6, 7, 8]).move_left() == Board([1, 2, 5, 0, 3, 4, 6, 7, 8])
assert Board([1, 2, 5, 3, 0, 4, 6, 7, 8]).move_right() == Board([1, 2, 5, 3, 4, 0, 6, 7, 8])

assert Board([1, 2, 0, 3, 4, 5, 6, 7, 8]).is_solved() is False
assert Board([0, 1, 2, 3, 4, 5, 6, 7, 8]).is_solved() is True

from search_problem import SearchProblem


def attacking_cost(board):
    attacking = 0

    for i in xrange(len(board)):
        for j in xrange(i + 1, len(board)):
            if board[i] == board[j] or (abs(i - j) == abs(board[i] - board[j])):
                attacking += 1

    return attacking


class ChessProblem(SearchProblem):
    def __init__(self, board):
        self._board = board

    def getStartState(self):
        return self._board

    def isGoalState(self, state):
        for i in xrange(len(state)):
            for j in xrange(i + 1, len(state)):
                if state[i] == state[j] or (abs(i - j) == abs(state[i] - state[j])):
                    return False

        return True

    def getSuccessors(self, state):
        for i in xrange(len(state)):
            for j in xrange(len(state)):
                if j != state[i]:
                    new_state = state[:]
                    new_state[i] = j
                    yield new_state


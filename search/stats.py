

class Stats:
    def __init__(self):
        self.expanded = 0
        self.expanded_states = []

    def expand(self, state):
        self.expanded += 1
        self.expanded_states.append(state)


class NullStats(Stats):
    def expand(self):
        pass

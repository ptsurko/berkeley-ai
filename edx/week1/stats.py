import time
import resource


class Stats(object):
  def __init__(self):
    self._path_to_goal = []
    self._nodes_expanded = 0
    self._fringe_size = 0
    self._max_fringe_size = 0
    self._search_depth = 0
    self._max_search_depth = 0
    self._start_time = time.time()
    self._end_time = None

  @property
  def path_to_goal(self):
    return self._path_to_goal

  @property
  def cost_of_path(self):
    return len(self._path_to_goal)

  @property
  def nodes_expanded(self):
    return self._nodes_expanded

  @property
  def fringe_size(self):
    return self._fringe_size

  @property
  def max_fringe_size(self):
    return self._max_fringe_size

  @property
  def search_depth(self):
    return self._search_depth

  @property
  def max_search_depth(self):
    return self._max_search_depth

  def end(self):
    self._end_time = time.time()

  def set_path_to_goal(self, path_to_goal):
    self._path_to_goal = path_to_goal

  def set_fringe_size(self, fringe_size):
    self._fringe_size = fringe_size
    if fringe_size > self._max_fringe_size:
      self._max_fringe_size = fringe_size

  def set_search_depth(self, search_depth):
    self._search_depth = search_depth
    if search_depth > self._max_search_depth:
      self._max_search_depth = search_depth

  def set_max_search_depth(self, max_search_depth):
    if self._max_search_depth < max_search_depth:
      self._max_search_depth = max_search_depth

  def expand_node(self):
    self._nodes_expanded += 1

  def __str__(self):
    return """path_to_goal: %s
cost_of_path: %d
nodes_expanded: %d
fringe_size: %d
max_fringe_size: %d
search_depth: %d
max_search_depth: %d
running_time: %f
max_ram_usage: %f
""" % (self.path_to_goal,
       self.cost_of_path,
       self.nodes_expanded,
       self.fringe_size,
       self.max_fringe_size,
       self.search_depth,
       self.max_search_depth,
       ((self._end_time or time.time()) - self._start_time),
       resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

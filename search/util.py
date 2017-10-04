from Queue import Queue as pQueue
import heapq

def Stack():
    return []


def Queue():
    return pQueue()


class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self.heap = []

    def push(self, priority, item):
        entry = (priority, item)
        heapq.heappush(self.heap, entry)

    def pop(self):
        _ , val = heapq.heappop(self.heap)
        return val

    def empty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)
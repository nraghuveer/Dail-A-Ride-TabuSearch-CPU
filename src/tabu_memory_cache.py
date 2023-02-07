# DataStructure to hold the tabu memory
# supports eviction using iteration count
# Operations -> put, expire, exists, incIteration

from collections import namedtuple, defaultdict
from typing import List, Dict, Set

MoveParams = namedtuple('MoveParams', "i k1 k2 p1 p2")

class TabuMemoryCache:
    def __init__(self, evictIterations: int) -> None:
        """_summary_

        :param evictIterations: iterations after which a item should be evicted
        :type evictIterations: int
        """
        self._cur_iterations = 0
        self._evict_iters = evictIterations
        self._slots: Dict[int, Set[MoveParams]] = defaultdict(set)

    def put(self, key: MoveParams):
        self._slots[self._cur_iterations].add(key)

    def exists(self, key: MoveParams) -> bool:
        if self._evict_iters <= 0:
            return False
        for x in self._slots:
            if key in self._slots[x]:
                return True
        return False

    def _evict(self):
        if self._evict_iters <= 0 or self._cur_iterations <= self._evict_iters:
            return
        self._slots.pop(self._cur_iterations - self._evict_iters)

    def inc_iteration(self) -> None:
        self._cur_iterations += 1
        self._evict()

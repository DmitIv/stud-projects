from pyFiles.robotReverse import RobotReverse, Robot2D, sys
from heapq import heappop, heappush


class Robot3D(RobotReverse):
    def __init__(self, start: int, end: int):
        RobotReverse.__init__(self, end, start)
        self._priority_queue: list = list()
        self._scores: dict = dict()

    def scores(self):
        return self._scores

    @staticmethod
    def _neighbors(current: int) -> list:
        return [current * 2, current + 3, current - 2]

    def _heuristic(self, current: int, previous) -> int:
        if current + 2 == previous:
            additional = self.end() // 2
        else:
            additional = 0
        result = 1
        return result

    def _pretty_output(self, finish: int):
        current = finish
        previous = None
        result: list = list()
        while current != previous:
            result.append(current)
            previous = current
            current = self.parents()[current]
        result.reverse()
        return result

    def _a_star_searching(self):
        self._priority_queue.append((0, self.start(), 0, self.start()))

        while len(self._priority_queue) > 0:
            priority, current, current_distance, previous = heappop(self._priority_queue)

            if current == self.end():
                self.parents()[current] = previous
                solution: list = self._pretty_output(current)
                result: str = "Steps number: {};\n".format(len(solution) - 1)
                result += "All values: {}.".format(solution)
                return result

            if self.parents().get(current, None) is not None:
                continue
            self.parents()[current] = previous

            current_distance += 1
            for neighbor in self._neighbors(current):
                if self.parents().get(neighbor, None) is not None:
                    continue

                if self.scores().get(neighbor, None) is not None:
                    neighbor_distance_from, neighbor_distance_to = self.scores()[neighbor]
                    if neighbor_distance_from <= current_distance:
                        continue
                    current_distance_to = neighbor_distance_to
                else:
                    current_distance_to = self._heuristic(neighbor, current)

                self.scores()[neighbor] = (current_distance, current_distance_to)
                heappush(self._priority_queue, (current_distance + current_distance_to, neighbor, current_distance,
                                                current))

        return "Oh, dead end!"

    def run(self):
        return self._a_star_searching()


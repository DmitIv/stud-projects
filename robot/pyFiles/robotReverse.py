from pyFiles.robot2d import Robot2D
import sys


class RobotReverse:
    def __init__(self, start: int, end: int):
        self._start: int = end
        self._end: int = start

        self._parents: dict = dict()

        self.ITER_LIMIT = self._end * 2

    def start(self):
        return self._start

    def end(self):
        return self._end

    def parents(self):
        return self._parents

    def _heuristic(self, current: int, previous: int) -> int:
        return current - self.end() if self.end() <= current else sys.maxsize - 1

    @staticmethod
    def _neighbors(current: int) -> list:
        result = []
        if not current & 1:
            result.append(current // 2)
        result.append(current - 3)
        return result

    def _pretty_output(self, finish: int):
        if finish is None:
            return []
        else:
            if finish == self.parents().get(finish, None):
                result = [finish]
            else:
                result = self._pretty_output(self.parents().get(finish, None)) + [finish]

            return result

    def _ida_star_recurrent_search(self, current: int, current_distance: int, threshold: int,
                                   previous: int) -> (bool, int, int):
        full_distance = self._heuristic(current, previous) + current_distance
        if full_distance > threshold:
            return False, full_distance, current

        if current == self.end():
            return True, full_distance, current

        minimum: int = sys.maxsize
        for neighbor in self._neighbors(current):
            if self.parents().get(neighbor, None) is None:
                self.parents()[neighbor] = current
                if neighbor >= self.end():
                    flag, bound, finish = self._ida_star_recurrent_search(neighbor, current_distance + 3,
                                                                          threshold, current)
                    if flag:
                        return flag, bound, finish
                    if bound < minimum:
                        minimum = bound

                    self.parents()[neighbor] = None

        return False, minimum, current

    def _ida_star_searching(self) -> str:
        threshold: int = self._heuristic(self.start(), self.start())
        self.parents()[self.start()] = self.start()

        counter: int = 0
        while counter < self.ITER_LIMIT:
            flag, bound, finish = self._ida_star_recurrent_search(self.start(), 0, threshold, self.start())
            if flag:
                solution: list = self._pretty_output(finish)
                result: str = "Steps number: {};\n".format(len(solution) - 1)
                result += "All values: {}.".format(solution)
                return result
            elif bound == sys.maxsize:
                return "Oh, dead end!"
            else:
                threshold = bound
            counter += 1
        return "Oh, dead end!"

    def run(self):
        return self._ida_star_searching()



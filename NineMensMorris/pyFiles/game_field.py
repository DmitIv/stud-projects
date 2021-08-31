from pyFiles.field_description import FieldDescription
from functools import reduce


class GamesField:
    def __init__(self):
        self._current_state: dict = FieldDescription.get_description()
        self._players_points_limit: list = [9, 9]
        self._players_points: list = [
            [],
            []
        ]
        self._players_mills_count: list = [0, 0]

        self._points_count: int = 24
        self._free_points_count: int = self._points_count

        self._mill_was: dict = dict()
        self._blocked: list = [0, 0, 0, 0]

    def move(self, player: int, old_point: tuple, new_point: tuple) -> bool:

        for reachable in self._current_state[old_point]["reachable"]:
            if len(reachable) > 0:
                self._current_state[reachable]["weight"] -= 1
                if self._current_state[reachable]["weight"] + 1 == self._current_state[reachable]["limit"]:
                    owner = self._current_state[reachable]["owner"]
                    self._blocked[owner] -= 1

        index: int = self._players_points[player].index(old_point)
        self._players_points[player][index] = new_point

        self._current_state[old_point]["owner"] = 0
        self._current_state[new_point]["owner"] = player + 1

        for reachable in self._current_state[new_point]["reachable"]:
            if len(reachable) > 0:
                self._current_state[reachable]["weight"] += 1
                if self._current_state[reachable]["weight"] == self._current_state[reachable]["limit"]:
                    owner = self._current_state[reachable]["owner"]
                    self._blocked[owner] += 1

        return True

    def delete_chip(self, player: int, point: tuple) -> bool:
        # print(player + 1)
        # print(self._current_state[point]["owner"])
        if self._current_state[point]["owner"] == player + 1:
            self._players_points[player].remove(point)
            self._current_state[point]["owner"] = 0
            self._free_points_count += 1

            if self._current_state[point]["weight"] == self._current_state[point]["limit"]:
                self._blocked[player + 1] -= 1

            for reachable in self._current_state[point]["reachable"]:
                if len(reachable) > 0:
                    self._current_state[reachable]["weight"] -= 1
                    if self._current_state[reachable]["weight"] + 1 == self._current_state[reachable]["limit"]:
                        owner = self._current_state[reachable]["owner"]
                        self._blocked[owner] -= 1

            return True
        else:
            return False

    def put_chip(self, player: int, point: tuple) -> bool:
        self._players_points[player].append(point)
        self._current_state[point]["owner"] = player + 1
        self._players_points_limit[player] -= 1
        self._free_points_count -= 1

        if self._current_state[point]["weight"] == self._current_state[point]["limit"]:
            self._blocked[player + 1] += 1

        for reachable in self._current_state[point]["reachable"]:
            if len(reachable) > 0:
                self._current_state[reachable]["weight"] += 1
                if self._current_state[reachable]["weight"] == self._current_state[reachable]["limit"]:
                    owner = self._current_state[reachable]["owner"]
                    self._blocked[owner] += 1

        return True

    def reachable_points(self, point: tuple = (0, 0), current_owner: int = 0) -> list:
        reachable_points: list = list()
        l: int = len(self._current_state[point]["reachable"])

        if l > 4:
            for observed_point in self._current_state[point]["reachable"]:
                if self._current_state[observed_point]["owner"] == current_owner:
                    reachable_points.append(observed_point)

        else:
            for i in range(l):
                observed_point = self._current_state[point]["reachable"][i]
                owner = self._current_state.get(observed_point, {"owner": -1})["owner"]

                if owner == current_owner:
                    reachable_points.append(observed_point)
                    candidate = self._current_state[observed_point]["reachable"][i]

                    if self._current_state.get(candidate, {"owner": -1})["owner"] == current_owner:
                        reachable_points.append(candidate)

        return reachable_points

    def player_can_put(self, player: int) -> bool:
        return self._players_points_limit[player] > 0

    def player_defeat(self, player) -> bool:
        return (self._players_points_limit[player] == 0) and (len(self._players_points[player]) < 3) or (
                    self._blocked[player + 1] == len(self._players_points[player]))

    def mill_made(self, point: tuple) -> int:
        directions = [[0, 2], [1, 3]]
        target_owner: int = self._current_state[point]["owner"]

        mill_counter = 0
        for direction in directions:
            line = []

            first = self._current_state[point]["reachable"][direction[0]]
            if self._current_state.get(first, {"owner": -1})["owner"] == target_owner:
                second = self._current_state[first]["reachable"][direction[0]]
                if self._current_state.get(second, {"owner": -1})["owner"] == target_owner:
                    line.append(second)
                line.append(first)

            line.append(point)

            first = self._current_state[point]["reachable"][direction[1]]
            if self._current_state.get(first, {"owner": -1})["owner"] == target_owner:
                line.append(first)
                second = self._current_state[first]["reachable"][direction[1]]
                if self._current_state.get(second, {"owner": -1})["owner"] == target_owner:
                    line.append(second)

            if len(line) == 3:
                line_t = tuple(line)
                result = self._mill_was.get(line_t, [False, False])
                if not result[target_owner - 1]:
                    result[target_owner - 1] = True
                    self._mill_was[line_t] = result
                    mill_counter += 1
        print(mill_counter)
        return mill_counter

    def player_score(self, player) -> int:
        return (self._players_mills_count[player] / 10. + self.player_free_chips() + self.player_current_points_count()) * (
                    self.free_space()[0] / self.free_space()[1])

    def player_make_mill(self, player) -> None:
        self._players_mills_count[player] += 1

    def player_current_points_count(self, player) -> int:
        return len(self._players_points[player])

    def free_space(self) -> tuple:
        return self._points_count, self._free_points_count

    def player_free_chips(self, player: int) -> int:
        return self._players_points_limit[player]

    def point_free(self, point, target_owner: int = 0) -> bool:
        return self._current_state[point]["owner"] == target_owner

    def blocked_chips(self, player):
        return self._blocked[player + 1]




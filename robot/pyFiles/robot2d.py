class Robot2D:
    def __init__(self, start: int, end: int):
        self._start: int = start
        self._end: int = end

        self._current_position: tuple = (1, 0)
        self._shift_x: int = 0
        self._shift_y: int = 0

        self._result_expression: str = ""
        self._result_values_sequence: list = list()

        self._solutions: list = list()

    def _big_step_forward(self) -> (int, int):
        self._shift_x += 1
        return self._current_position[0] * 2, self._current_position[1]

    def _big_step_backward(self) -> (int, int):
        self._shift_x -= 1
        return int(self._current_position[0] / 2), self._current_position[1]

    def _small_step_forward(self) -> (int, int):
        self._shift_y += 1
        return self._current_position[0], self._current_position[1] + 3

    def _small_step_backward(self) -> (int, int):
        self._shift_y -= 1
        return self._current_position[0], self._current_position[1] - 3

    def _calculate_current_position_value(self) -> int:
        return self._current_position[0] * self._start + self._current_position[1]

    def _within_field(self) -> bool:
        return self._calculate_current_position_value() < self._end

    def _searching_in_progress(self) -> bool:
        return self._shift_x >= 0

    def _found(self) -> bool:
        return self._calculate_current_position_value() == self._end

    def _solution_optimization(self) -> None:
        self._result_values_sequence.append(self._end)
        self._result_expression = "{}"
        first_axis: int = self._current_position[0]
        second_axis: int = self._current_position[1]

        while first_axis >= 2 or second_axis > 0:
            if first_axis == 1:
                self._result_expression = self._result_expression.format("{} + 3")
                second_axis -= 3
            else:
                if second_axis >= 3:
                    if second_axis & 1:
                        second_axis -= 3
                        self._result_expression = self._result_expression.format("{} + 3")

                    else:
                        second_axis //= 2
                        first_axis //= 2
                        self._shift_y -= second_axis // 3
                        self._result_expression = self._result_expression.format("2*({})")

                else:
                    first_axis //= 2
                    self._result_expression = self._result_expression.format("2*{}")

            self._current_position = (first_axis, second_axis)
            self._result_values_sequence.append(self._calculate_current_position_value())

        self._result_expression = self._result_expression.format(str(self._start))

    def _success_solution(self) -> str:
        self._solution_optimization()
        return "Steps number: {0};\nResult expression: {1};\nAll values: {2}".format(self._shift_y + self._shift_x,
                                                                                     self._result_expression,
                                                                                     str(
                                                                                         self._result_values_sequence[::-1]))

    @staticmethod
    def _failed() -> str:
        return "Solution doesn't exists"

    def run(self) -> str:
        while self._within_field():
            self._current_position = self._big_step_forward()

        if self._found():
            self._solutions.append((self._shift_y, self._shift_x, self._current_position))
        else:
            self._current_position = self._big_step_backward()

        need_left_shift: bool = False
        while self._searching_in_progress():
            if need_left_shift:
                self._current_position = self._big_step_backward()
                need_left_shift = False

            else:
                if self._within_field():
                    self._current_position = self._small_step_forward()
                else:
                    if self._found():
                        self._solutions.append((self._shift_y, self._shift_x, self._current_position))

                    need_left_shift = True
                    self._current_position = self._small_step_backward()

        if len(self._solutions) > 0:
            min_solution: tuple = min(self._solutions, key=lambda x: x[0] + x[1])
            self._shift_y, self._shift_x, self._current_position = min_solution
            return self._success_solution()

        return self._failed()


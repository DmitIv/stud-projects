import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QLineEdit, QTextBrowser
from PyQt5.QtGui import QIcon, QPainter, QBrush, QPen
from PyQt5.QtCore import pyqtSlot
from pyFiles.game_field import GamesField
from pyFiles.field_description import FieldDescription
from PyQt5.QtCore import Qt
import random

players_color = ['background-color:#0000FF;color:#000000;', 'background-color:#FF0000;color:#000000;']
blocked_players_color = ['background-color:#0000FF;color:#FFFFFF;', 'background-color:#FF0000;color:#FFFFFF;']
opened_color = 'background-color:#FFFFFF;color:#000000;'
blocked_color = 'background-color:#000000;color:#000000;'


class Position(QPushButton):
    def __init__(self, position: tuple, scale: int, parent=None):
        QPushButton.__init__(self, parent)
        self._parent = parent
        self._position = position
        self._color = blocked_color
        self._blocked = True
        QPushButton.setText(self, "{0} - {1}".format(position[0], position[1]))
        QPushButton.move(self, 1 * scale + position[1] * scale, position[0] * scale)
        QPushButton.setStyleSheet(self, self._color)

    def blocked(self):
        if self._color == opened_color:
            self._color = blocked_color
            self.setStyleSheet(self._color)
        self._blocked = True

    def opened(self):
        self._color = opened_color
        self.setStyleSheet(self._color)
        self._blocked = False

    def clicking(self) -> None:
        if not self._blocked and self._parent._end_of_moving and self._parent._mill_counter == 0:
            self._color = players_color[self._parent._current_player]
            self.setStyleSheet(self._color)

            self.blocked()

            self._parent._handling_point = self._position
            self._parent.step()

        elif self._blocked and (self._parent._mill_counter > 0 or not self._parent._end_of_moving):
            self._parent._handling_point = self._position
            self._parent.step()


class NineMensMorrisGame(QWidget):
    def __init__(self, scale: int = 100, first_player: int = 1, parent=None):
        random.seed(-1000, 1000)
        QWidget.__init__(self, parent)

        self._game_field = GamesField()
        self._current_player = first_player
        self._second_phase = False
        self._end_of_moving = True
        self._handling_point = ()
        self._previous_point = ()
        self._mill_counter = 0
        self._was_blocked = []

        self._scale = scale
        self._points = dict()
        self.text_browser_1 = QTextBrowser(self)
        self.text_browser_2 = QTextBrowser(self)
        self.text_browser_3 = QTextBrowser(self)
        self.text_browser_4 = QTextBrowser(self)
        self.text_browser_5 = QTextBrowser(self)
        self.text_browser_6 = QTextBrowser(self)
        self.text_browser_7 = QTextBrowser(self)
        self.initUI()

    def paintEvent(self, event):
        lines = QPainter(self)
        lines.setPen(QPen(Qt.black, 5, Qt.SolidLine))
        append = 1 * self._scale + 35
        append_y = 10
        # bigger square
        lines.drawLine(append + 1 * self._scale, 1 * self._scale + append_y, append + 1 * self._scale,
                       7 * self._scale + append_y)
        lines.drawLine(append + 1 * self._scale, 1 * self._scale + append_y, append + 7 * self._scale,
                       1 * self._scale + append_y)
        lines.drawLine(append + 7 * self._scale, 1 * self._scale + append_y, append + 7 * self._scale,
                       7 * self._scale + append_y)
        lines.drawLine(append + 1 * self._scale, 7 * self._scale + append_y, append + 7 * self._scale,
                       7 * self._scale + append_y)
        # middle square
        lines.drawLine(append + 2 * self._scale, 2 * self._scale + append_y, append + 2 * self._scale,
                       6 * self._scale + append_y)
        lines.drawLine(append + 2 * self._scale, 2 * self._scale + append_y, append + 6 * self._scale,
                       2 * self._scale + append_y)
        lines.drawLine(append + 6 * self._scale, 6 * self._scale + append_y, append + 2 * self._scale,
                       6 * self._scale + append_y)
        lines.drawLine(append + 6 * self._scale, 6 * self._scale + append_y, append + 6 * self._scale,
                       2 * self._scale + append_y)
        # smaller square
        lines.drawLine(append + 3 * self._scale, 3 * self._scale + append_y, append + 3 * self._scale,
                       5 * self._scale + append_y)
        lines.drawLine(append + 3 * self._scale, 3 * self._scale + append_y, append + 5 * self._scale,
                       3 * self._scale + append_y)
        lines.drawLine(append + 5 * self._scale, 5 * self._scale + append_y, append + 3 * self._scale,
                        5 * self._scale + append_y)
        lines.drawLine(append + 5 * self._scale, 5 * self._scale + append_y, append + 5 * self._scale,
                       3 * self._scale + append_y)
        # horizontal
        lines.drawLine(append + 1 * self._scale, 4 * self._scale + append_y, append + 3 * self._scale,
                       4 * self._scale + append_y)
        lines.drawLine(append + 5 * self._scale, 4 * self._scale + append_y, append + 7 * self._scale,
                       4 * self._scale + append_y)
        # vertical
        lines.drawLine(append + 4 * self._scale, 1 * self._scale + append_y, append + 4 * self._scale,
                       3 * self._scale + append_y)
        lines.drawLine(append + 4 * self._scale, 5 * self._scale + append_y, append + 4 * self._scale,
                       7 * self._scale + append_y)

    def initUI(self):
        for point in list(FieldDescription.get_description().keys()):
            if point != (0, 0):
                button = Position(point, self._scale, self)
                button.clicked.connect(button.clicking)
                self._points[point] = button

        self.text_browser_1.move(10, 10)
        self.text_browser_1.setText(
            "Фишки для размещения игрока blue: {}".format(self._game_field.player_free_chips(0)))
        self.text_browser_1.setFixedSize(140, 50)
        self.text_browser_2.move(160, 10)
        self.text_browser_2.setText(
            "Фишек на поле у игрока blue: {}".format(self._game_field.player_current_points_count(0)))
        self.text_browser_2.setFixedSize(140, 50)
        self.text_browser_3.move(10, 7 * self._scale + 50)
        self.text_browser_3.setText(
            "Фишки для размещения игрока red: {}".format(self._game_field.player_free_chips(1)))
        self.text_browser_3.setFixedSize(140, 50)
        self.text_browser_4.move(160, 7 * self._scale + 50)
        self.text_browser_4.setText(
            "Фишек на поле у игрока red: {}".format(self._game_field.player_current_points_count(1)))
        self.text_browser_4.setFixedSize(140, 50)

        current_color = "red" if self._current_player == 1 else "blue"
        self.text_browser_5.move(10, 4 * self._scale + 50)
        self.text_browser_5.setText(
            "Ходит игрок: {}".format(current_color))
        self.text_browser_5.setFixedSize(140, 50)

        self.text_browser_6.move(10, 2 * self._scale + 50)
        self.text_browser_6.setText(
            "Заблокированных фишек игрока blue: {}".format(self._game_field.blocked_chips(0)))
        self.text_browser_6.setFixedSize(140, 50)
        self.text_browser_7.move(10, 6 * self._scale + 50)
        self.text_browser_7.setText(
            "Заблокированных фишек игрока red: {}".format(self._game_field.blocked_chips(1)))
        self.text_browser_7.setFixedSize(140, 50)

        reachable_points = self._game_field.reachable_points()
        for point in reachable_points:
            self._points[point].opened()

        self.setGeometry(0, 0, 10 * self._scale, 9 * self._scale)
        self.setWindowTitle("Nine men's morris")
        self.show()

        player_color = "Вы" if self._current_player == 0 else "Компьютер"
        QMessageBox.question(self, "Первый ход", "Первым ходит игрок: {}".format(player_color), QMessageBox.Ok)

    def step(self):
        if self._mill_counter > 0:
            if self._game_field.delete_chip(self._current_player, self._handling_point):
                for point, button in self._points.items():
                    if point == self._handling_point:
                        button.opened()
                self._mill_counter -= 1
            else:
                QMessageBox.question(self, "Мельница", "Выберите фишку соперника.",
                                     QMessageBox.Ok)

        elif not self._second_phase:
            self._game_field.put_chip(self._current_player, self._handling_point)

            if not (self._game_field.player_can_put(0) or self._game_field.player_can_put(1)):
                self._second_phase = True
                self._end_of_moving = False

            self._mill_counter = self._game_field.mill_made(self._handling_point)
            self._current_player = (self._current_player + 1) % 2

        elif self._end_of_moving:
            self._game_field.move(self._current_player, self._previous_point, self._handling_point)

            self._mill_counter = self._game_field.mill_made(self._handling_point)
            self._current_player = (self._current_player + 1) % 2

            if self._game_field.player_defeat((self._current_player + 1) % 2):
                self._current_player = (self._current_player + 1) % 2
                current_color = "red" if self._current_player == 1 else "blue"
                QMessageBox.question(self, "Конец", "{} player win!".format(current_color))

            self._end_of_moving = False

            for point in self._was_blocked:
                if self._game_field.point_free(point):
                    self._points[point].opened()
        else:
            self._previous_point = self._handling_point
            if self._game_field.point_free(self._previous_point, self._current_player + 1):
                self._end_of_moving = True
                self._was_blocked = []
                reachable_points = self._game_field.reachable_points(self._previous_point, 0)
                for point, button in self._points.items():
                    if point not in reachable_points:
                        button.blocked()
                        self._was_blocked.append(point)

        self.text_browser_1.setText(
            "Фишки для размещения игрока blue: {}".format(self._game_field.player_free_chips(0)))
        self.text_browser_2.setText(
            "Фишек на поле у игрока blue: {}".format(self._game_field.player_current_points_count(0)))
        self.text_browser_3.setText(
            "Фишки для размещения игрока red: {}".format(self._game_field.player_free_chips(1)))
        self.text_browser_4.setText(
            "Фишек на поле у игрока red: {}".format(self._game_field.player_current_points_count(1)))
        self.text_browser_7.setText(
            "Заблокированных фишек игрока red: {}".format(self._game_field.blocked_chips(1)))
        self.text_browser_6.setText(
            "Заблокированных фишек игрока blue: {}".format(self._game_field.blocked_chips(0)))
        current_color = "red" if self._current_player == 1 else "blue"
        message = "Ходит игрок: " if self._mill_counter == 0 else "Удаление фишки игрока: "
        self.text_browser_5.setText(
            "{0}: {1}".format(message, current_color))





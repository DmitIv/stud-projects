from pyFiles.interface import *


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = NineMensMorrisGame()
    # window.resize(640, 480)
    window.show()
    sys.exit(app.exec_())

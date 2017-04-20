#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
from PyQt5 import QtWidgets
from ui.main_window import Ui_MainWindow

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = Ui_MainWindow()
    form.show()
    form.raise_()
    sys.exit(app.exec_())

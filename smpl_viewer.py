#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
from PyQt4 import QtGui
from ui.main_window import Ui_MainWindow

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    form = Ui_MainWindow()
    form.show()
    form.raise_()
    sys.exit(app.exec_())

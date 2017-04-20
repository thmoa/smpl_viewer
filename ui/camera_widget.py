#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np

from PyQt5 import QtWidgets

from gen.camera_widget import Ui_CameraWidget as Ui_CameraWidget_Base


class Ui_CameraWidget(QtWidgets.QMainWindow, Ui_CameraWidget_Base):

    def __init__(self, camera, frustum, draw):
        super(self.__class__, self).__init__()
        self.setupUi(self)

        self.camera = camera
        self.frustum = frustum
        self.draw = draw

        self.pos_0.valueChanged[float].connect(lambda val: self._update_position(0, val))
        self.pos_1.valueChanged[float].connect(lambda val: self._update_position(1, val))
        self.pos_2.valueChanged[float].connect(lambda val: self._update_position(2, val))

        self.rot_0.valueChanged[float].connect(lambda val: self._update_rotation(0, val))
        self.rot_1.valueChanged[float].connect(lambda val: self._update_rotation(1, val))
        self.rot_2.valueChanged[float].connect(lambda val: self._update_rotation(2, val))

        self.dist_0.valueChanged[float].connect(lambda val: self._update_distortion(0, val))
        self.dist_1.valueChanged[float].connect(lambda val: self._update_distortion(1, val))
        self.dist_2.valueChanged[float].connect(lambda val: self._update_distortion(2, val))
        self.dist_3.valueChanged[float].connect(lambda val: self._update_distortion(3, val))
        self.dist_4.valueChanged[float].connect(lambda val: self._update_distortion(4, val))

        self.center_0.valueChanged[float].connect(lambda val: self._update_center(0, val))
        self.center_1.valueChanged[float].connect(lambda val: self._update_center(1, val))

        self.focal_len.valueChanged[float].connect(self._update_focal_length)

        self.reset.clicked.connect(self._reset)

    def set_values(self, pos, rot,  focal_len, c, dist):
        self.pos_0.setValue(pos[0])
        self.pos_1.setValue(pos[1])
        self.pos_2.setValue(pos[2])

        self.rot_0.setValue(rot[0])
        self.rot_1.setValue(rot[1])
        self.rot_2.setValue(rot[2])

        self.dist_0.setValue(dist[0])
        self.dist_1.setValue(dist[1])
        self.dist_2.setValue(dist[2])
        self.dist_3.setValue(dist[3])
        self.dist_4.setValue(dist[4])

        self.center_0.setValue(c[0])
        self.center_1.setValue(c[1])

        self.focal_len.setValue(focal_len)

    def _update_position(self, id, val):
        self.camera.t[id] = val
        self.draw()

    def _update_rotation(self, id, val):
        self.camera.rt[id] = val
        self.draw()

    def _update_distortion(self, id, val):
        self.camera.k[id] = val
        self.draw()

    def _update_center(self, id, val):
        d = self.frustum['width'] if id == 0 else self.frustum['height']
        self.camera.c[id] = val * d
        self.draw()

    def _update_focal_length(self):
        w = self.frustum['width']
        self.camera.f[:] = np.array([w, w]) * self.focal_len.value()
        self.draw()

    def _reset(self):
        self.update_canvas = False

        self.pos_0.setValue(0)
        self.pos_1.setValue(0)
        self.pos_2.setValue(3)

        self.rot_0.setValue(0)
        self.rot_1.setValue(0)
        self.rot_2.setValue(0)

        self.dist_0.setValue(0)
        self.dist_1.setValue(0)
        self.dist_2.setValue(0)
        self.dist_3.setValue(0)
        self.dist_4.setValue(0)

        self.center_0.setValue(0.5)
        self.center_1.setValue(0.5)

        self.focal_len.setValue(1)

        self.update_canvas = True
        self.draw()

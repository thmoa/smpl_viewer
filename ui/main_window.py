#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
import numpy as np
import cv2
import ConfigParser

from PyQt4 import QtGui
from opendr.camera import ProjectPoints, Rodrigues
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight

from gen.main_window import Ui_MainWindow as Ui_MainWindow_Base
from camera_widget import Ui_CameraWidget
from smpl.smpl_webuser.serialization import load_model


class Ui_MainWindow(QtGui.QMainWindow, Ui_MainWindow_Base):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)

        self._moving = False
        self._rotating = False
        self._mouse_begin_pos = None
        self._loaded_gender = None
        self._update_canvas = False

        self.camera = ProjectPoints(rt=np.zeros(3), t=np.zeros(3))
        self.joints2d = ProjectPoints(rt=np.zeros(3), t=np.zeros(3))
        self.frustum = {'near': 0.1, 'far': 1000., 'width': 100, 'height': 30}
        self.light = LambertianPointLight(vc=np.array([0.94, 0.94, 0.94]), light_color=np.array([1., 1., 1.]))
        self.rn = ColoredRenderer(bgcolor=np.zeros(3), frustum=self.frustum, camera=self.camera, vc=self.light)

        self.model = None
        self._init_model('f')
        self.model.pose[0] = np.pi

        self.joints2d.set(v=self.model.J_transformed)

        self.camera_widget = Ui_CameraWidget(self.camera, self.frustum, self.draw)
        self.btn_camera.clicked.connect(lambda: self._show_camera_widget())

        for key, shape in self._shapes():
            shape.valueChanged[int].connect(lambda val, k=key: self._update_shape(k, val))

        for key, pose in self._poses():
            pose.valueChanged[int].connect(lambda val, k=key: self._update_pose(k, val))

        self.pos_0.valueChanged[float].connect(lambda val: self._update_position(0, val))
        self.pos_1.valueChanged[float].connect(lambda val: self._update_position(1, val))
        self.pos_2.valueChanged[float].connect(lambda val: self._update_position(2, val))

        self.radio_f.pressed.connect(lambda: self._init_model('f'))
        self.radio_m.pressed.connect(lambda: self._init_model('m'))

        self.reset_pose.clicked.connect(self._reset_pose)
        self.reset_shape.clicked.connect(self._reset_shape)
        self.reset_postion.clicked.connect(self._reset_position)

        self.canvas.wheelEvent = self._zoom
        self.canvas.mousePressEvent = self._mouse_begin
        self.canvas.mouseMoveEvent = self._move
        self.canvas.mouseReleaseEvent = self._mouse_end

        self.action_save.triggered.connect(self._save_config_dialog)
        self.action_open.triggered.connect(self._open_config_dialog)
        self.action_save_screenshot.triggered.connect(self._save_screenshot_dialog)
        self.action_save_mesh.triggered.connect(self._save_mesh_dialog)

        self.view_joints.triggered.connect(self.draw)
        self.view_joint_ids.triggered.connect(self.draw)
        self.view_bones.triggered.connect(self.draw)

        self._update_canvas = True

    def showEvent(self, event):
        self._init_camera()
        super(self.__class__, self).showEvent(event)

    def resizeEvent(self, event):
        self._init_camera()
        super(self.__class__, self).resizeEvent(event)

    def closeEvent(self, event):
        self.camera_widget.close()
        super(self.__class__, self).closeEvent(event)

    def draw(self):
        if self._update_canvas:
            img = np.array(self.rn.r)

            if self.view_joints.isChecked() or self.view_joint_ids.isChecked() or self.view_bones.isChecked():
                img = self._draw_annotations(img)

            self.canvas.setScaledContents(False)
            self.canvas.setPixmap(self._to_pixmap(img))

    def _draw_annotations(self, img):
        self.joints2d.set(t=self.camera.t, rt=self.camera.rt, f=self.camera.f, c=self.camera.c, k=self.camera.k)

        if self.view_bones.isChecked():
            kintree = self.model.kintree_table[:, 1:]
            for k in range(kintree.shape[1]):
                cv2.line(img, (int(self.joints2d.r[kintree[0, k], 0]), int(self.joints2d.r[kintree[0, k], 1])),
                         (int(self.joints2d.r[kintree[1, k], 0]), int(self.joints2d.r[kintree[1, k], 1])),
                         (0.24, 0.29, 0.91), 2)

        if self.view_joints.isChecked():
            for j in self.joints2d.r:
                cv2.circle(img, (int(j[0]), int(j[1])), 4, (0.38, 0.68, 0.15), -1)

        if self.view_joint_ids.isChecked():
            for k, j in enumerate(self.joints2d.r):
                cv2.putText(img, str(k), (int(j[0]), int(j[1])), cv2.FONT_HERSHEY_PLAIN, 1.2, (1, 1, 1))

        return img

    def _init_model(self, g):
        pose = None
        betas = None
        trans = None

        if self.model is not None:
            pose = self.model.pose.r
            betas = self.model.betas.r
            trans = self.model.trans.r

        if g == 'f':
            self.model = load_model('smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
        else:
            self.model = load_model('smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
        self._loaded_gender = g

        if pose is not None:
            self.model.pose[:] = pose
            self.model.betas[:] = betas
            self.model.trans[:] = trans

        self.light.set(v=self.model, f=self.model.f, num_verts=len(self.model))
        self.rn.set(v=self.model, f=self.model.f)
        self.camera.set(v=self.model)

        self.draw()

    def _init_camera(self):
        w = self.canvas.width()
        h = self.canvas.height()

        if w != self.frustum['width'] and h != self.frustum['height']:
            self.camera.set(rt=np.array([self.camera_widget.rot_0.value(), self.camera_widget.rot_1.value(),
                                         self.camera_widget.rot_2.value()]),
                            t=np.array([self.camera_widget.pos_0.value(), self.camera_widget.pos_1.value(),
                                        self.camera_widget.pos_2.value()]),
                            f=np.array([w, w]) * self.camera_widget.focal_len.value(),
                            c=np.array([w, h]) / 2.,
                            k=np.array([self.camera_widget.dist_0.value(), self.camera_widget.dist_1.value(),
                                        self.camera_widget.dist_2.value(), self.camera_widget.dist_3.value(),
                                        self.camera_widget.dist_4.value()]))

            self.frustum['width'] = w
            self.frustum['height'] = h

            self.light.set(light_pos=Rodrigues(self.camera.rt).T.dot(self.camera.t) * -10.)
            self.rn.set(frustum=self.frustum, camera=self.camera)

            self.draw()

    def _save_config_dialog(self):
        filename = QtGui.QFileDialog.getSaveFileName(self, 'Save config', '~', 'Config File (*.ini)')
        if filename:
            with open(str(filename), 'w') as fp:
                config = ConfigParser.ConfigParser()
                config.add_section('Model')
                config.set('Model', 'gender', self._loaded_gender)
                config.set('Model', 'shape', ','.join(str(s) for s in self.model.betas.r))
                config.set('Model', 'pose', ','.join(str(p) for p in self.model.pose.r))
                config.set('Model', 'translation', ','.join(str(p) for p in self.model.trans.r))

                config.add_section('Camera')
                config.set('Camera', 'translation', ','.join(str(t) for t in self.camera.t.r))
                config.set('Camera', 'rotation', ','.join(str(r) for r in self.camera.rt.r))
                config.set('Camera', 'focal_length', self.camera_widget.focal_len.value())
                config.set('Camera', 'distortion', ','.join(str(r) for r in self.camera.k.r))

                config.write(fp)

    def _open_config_dialog(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Load config', '~', 'Config File (*.ini)')
        if filename:
            config = ConfigParser.ConfigParser()
            config.read(str(filename))

            self._update_canvas = False
            self._init_model(config.get('Model', 'gender'))

            shapes = np.fromstring(config.get('Model', 'shape'), dtype=np.float64, sep=',')
            poses = np.fromstring(config.get('Model', 'pose'), dtype=np.float64, sep=',')
            position = np.fromstring(config.get('Model', 'translation'), dtype=np.float64, sep=',')

            for key, shape in self._shapes():
                val = shapes[key] / 5.0 * 50.0 + 50.0
                shape.setValue(val)
            for key, pose in self._poses():
                if key == 0:
                    val = (poses[key] - np.pi) / np.pi * 50.0 + 50.0
                else:
                    val = poses[key] / np.pi * 50.0 + 50.0
                pose.setValue(val)

            self.pos_0.setValue(position[0])
            self.pos_1.setValue(position[1])
            self.pos_2.setValue(position[2])

            cam_pos = np.fromstring(config.get('Camera', 'translation'), dtype=np.float64, sep=',')
            cam_rot = np.fromstring(config.get('Camera', 'rotation'), dtype=np.float64, sep=',')
            cam_dist = np.fromstring(config.get('Camera', 'distortion'), dtype=np.float64, sep=',')
            cam_f = config.getfloat('Camera', 'focal_length')

            self.camera_widget.set_values(cam_pos, cam_rot, cam_f, cam_dist)

            self._update_canvas = True
            self.draw()

    def _save_screenshot_dialog(self):
        filename = QtGui.QFileDialog.getSaveFileName(self, 'Save screenshot', '~', 'Images (*.png *.jpg *.ppm)')
        if filename:
            img = np.array(self.rn.r)
            if self.view_joints.isChecked() or self.view_joint_ids.isChecked() or self.view_bones.isChecked():
                img = self._draw_annotations(img)
            cv2.imwrite(str(filename), np.uint8(img * 255))

    def _save_mesh_dialog(self):
        filename = QtGui.QFileDialog.getSaveFileName(self, 'Save mesh', '~', 'Mesh (*.obj)')
        if filename:
            with open(filename, 'w') as fp:
                for v in self.model.r:
                    fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

                for f in self.model.f + 1:
                    fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def _zoom(self, event):
        delta = -event.delta() / 1200.0
        self.camera_widget.pos_2.setValue(self.camera_widget.pos_2.value() + delta)

    def _mouse_begin(self, event):
        if event.button() == 1:
            self._moving = True
        elif event.button() == 2:
            self._rotating = True
        self._mouse_begin_pos = event.pos()

    def _mouse_end(self, event):
        self._moving = False
        self._rotating = False

    def _move(self, event):
        if self._moving:
            delta = event.pos() - self._mouse_begin_pos
            self.camera_widget.pos_0.setValue(self.camera_widget.pos_0.value() + delta.x() / 500.)
            self.camera_widget.pos_1.setValue(self.camera_widget.pos_1.value() + delta.y() / 500.)
            self._mouse_begin_pos = event.pos()
        elif self._rotating:
            delta = event.pos() - self._mouse_begin_pos
            self.camera_widget.rot_0.setValue(self.camera_widget.rot_0.value() + delta.y() / 300.)
            self.camera_widget.rot_1.setValue(self.camera_widget.rot_1.value() - delta.x() / 300.)
            self._mouse_begin_pos = event.pos()

    def _show_camera_widget(self):
        self.camera_widget.show()
        self.camera_widget.raise_()

    def _update_shape(self, id, val):
        val = (val - 50) / 50.0 * 5.0
        self.model.betas[id] = val
        self.draw()

    def _reset_shape(self):
        self._update_canvas = False
        for key, shape in self._shapes():
            shape.setValue(50)
        self._update_canvas = True
        self.draw()

    def _update_pose(self, id, val):
        val = (val - 50) / 50.0 * np.pi

        if id == 0:
            val += np.pi

        self.model.pose[id] = val
        self.draw()

    def _reset_pose(self):
        self._update_canvas = False
        for key, pose in self._poses():
            pose.setValue(50)
        self._update_canvas = True
        self.draw()

    def _update_position(self, id, val):
        self.model.trans[id] = val
        self.draw()

    def _reset_position(self):
        self._update_canvas = False
        self.pos_0.setValue(0)
        self.pos_1.setValue(0)
        self.pos_2.setValue(0)
        self._update_canvas = True
        self.draw()

    def _poses(self):
        return enumerate([
            self.pose_0,
            self.pose_1,
            self.pose_2,
            self.pose_3,
            self.pose_4,
            self.pose_5,
            self.pose_6,
            self.pose_7,
            self.pose_8,
            self.pose_9,
            self.pose_10,
            self.pose_11,
            self.pose_12,
            self.pose_13,
            self.pose_14,
            self.pose_15,
            self.pose_16,
            self.pose_17,
            self.pose_18,
            self.pose_19,
            self.pose_20,
            self.pose_21,
            self.pose_22,
            self.pose_23,
            self.pose_24,
            self.pose_25,
            self.pose_26,
            self.pose_27,
            self.pose_28,
            self.pose_29,
            self.pose_30,
            self.pose_31,
            self.pose_32,
            self.pose_33,
            self.pose_34,
            self.pose_35,
            self.pose_36,
            self.pose_37,
            self.pose_38,
            self.pose_39,
            self.pose_40,
            self.pose_41,
            self.pose_42,
            self.pose_43,
            self.pose_44,
            self.pose_45,
            self.pose_46,
            self.pose_47,
            self.pose_48,
            self.pose_49,
            self.pose_50,
            self.pose_51,
            self.pose_52,
            self.pose_53,
            self.pose_54,
            self.pose_55,
            self.pose_56,
            self.pose_57,
            self.pose_58,
            self.pose_59,
            self.pose_60,
            self.pose_61,
            self.pose_62,
            self.pose_63,
            self.pose_64,
            self.pose_65,
            self.pose_66,
            self.pose_67,
            self.pose_68,
            self.pose_69,
            self.pose_70,
            self.pose_71,
        ])

    def _shapes(self):
        return enumerate([
            self.shape_0,
            self.shape_1,
            self.shape_2,
            self.shape_3,
            self.shape_4,
            self.shape_5,
            self.shape_6,
            self.shape_7,
            self.shape_8,
            self.shape_9,
        ])

    @staticmethod
    def _to_pixmap(im):
        if im.dtype == np.float32 or im.dtype == np.float64:
            im = np.uint8(im * 255)

        if len(im.shape) < 3 or im.shape[-1] == 1:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        qimg = QtGui.QImage(im, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888)

        return QtGui.QPixmap(qimg)

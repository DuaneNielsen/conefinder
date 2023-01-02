#!/usr/bin/env python3

import sys
import numpy as np

import argparse
import torch
import cv2
import pyzed.sl as sl

from pathlib import Path

import conefinder.conefinder
from conefinder.conefinder import objects, init, find, Timer

sys.path.insert(0, str(Path.home()) + '/yolov7')
sys.path.insert(0, str(Path.home()) + '/YOLOv7_Tensorrt')

import conefinder.ogl_viewer.viewer as gl
import conefinder.cv_viewer.tracking_viewer as cv_viewer

import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


class Viewer2D:
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 1)
        self.zed_ax = self.axes[0]
        self.zed_img = None
        self.cluster_ax = self.axes[1]
        self.cluster_img = None
        plt.pause(0.01)

    def update_zed(self, image):
        if self.zed_img is None:
            self.zed_img = self.zed_ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            self.zed_img.set_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.pause(0.01)

    def update_cluster(self, centers, dataset):
        self.cluster_ax.clear()
        self.cluster_ax.set_xlim(-5., 5.)
        self.cluster_ax.set_ylim(-2., 2.)
        self.cluster_ax.scatter(dataset[:, 0], dataset[:, 1])
        self.cluster_ax.scatter(centers[:, 0], centers[:, 1])
        # if self.zed_img is None:
        #     self.zed_img = self.cluster_ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # else:
        #     self.zed_img.set_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.pause(0.01)


def main(args_string=None, control_callback=None):
    global detections

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, help='model.pt path(s)')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640],
                        help='the image size the yolo was trained at')
    parser.add_argument('--engine', type=str, default=str(Path.home()/'yolov7.engine'), help='youlv7.engine path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--frame_skip', type=int, default=1, help='skip frames')
    parser.add_argument('--save', type=str, default=None, help='write to file')

    if args_string:
        opt = parser.parse_args(args_string.split())
    else:
        opt = parser.parse_args()

    zed, init_params, obj_param, runtime_params, obj_runtime_param = init(opt)

    # Display
    camera_infos = zed.get_camera_information()
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_infos.camera_resolution.width, 720),
                                    min(camera_infos.camera_resolution.height, 404))
    point_cloud_render = sl.Mat()
    viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    image_left = sl.Mat()

    # Utilities for 2D display
    display_resolution = sl.Resolution(min(camera_infos.camera_resolution.width, 1280),
                                       min(camera_infos.camera_resolution.height, 720))
    image_scale = [display_resolution.width / camera_infos.camera_resolution.width,
                   display_resolution.height / camera_infos.camera_resolution.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    # matplotlib 2D display
    plot2D = Viewer2D()

    # Utilities for tracks view
    camera_config = zed.get_camera_information().camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.camera_fps,
                                                    init_params.depth_maximum_distance)
    track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)

    # Camera pose
    cam_w_pose = sl.Pose()

    # state variables for detector loop
    frame = 0

    print('Viewers loaded')

    while viewer.is_available() and not conefinder.conefinder.exit_signal:

        best_track, _ = find(zed, runtime_params, obj_runtime_param)

        if best_track is not None and control_callback is not None:
            control_callback(best_track.position)

        if frame % opt.frame_skip != 0:
            continue

        # -- Display
        # Retrieve display data
        retrieve_display_data = Timer('retrieve display data')
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
        point_cloud.copy_to(point_cloud_render)
        zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
        zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)
        retrieve_display_data.print()

        render = Timer("render")
        # 3D rendering
        # viewer.updateData(point_cloud_render, objects)
        # 2D rendering
        np.copyto(image_left_ocv, image_left.get_data())
        cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)
        global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
        # global_image = cv2.hconcat([image_left_ocv])
        try:
            track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)
        except Exception:
            continue

        # plot2D.update_zed(global_image)
        # print('global image')
        cv2.namedWindow("ZED2i | 2D View and Birds View", cv2.WINDOW_AUTOSIZE)
        # print('create window')
        cv2.imshow("ZED2i | 2D View and Birds View", global_image)
        key = cv2.waitKey(10)
        render.print()
        if key & 0xFF == ord('q'):
            conefinder.conefinder.exit_signal = True
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    viewer.exit()
    exit(zed)


if __name__ == '__main__':
    controls = torch.zeros(2)
    STEERING, THROTTLE = 0, 1
    p = torch.tensor([1.0, 0.1])


    def control_callback(position):
        controls[STEERING] = position[0] * p[STEERING]
        controls[THROTTLE] = position[2] * p[STEERING]
        print(f'STEERING {controls[STEERING]}  THROTTLE {controls[THROTTLE]}')


    with torch.no_grad():
        main(control_callback=control_callback)

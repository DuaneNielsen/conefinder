#!/usr/bin/env python3

import sys
import numpy as np

import argparse
import torch
import cv2
import pyzed.sl as sl
import torch.backends.cudnn as cudnn

from pathlib import Path
sys.path.insert(0, str(Path.home()) + '/yolov7')

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device
from utils.augmentations import letterbox

from threading import Lock, Thread, Event
from time import sleep

import conefinder.ogl_viewer.viewer as gl
import conefinder.cv_viewer.tracking_viewer as cv_viewer
import time

import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from collections import deque
from conefinder.kmeans import cluster, random_init

timers = False


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


class Timer:
    def __init__(self, name):
        self.name = name
        self._start = time.time()

    def print(self):
        global timers
        if timers:
            print(f'{self.name}:{time.time() - self._start}')


lock = Lock()
run_signal = False
exit_signal = False
detector_loaded = Event()
detector_exited = Event()


def img_preprocess(img, device, half, net_size):
    net_image, ratio, pad = letterbox(img[:, :, :3], net_size, auto=False)
    net_image = net_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    net_image = np.ascontiguousarray(net_image)

    img = torch.from_numpy(net_image).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img, ratio, pad


def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5 * xywh[2]) * im_shape[1]
    x_max = (xywh[0] + 0.5 * xywh[2]) * im_shape[1]
    y_min = (xywh[1] - 0.5 * xywh[3]) * im_shape[0]
    y_max = (xywh[1] + 0.5 * xywh[3]) * im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_min
    output[2][1] = y_max

    output[3][0] = x_max
    output[3][1] = y_max
    return output


def detections_to_custom_box(detections, im, im0):
    output = []
    for i, det in enumerate(detections):
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                # Creating ingestable objects for the ZED SDK
                obj = sl.CustomBoxObjectData()
                obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
                obj.label = cls
                obj.probability = conf
                obj.is_grounded = False
                output.append(obj)
    return output


def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections, detector_loaded, detector_exited

    print("Intializing Network...")

    device = select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA
    imgsz = img_size

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
    cudnn.benchmark = True

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    detector_loaded.set()

    while not exit_signal:
        if run_signal:
            lock.acquire()
            img, ratio, pad = img_preprocess(image_net, device, half, imgsz)

            pred = model(img)[0]
            det = non_max_suppression(pred, conf_thres, iou_thres)

            # ZED CustomBox format (with inverse letterboxing tf applied)
            detections = detections_to_custom_box(det, img, image_net)
            lock.release()
            run_signal = False
        sleep(0.01)

    detector_exited.set()
    print('torch exited')


def main(args_string=None, control_callback=None):
    global image_net, exit_signal, run_signal, detections, detector_loaded, detector_exited

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--frame_skip', type=int, default=1, help='skip frames')
    parser.add_argument('--save', type=str, default=None, help='write to file')

    if args_string:
        opt = parser.parse_args(args_string.split())
    else:
        opt = parser.parse_args()

    capture_thread = Thread(target=torch_thread,
                            kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()

    print("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50
    if opt.svo is not None:
        init_params.svo_real_time_mode = False

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    image_left_tmp = sl.Mat()

    print("Initialized Camera")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = False
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

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

    detector_loaded.wait()

    # state variables for detector loop
    frame = 0
    object_queue = deque(maxlen=5)
    max_queue = deque(maxlen=5)

    while viewer.is_available() and not exit_signal:

        frame += 1

        zed_grab_time = Timer("zed_grab")
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed_grab_time.print()

            if frame % opt.frame_skip != 0:
                continue

            get_image_time = Timer("get_image")
            # -- Get the image
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            lock.release()
            run_signal = True
            get_image_time.print()

            time_detector = Timer("detector time")
            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)
            time_detector.print()

            wait_detection = Timer("wait_detection")
            # Wait for detections
            lock.acquire()
            # -- Ingest detections
            zed.ingest_custom_box_objects(detections)
            lock.release()
            wait_detection.print()
            zed.retrieve_objects(objects, obj_runtime_param)

            # -- Display
            # Retrieve display data
            retrieve_display_data = Timer('retrieve display data')
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
            point_cloud.copy_to(point_cloud_render)
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)
            retrieve_display_data.print()

            # determine cone positions by clustering prev detections using k-means
            with torch.no_grad():
                if len(objects.object_list) > 0:

                    # cluster detected tracks
                    object_queue.append(objects.object_list)
                    max_queue.append(len(objects.object_list))
                    for object in sorted(objects.object_list, key=lambda k: k.id):
                        print(
                            f'{object.id}, {object.position}, {object.velocity}, {object.tracking_state} {object.action_state}')

                    # current_frame_pos = torch.stack([torch.from_numpy(o.position) for o in objects.object_list])
                    prev_pos = torch.stack([torch.from_numpy(o.position) for t in object_queue for o in t]).to(
                        torch.float32)
                    centers = random_init(prev_pos, max(max_queue))
                    centers, codes = cluster(prev_pos, max(max_queue), centers)
                    plot2D.update_cluster(centers, prev_pos)

                    if control_callback:
                        # center [cone_pos, cone_pos, ...]
                        # cone_pos [left-right, up-down, distance]
                        control_callback(centers)

            render = Timer("render")
            # 3D rendering
            viewer.updateData(point_cloud_render, objects)
            # 2D rendering
            np.copyto(image_left_ocv, image_left.get_data())
            cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)
            global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
            # Tracking view
            try:
                track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)
            except Exception:
                continue

            plot2D.update_zed(global_image)
            # print('global image')
            # # cv2.namedWindow("ZED2i | 2D View and Birds View", cv2.WINDOW_AUTOSIZE)
            # print('create window')
            # # cv2.imshow("ZED2i | 2D View and Birds View", global_image)
            # print('waitkey')
            key = cv2.waitKey(10)
            render.print()
            if key & 0xFF == ord('q'):
                exit_signal = True
                break
        else:
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    viewer.exit()
    exit_signal = True
    detector_exited.wait()
    print("I'm about to sigsegv")
    capture_thread.join()
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()


if __name__ == '__main__':
    controls = torch.zeros(2)
    STEERING, THROTTLE = 0, 0
    p = torch.tensor([1.0, 0.1])

    def control_callback(centers):
        # this could jump around with random init, need a track ID fix
        controls[STEERING] = centers[0][0] * p[STEERING]
        controls[THROTTLE] = centers[0][2] * p[THROTTLE]
        print(controls)

    with torch.no_grad():
        main(control_callback=control_callback)

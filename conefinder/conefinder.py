import time
from threading import Lock, Event, Thread
from time import sleep

import numpy as np
from pyzed import sl as sl

import sys
from pathlib import Path
import conefinder.logs

log = conefinder.logs.setup(__package__).getChild(__name__.split('.')[-1])

def print(msg):
    log.debug(msg)


"""
expects yolov7 repo in home directory for pytorch
"""
# sys.path.insert(0, str(Path.home()) + '/yolov7')
# from models.experimental import attempt_load
# from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
# from utils.torch_utils import select_device
# from utils.datasets import letterbox

"""
expects YOLOv7_Tensorrt in home directory for Tensor RT
"""
sys.path.insert(0, str(Path.home()) + '/YOLOv7_Tensorrt')
from infer import TRT_engine_8_4_1_5 as TRT_engine
from datetime import datetime

timers = False

#
# def img_preprocess(img, device, half, net_size):
#     net_image, ratio, pad = letterbox(img[:, :, :3], net_size, auto=False)
#     net_image = net_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#     net_image = np.ascontiguousarray(net_image)
#
#     img = torch.from_numpy(net_image).to(device)
#     img = img.half() if half else img.float()  # uint8 to fp16/32
#     img /= 255.0  # 0 - 255 to 0.0 - 1.0
#
#     if img.ndimension() == 3:
#         img = img.unsqueeze(0)
#     return img, ratio, pad
#
#
# def xywh2abcd(xywh, im_shape):
#     output = np.zeros((4, 2))
#
#     # Center / Width / Height -> BBox corners coordinates
#     x_min = (xywh[0] - 0.5 * xywh[2]) * im_shape[1]
#     x_max = (xywh[0] + 0.5 * xywh[2]) * im_shape[1]
#     y_min = (xywh[1] - 0.5 * xywh[3]) * im_shape[0]
#     y_max = (xywh[1] + 0.5 * xywh[3]) * im_shape[0]
#
#     # A ------ B
#     # | Object |
#     # D ------ C
#
#     output[0][0] = x_min
#     output[0][1] = y_min
#
#     output[1][0] = x_max
#     output[1][1] = y_min
#
#     output[2][0] = x_min
#     output[2][1] = y_max
#
#     output[3][0] = x_max
#     output[3][1] = y_max
#     return output
#
#
# def detections_to_custom_box(detections, im, im0):
#     output = []
#     for i, det in enumerate(detections):
#         if len(det):
#             det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#
#             for *xyxy, conf, cls in reversed(det):
#                 xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#
#                 # Creating ingestable objects for the ZED SDK
#                 obj = sl.CustomBoxObjectData()
#                 obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
#                 obj.label = cls
#                 obj.probability = conf
#                 obj.is_grounded = False
#                 output.append(obj)
#     return output
#
#
# def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
#     global image_net, exit_signal, run_signal, detections, detector_loaded, detector_exited
#
#     print("Intializing Network...")
#
#     device = select_device()
#     half = device.type != 'cpu'  # half precision only supported on CUDA
#     imgsz = img_size
#
#     # Load model
#     model = attempt_load(weights, map_location=device)  # load FP32
#     stride = int(model.stride.max())  # model stride
#     imgsz = check_img_size(imgsz, s=stride)  # check img_size
#     if half:
#         model.half()  # to FP16
#     cudnn.benchmark = True
#
#     # Run inference
#     if device.type != 'cpu':
#         model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
#
#     detector_loaded.set()
#
#     while not exit_signal:
#         if run_signal:
#             lock.acquire()
#             img, ratio, pad = img_preprocess(image_net, device, half, imgsz)
#
#             pred = model(img)[0]
#             det = non_max_suppression(pred, conf_thres, iou_thres)
#
#             # ZED CustomBox format (with inverse letterboxing tf applied)
#             detections = detections_to_custom_box(det, img, image_net)
#             lock.release()
#             run_signal = False
#         sleep(0.01)
#
#     detector_exited.set()
#     print('torch exited')


def xyxy2abcd(x_min, y_min, x_max, y_max):
    output = np.zeros((4, 2))
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

    # output = np.clip(output, a_min=0, a_max=)

    return output


def tensorrt_detections_to_custom_box(detections):
    output = []
    for d in detections:
        xmin = int(d[2])
        ymin = int(d[3])
        xmax = int(d[4])
        ymax = int(d[5])
        clas = int(d[0])
        score = d[1]

        obj = sl.CustomBoxObjectData()
        # print(xmin, ymin, xmax, ymax)
        obj.bounding_box_2d = xyxy2abcd(xmin, ymin, xmax, ymax)
        obj.label = clas
        obj.probability = score
        obj.is_grounded = False
        output.append(obj)
    return output


capture_thread = None
frame = 0
image_left_tmp = sl.Mat()
objects = sl.Objects()
image_net = None
lock = Lock()
run_signal = False
exit_signal = False
detector_loaded = Event()
detector_exited = Event()


def tensort_thread(engine, img_size, conf_thres=0.5, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections, detector_loaded, detector_exited

    print("Intializing Network...")
    trt_engine = TRT_engine(engine, img_size, debug=True)
    detector_loaded.set()

    while not exit_signal:
        if run_signal:
            lock.acquire()
            predict_timer = Timer('predict')
            results = trt_engine.predict(image_net, threshold=conf_thres)
            predict_timer.print()

            # ZED CustomBox format (with inverse letterboxing tf applied)
            detections = tensorrt_detections_to_custom_box(results)
            lock.release()
            run_signal = False
        sleep(0.01)

    detector_exited.set()
    print('torch exited')


def init(opt):
    global capture_thread, detector_loaded
    capture_thread = Thread(target=tensort_thread,
                            kwargs={'engine': opt.engine, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()

    print("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.camera_resolution = sl.RESOLUTION.VGA
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50
    if opt.svo is not None:
        init_params.svo_real_time_mode = False

    runtime_params = sl.RuntimeParameters()
    runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    print("Initialized Camera")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)

    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    # wait for detector thread
    detector_loaded.wait()
    print('Detector loaded')

    return zed, init_params, obj_param, runtime_params, obj_runtime_param


def exit_finder(zed):
    exit_signal = True
    if capture_thread is not None:
        detector_exited.wait()
        print("I'm about to sigsegv")
        capture_thread.join()
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()


class ZEDGrabError(Exception): pass


def find(zed, runtime_params, obj_runtime_param):
    global image_net, exit_signal, run_signal, detections, detector_loaded, detector_exited, image_left_tmp

    zed_grab_time = Timer("zed_grab")
    if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
        raise ZEDGrabError('Grab failed')
    zed_grab_time.print()

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

    print(f'***** CURRENT FRAME : {datetime.now()} ******************')
    best_track = None
    best_confidence = 0.
    for object in objects.object_list:
        if object.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
            if object.confidence > best_confidence:
                best_track = object
                best_confidence = object.confidence

    for object in objects.object_list:
        if best_track is not None and object.id == best_track.id:
            print('*******  SELECTED TRACK ****************************')
        print(
            f"id: {object.id} {object.confidence} {object.tracking_state} pos: f{object.position} vel:{object.velocity}")
        if best_track is not None and object.id == best_track.id:
            print('****************************************************')

    return best_track


class Timer:
    def __init__(self, name):
        self.name = name
        self._start = time.time()

    def print(self):
        global timers
        if timers:
            print(f'{self.name}:{time.time() - self._start}')

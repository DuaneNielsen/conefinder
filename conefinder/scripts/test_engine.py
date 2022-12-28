import cv2
import pkg_resources

import sys
from pathlib import Path
sys.path.insert(0, str(Path.home()) + '/YOLOv7_Tensorrt')
from infer import TRT_engine_8_4_1_5 as TRT_engine
from argparse import ArgumentParser

def visualize(img, bbox_array):
    for temp in bbox_array:
        xmin = int(temp[2])
        ymin = int(temp[3])
        xmax = int(temp[4])
        ymax = int(temp[5])
        clas = int(temp[0])
        score = temp[1]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (105, 237, 249), 2)
        img = cv2.putText(img, "class:" + str(clas) + " " + str(round(score, 2)), (xmin, int(ymin) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (105, 237, 249), 1)
    return img


def main():
    resource_dir = pkg_resources.resource_filename("conefinder", "resources")


    parser = ArgumentParser()
    parser.add_argument('--engine', default=str(Path.home()/'yolov7.engine'))
    parser.add_argument('--image', default=resource_dir + "/images/cone.png")
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640],
                        help='the image size the yolo was trained at')
    args = parser.parse_args()

    trt_engine = TRT_engine(args.engine, args.img_size)
    img = cv2.imread(args.image)
    results = trt_engine.predict(img, threshold=0.5)
    img = visualize(img, results)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

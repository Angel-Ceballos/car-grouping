"""trt_yolo_cv.py

This script could be used to make object detection video with
TensorRT optimized YOLO engine.

"cv" means "create video"
made by BigJoon (ref. jkjung-avt)
"""


import os
import argparse
from pickle import TRUE
import time
import datetime
from pprint import pprint
import cv2
import numpy as np
import pycuda.autoinit  # This is needed for initializing CUDA driver
from trt_infer.infer import TensorRTInfer
from trt_infer.image_batcher import ImageBatcher
from trt_utils.functions import *
import matplotlib.pyplot as plt
import psutil

def parse_args():
    """Parse input arguments."""
    desc = ('Run the TensorRT optimized object detecion model on an input '
            'video and save BBoxed overlaid output as another video.')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-v', '--video', type=str, required=False,
        help='input video file name')
    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help='path to images')
    parser.add_argument(
        '-o', '--output', type=str,
        help='output video file name')
    parser.add_argument(
        '-s', '--show_vid', action='store_true',
        help='displays video with bboxes')
    parser.add_argument(
        '-z', '--detection_zone', action='store_true',
        help='shows detection zone and track')
    parser.add_argument(
        '-d', '--detect_car', action='store_true',
        help='enables car detection')
    parser.add_argument(
        '-t', '--threshold', type=float, default=0.3,
        help='threshold for inference')
    parser.add_argument(
        '-e', '--engine', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        "-p",
        "--preprocessor",
        default="V2",
        choices=["V1", "V1MS", "V2"],
        help="Select the image preprocessor to use, either 'V2', 'V1' or 'V1MS', default: V2",
    )
    args = parser.parse_args()
    return args

def read_stats(jetson):
    print(jetson.stats)

# @profile
def loop_and_detect(batcher, trt_resnet, viz, show=True):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cap: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
      writer: the VideoWriter object for the output video.
    """
    fps_list = []
    for img, batch,_ in batcher.get_batch():
        start_time = time.time()
        prediction = trt_resnet.infer(batch)
        fps_list.append(int(1/(time.time()-start_time)))
        
    cv2.destroyAllWindows()       
    print('\nDone.')
    print(f'Avg FPS: {np.array(fps_list).sum()/len(fps_list)}')
    return


def main():
    args = parse_args()
    if not os.path.isfile('%s' % args.engine):
        raise SystemExit('ERROR: file (%s) not found!' % args.engine)

    img_fn_list = os.listdir(args.input)
    if not img_fn_list:
        raise SystemExit('ERROR: failed to open the input img dir!')
    trt_infer = TensorRTInfer(args.engine)
    batcher = ImageBatcher(args.input, *trt_infer.input_spec() ,preprocessor=args.preprocessor)

    loop_and_detect(batcher, trt_infer, args.show_vid)


if __name__ == '__main__':
    main()

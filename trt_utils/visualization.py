"""visualization.py

The BBoxVisualization class implements drawing of nice looking
bounding boxes based on object detection results.
"""


from pickle import TRUE
import numpy as np
import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


# Constants
ALPHA = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def gen_colors(num_colors):
  """Generate different colors.

  # Arguments
    num_colors: total number of colors/classes.

  # Output
    bgrs: a list of (B, G, R) tuples which correspond to each of
          the colors/classes.
  """
  import random
  import colorsys

  hsvs = [[float(x) / num_colors, 1., 0.7] for x in range(num_colors)]
  random.seed(1234)
  random.shuffle(hsvs)
  rgbs = list(map(lambda x: list(colorsys.hsv_to_rgb(*x)), hsvs))
  bgrs = [(int(rgb[2] * 255), int(rgb[1] * 255),  int(rgb[0] * 255))
          for rgb in rgbs]
  return bgrs


def draw_boxed_text(img, text, topleft, color):
  """Draw a transluent boxed text in white, overlayed on top of a
  colored patch surrounded by a black border. FONT, TEXT_SCALE,
  TEXT_THICKNESS and ALPHA values are constants (fixed) as defined
  on top.

  # Arguments
    img: the input image as a numpy array.
    text: the text to be drawn.
    topleft: XY coordinate of the topleft corner of the boxed text.
    color: color of the patch, i.e. background of the text.

  # Output
    img: note the original image is modified inplace.
  """
  assert img.dtype == np.uint8       
  img_h, img_w, _ = img.shape
  if topleft[0] >= img_w or topleft[1] >= img_h:
      return img
  margin = 3
  size = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)
  w = size[0][0] + margin * 2
  h = size[0][1] + margin * 2
  # the patch is used to draw boxed text
  patch = np.zeros((h, w, 3), dtype=np.uint8)
  patch[...] = color
  cv2.putText(patch, text, (margin+1, h-margin-2), FONT, TEXT_SCALE,
              WHITE, thickness=TEXT_THICKNESS, lineType=cv2.LINE_8)
  cv2.rectangle(patch, (0, 0), (w-1, h-1), BLACK, thickness=1)
  w = min(w, img_w - topleft[0])  # clip overlay at image boundary
  h = min(h, img_h - topleft[1])
  # Overlay the boxed text onto region of interest (roi) in img
  roi = img[topleft[1]:topleft[1]+h, topleft[0]:topleft[0]+w, :]
  cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)
  return img

def draw_polygon(img, points, zone_color):
  poly_mask = img.copy()
  alpha_mask = 0.3
  cv2.fillPoly(poly_mask, [points], color=zone_color)
  img = cv2.addWeighted(poly_mask, alpha_mask, img, 1 - alpha_mask, 0)
  cv2.polylines(img, [points], isClosed=True, color=zone_color, thickness=2)
  return img

def validate_detection(polygon, centroid):
  point = Point(centroid)
  return polygon.contains(point)


class BBoxVisualization():
    """BBoxVisualization class implements nice drawing of boudning boxes.

    # Arguments
      cls_dict: a dictionary used to translate class id to its name.
    """

    def __init__(self, cls_dict, d_zone):
        self.cls_dict = cls_dict
        self.colors = gen_colors(len(cls_dict))
        self.d_zone = np.array(d_zone, np.int32)
        self.polygon = Polygon(self.d_zone)

    def draw_bboxes(self, img, boxes, confs, clss, draw_zone, detect):
        """Draw detected bounding boxes on the original image."""
        validation, is_valid = False, False
        valid_list = []
        zone_color = (226, 43, 138)
        
        for bb, cf, cl in zip(boxes, confs, clss):
            cl = int(cl)
            x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
            color = self.colors[cl]
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
            txt_loc = (max(x_min+2, 0), max(y_min+2, 0))
            cls_name = self.cls_dict.get(cl, 'CLS{}'.format(cl))
            txt = '{} {:.2f}'.format(cls_name, cf)
            img = draw_boxed_text(img, txt, txt_loc, color)
            # car class
            if detect and cl == 1:
              centroid = (int(x_min+x_max)//2, int(y_min+y_max)//2)
              cv2.circle(img, centroid, radius=5, color=color, thickness=-1)
              validation = validate_detection(self.polygon, centroid)
              valid_list.append(validation)
            # license plate class
            elif detect and cl == 0:
              valid_list.append(False)
        
        if detect:
            is_valid = any(valid_list) == True
            if is_valid:  zone_color = (0, 255, 0)

        if draw_zone:
            img = draw_polygon(img, self.d_zone, zone_color)

        return img, is_valid, valid_list


class TCD5KVisualizer:
    def __init__(self, margin=64, save=False) -> None:
        self.margin = margin
        self.save = save
        self.img_list = []
    
    def paint(self, c_key, keypoints):
        left_d = keypoints[1] - keypoints[0]
        top_d = keypoints[2] -  keypoints[0]
        right_d = keypoints[3] - keypoints[0]
        buttom_d = keypoints[4] - keypoints[0]

        # towball
        cv2.circle(img=self.frame, center=(943, 1024), radius=15, color=(160,114,0), thickness=2)
        
        x_len = right_d[0]
        y_len = buttom_d[1]
        cv2.circle(img=self.frame, center=c_key, radius=4, color=(255,0,255), thickness=-1)
        cv2.ellipse(img=self.frame, center=c_key, axes=(x_len, y_len), angle=0, startAngle=0, endAngle=360, color=(255, 0, 255), thickness=2)
        cv2.circle(img=self.frame, center=c_key+left_d, radius=2, color=(0,255,0), thickness=-1)
        cv2.circle(img=self.frame, center=c_key+top_d, radius=2, color=(0,255,0), thickness=-1)
        cv2.circle(img=self.frame, center=c_key+right_d, radius=2, color=(0,255,0), thickness=-1)
        cv2.circle(img=self.frame, center=c_key+buttom_d, radius=2, color=(0,255,0), thickness=-1)
        cv2.rectangle(self.frame, c_key-self.margin, c_key+self.margin, (0, 128, 255), 2)
        if self.save:
            self.img_list.append(self.frame)

    def update(self, img, c_key, keypoints=np.array([])):
        self.frame = img
        self.paint(c_key, keypoints)
        self.frame = cv2.resize(self.frame, (960, 640))
        return self.frame
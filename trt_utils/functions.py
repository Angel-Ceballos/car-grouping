import json
import os
import cv2
import requests
import io
import matplotlib.pyplot as plt

def get_src_img_center(r_center, prev_center, margin=64):
    diff = r_center - margin
    new_center = prev_center + diff
    return new_center

def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def get_cls_dict(category_list):
    """Get the class ID to name translation dictionary."""
    return {i: n for i, n in enumerate(category_list)}

def crop_objects(frame, boxes, path, object_list, curr_frame, save = False):
    img_list = []
    for i, object in enumerate(object_list):
        # not in detction zone
        if not object:
            continue
        
        # crop bounding box of car
        x_min, y_min, x_max, y_max = boxes[i]
        margin = 15
        cropped_car = frame[int(y_min)-margin:int(y_max)+margin, int(x_min)-margin:int(x_max)+margin]
        if save:
            img_name = "car_" + str(i) + "_frame_" + str(curr_frame) + ".png"
            img_path = os.path.join(path, img_name)
            cv2.imwrite(img_path, cropped_car)
        # append car just in case more than one is detected
        img_list.append(cropped_car)
    
    return img_list

def ocr_api(img, key, regions):
    buf = io.BytesIO()
    plt.imsave(buf, img, format="png")
    img_data = buf.getvalue()
    res = requests.post(
        'https://api.platerecognizer.com/v1/plate-reader/',
        data=dict(regions=regions),  # Optional
        files=dict(upload=img_data),
        headers={'Authorization': f'Token {key}'})
    #res = response.json()['results'][0]['plate']
    return res.json()
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import PIL
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import yolo_head, draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, preprocess_image
from io import BytesIO


class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
yolo_model = load_model("model_data/", compile=False)


def threshold_filter(boxes, box_confidence, box_class_probs, threshold = .6):
    
    box_scores = box_confidence * box_class_probs

    box_classes = tf.math.argmax(box_scores, axis = -1)
    box_class_scores = tf.math.reduce_max(box_scores, axis = -1)
    
    filtering_mask = (box_class_scores >= threshold)

    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes


def non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
 
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')

    nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes_tensor,iou_threshold)

    scores = tf.gather(scores,nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)
 
    return scores, boxes, classes


def box_conversion(box_xy, box_wh):
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])


def output_process(yolo_outputs, image_shape = (720, 1280), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    
    boxes = box_conversion(box_xy, box_wh)
    
    scores, boxes, classes = threshold_filter(boxes, box_confidence, box_class_probs, score_threshold)
    
    boxes = scale_boxes(boxes, image_shape)

    scores, boxes, classes = non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes


def predict(image_file):

    image, image_data = preprocess_image(image_file, model_image_size = (608, 608))
    
    yolo_model_outputs = yolo_model(image_data)
    yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
    
    out_scores, out_boxes, out_classes = output_process(yolo_outputs, [image.size[1],  image.size[0]], 10, 0.3, 0.5)

    colors = get_colors_for_classes(len(class_names))

    draw_boxes(image, out_boxes, out_classes, class_names, out_scores)
    image = np.array(image)

    return image


def save_image(img):
    pil_img = Image.fromarray(img)
    buffer = BytesIO()
    pil_img.save(buffer, format = "PNG")
    byte_img = buffer.getvalue()
    return byte_img


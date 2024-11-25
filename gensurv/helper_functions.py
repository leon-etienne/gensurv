from datetime import datetime
from IPython.display import Video
from tqdm import tqdm
from ultralytics import YOLO
import cv2
import ffmpegcv
import numpy as np
import PIL
import skimage
import torch
# Boxes id
# Masks id
# Center Positions with id
# Box positions with id
# Mask positions with id

# Shrink/Grow positions
# Contours Positions

# Composite over
# Composite transparency

# Getting frames before and after
# Difference between frames
# Loading multiple videos

# Image2image
# Controlnet
# Inpainting
# Marigold Depth and normals

def process_results_to_masks(results, frame, classes=[], ids=[], color=(255, 255, 255), thickness=-1):
    """
    Generates a binary mask with objects as white (255) and the background as black (0).
    """
    classes = [classes] if isinstance(classes, (int, float)) else classes
    ids = [ids] if isinstance(ids, (int, float)) else ids
    masks = np.zeros_like(frame)
    for index, (mask, box) in enumerate(zip(results[0].masks.xy, results[0].boxes)):
        class_id = int(box.cls[0])
        if not classes or class_id in classes:
            points = np.int32([mask])
            cv2.drawContours(masks, points, contourIdx=-1, color=color, thickness=thickness)
        if not ids or index in ids:
            points = np.int32([mask])
            cv2.drawContours(masks, points, contourIdx=-1, color=color, thickness=thickness)
    return masks


def process_results_to_boxes(results, frame, classes=[], color=(255, 255, 255), thickness=-1):
    """
    Generates a mask with bounding boxes drawn in white (255) on a black (0) background.
    """
    classes = [classes] if isinstance(classes, (int, float)) else classes
    masks = np.zeros_like(frame)
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if not classes or class_id in classes:
            x0, y0, x1, y1 = box.xyxy[0].int().tolist()
            masks = cv2.rectangle(masks, (x0, y0), (x1, y1), color, thickness)
    return masks


def process_results_to_masks_normalized(results, frame, classes=[], color=(255, 255, 255), thickness=-1):
    """
    Generates a normalized mask with objects as 1 and the background as 0.
    """
    masks = process_results_to_masks(results, frame, classes=classes, color=color, thickness=thickness)
    masks = masks / 255.0
    return masks


def process_results_to_boxes_normalized(results, frame, classes=[], color=(255, 255, 255), thickness=-1):
    """
    Generates a normalized mask with bounding boxes as 1 and the background as 0.
    """
    masks = process_results_to_boxes(results, frame, classes=classes, color=color, thickness=thickness)
    masks = masks / 255.0
    return masks


def process_results_to_center_points(results, classes=[]):
    """
    Extracts the center points of bounding boxes as an array of coordinates.
    """
    classes = [classes] if isinstance(classes, (int, float)) else classes
    points = np.empty((0, 2), dtype=np.int32)
    
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if not classes or class_id in classes:
            x0, y0, x1, y1 = box.xyxy[0].int().tolist()
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2
            points = np.append(points, [np.array([center_x, center_y], np.int32)], axis=0)

    return points


def process_results_to_boxes_points(results, classes=[]):
    """
    Extracts the corner points of bounding boxes as an array of coordinates.
    """
    classes = [classes] if isinstance(classes, (int, float)) else classes
    points = np.empty((0, 2), dtype=np.int32)
    
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if not classes or class_id in classes:
            x0, y0, x1, y1 = box.xyxy[0].int().tolist()
            points = np.append(points, [np.array([x0, y0], np.int32)], axis=0)
            points = np.append(points, [np.array([x0, y1], np.int32)], axis=0)
            points = np.append(points, [np.array([x1, y1], np.int32)], axis=0)
            points = np.append(points, [np.array([x1, y0], np.int32)], axis=0)

    return points


def process_results_to_masks_points(results, classes=[]):
    """
    Extracts points from masks as an array of coordinates.
    """
    classes = [classes] if isinstance(classes, (int, float)) else classes
    points = np.empty((0, 2), dtype=np.int32)
    
    for mask, box in zip(results[0].masks.xy, results[0].boxes):
        class_id = int(box.cls[0])
        if not classes or class_id in classes:
            points = np.concatenate((points, mask.astype(np.int32)), axis=0)

    return points


def process_results_to_labels(results, model):
    """
    Extracts class labels for bounding boxes based on the model's class names.
    """
    labels = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        labels.append(model.names[class_id])
    return labels


def draw_lines_from_points(points, frame, isClosed=False, color=(0, 255, 0), thickness=2):
    """
    Draws lines connecting the points on a blank frame.
    """
    frame = np.zeros_like(frame)  # Avoid modifying the original image
    points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [points], isClosed=isClosed, color=color, thickness=thickness, lineType=cv2.LINE_8, shift=0)
    return frame


def draw_circles_from_points(points, frame, radius=5, color=(0, 255, 0), thickness=-1):
    """
    Draws circles on the specified points on a blank frame.
    """
    frame = np.zeros_like(frame)  # Avoid modifying the original image
    for point in points:
        cv2.circle(frame, tuple(point), radius, color, thickness)
    return frame


def draw_text_from_points(points, frame, labels, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=4):
    """
    Draws text labels at specified points on a blank frame.
    """
    frame = np.zeros_like(frame)  # Avoid modifying the original image
    if isinstance(labels, str):
        labels = [labels] * len(points)

    for point, label in zip(points, labels):
        text_size, baseline = cv2.getTextSize(label, fontFace, fontScale, thickness)
        text_width, text_height = text_size
        text_x = point[0] - text_width // 2
        text_y = point[1] + text_height // 2
        cv2.putText(frame, label, (text_x, text_y), fontFace, fontScale, color, thickness, cv2.LINE_AA)
    return frame


def combine_images_with_transparency(a, b, t):
    """
    Combines two images with transparency, where `t` controls the transparency of `b`.
    """
    t = max(0.0, min(1.0, t))  # Clamp t between 0.0 and 1.0
    if a.shape != b.shape:
        raise ValueError("Images must have the same dimensions and number of channels")
    result = np.clip(a.astype(np.float64) + t * b.astype(np.float64), 0, 255).astype(np.uint8)
    return result


def combine_images_with_mask(a, b):
    """
    Combines two images where non-black pixels in `b` are drawn over `a`.
    """
    if a.shape != b.shape:
        raise ValueError("Images must have the same dimensions and number of channels")
    
    result = a.copy()
    non_black_mask = np.any(b != [0, 0, 0], axis=-1)  # Find non-black pixels in `b`
    result[non_black_mask] = b[non_black_mask]

    return result
from collections import defaultdict
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
    Generates a binary mask with specified objects as white (255) and the background as black (0).
    
    Parameters:
        results (list): Detection results containing masks and bounding boxes.
        frame (np.array): The frame/image where the mask is generated.
        classes (list or int, optional): List of class IDs to include in the mask. Can be a single integer.
        ids (list or int, optional): List of instance IDs to include in the mask. Can be a single integer.
        color (tuple, optional): Color for the mask, default is white (255, 255, 255).
        thickness (int, optional): Thickness for the mask contours. Default is -1 (filled).
    
    Returns:
        np.array: Binary mask image with objects as specified.
    """
    # Ensure `classes` and `ids` are lists
    if isinstance(classes, (int, float)):
        classes = [classes]
    if isinstance(ids, (int, float)):
        ids = [ids]

    # Initialize an empty mask with the same dimensions as the frame
    masks = np.zeros_like(frame)

    # Iterate through detected masks and bounding boxes
    for mask, box in zip(results[0].masks.xy, results[0].boxes):
        class_id = int(box.cls[0])
        instance_id = int(box.id[0])

        # Check if the current detection matches the desired classes or ids
        include_mask = (
            (instance_id in ids) or
            (class_id in classes) or
            (not classes and not ids)
        )

        if include_mask:
            # Draw the mask contour on the binary mask
            points = np.int32([mask])
            cv2.drawContours(masks, points, contourIdx=-1, color=color, thickness=thickness)

    return masks


def process_results_to_boxes(results, frame, classes=[], ids=[], color=(255, 255, 255), thickness=-1):
    """
    Generates a mask with bounding boxes drawn in white (255) on a black (0) background.
    
    Parameters:
        results (list): Detection results containing bounding boxes.
        frame (np.array): The frame/image where the mask is generated.
        classes (list or int, optional): List of class IDs to include in the mask. Can be a single integer.
        ids (list or int, optional): List of instance IDs to include in the mask. Can be a single integer.
        color (tuple, optional): Color for the bounding boxes, default is white (255, 255, 255).
        thickness (int, optional): Thickness for the bounding box borders. Default is -1 (filled).
    
    Returns:
        np.array: Mask with bounding boxes drawn.
    """
    # Ensure `classes` and `ids` are lists
    if isinstance(classes, (int, float)):
        classes = [classes]
    if isinstance(ids, (int, float)):
        ids = [ids]

    # Initialize an empty mask with the same dimensions as the frame
    masks = np.zeros_like(frame)

    # Iterate through detected bounding boxes
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        instance_id = int(box.id[0])

        # Check if the current detection matches the desired classes or ids
        include_box = (
            (instance_id in ids) or
            (class_id in classes) or
            (not classes and not ids)
        )

        if include_box:
            # Extract bounding box coordinates and draw the rectangle
            x0, y0, x1, y1 = box.xyxy[0].int().tolist()
            masks = cv2.rectangle(masks, (x0, y0), (x1, y1), color, thickness)

    return masks



def process_results_to_masks_normalized(results, frame, classes=[], ids=[], color=(255, 255, 255), thickness=-1):
    """
    Generates a normalized mask with objects as 1 and the background as 0.
    """
    masks = process_results_to_masks(results, frame, classes=classes, ids=ids, color=color, thickness=thickness)
    masks = masks / 255.0
    return masks


def process_results_to_boxes_normalized(results, frame, classes=[], ids=[], color=(255, 255, 255), thickness=-1):
    """
    Generates a normalized mask with objects as 1 and the background as 0.
    """
    masks = process_results_to_boxes(results, frame, classes=classes, ids=ids, color=color, thickness=thickness)
    masks = masks / 255.0
    return masks


def process_results_to_center_points(results, classes=[], ids=[]):
    """
    Extracts the center points of bounding boxes as an array of coordinates.
    """
    # Ensure `classes` and `ids` are lists
    if isinstance(classes, (int, float)):
        classes = [classes]
    if isinstance(ids, (int, float)):
        ids = [ids]    
        
    points = np.empty((0, 2), dtype=np.int32)
    
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        instance_id = int(box.id[0])

        # Check if the current detection matches the desired classes or ids
        include_center = (
            (instance_id in ids) or
            (class_id in classes) or
            (not classes and not ids)
        )
        
        if include_center:
            x0, y0, x1, y1 = box.xyxy[0].int().tolist()
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2
            points = np.append(points, [np.array([center_x, center_y], np.int32)], axis=0)

    return points


def process_results_to_boxes_points(results, classes=[], ids=[]):
    """
    Extracts the corner points of bounding boxes as an array of coordinates.
    """
    # Ensure `classes` and `ids` are lists
    if isinstance(classes, (int, float)):
        classes = [classes]
    if isinstance(ids, (int, float)):
        ids = [ids]

    points = np.empty((0, 2), dtype=np.int32)
    
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        instance_id = int(box.id[0])

        # Check if the current detection matches the desired classes or ids
        include_boxes = (
            (instance_id in ids) or
            (class_id in classes) or
            (not classes and not ids)
        )
        
        if include_boxes:
            x0, y0, x1, y1 = box.xyxy[0].int().tolist()
            points = np.append(points, [np.array([x0, y0], np.int32)], axis=0)
            points = np.append(points, [np.array([x0, y1], np.int32)], axis=0)
            points = np.append(points, [np.array([x1, y1], np.int32)], axis=0)
            points = np.append(points, [np.array([x1, y0], np.int32)], axis=0)

    return points


def process_results_to_masks_points(results, classes=[], ids=[]):
    """
    Extracts points from masks as an array of coordinates.
    """
    # Ensure `classes` and `ids` are lists
    if isinstance(classes, (int, float)):
        classes = [classes]
    if isinstance(ids, (int, float)):
        ids = [ids]

    points = np.empty((0, 2), dtype=np.int32)
    
    for mask, box in zip(results[0].masks.xy, results[0].boxes):
        class_id = int(box.cls[0])
        instance_id = int(box.id[0])

        # Check if the current detection matches the desired classes or ids
        include_mask = (
            (instance_id in ids) or
            (class_id in classes) or
            (not classes and not ids)
        )
        
        if include_mask:
            points = np.concatenate((points, mask.astype(np.int32)), axis=0)

    return points


def process_results_to_labels(results, model, include_ids=False, include_classes=False, include_confidences=False, ids=[], classes=[]):
    """
    Processes bounding box results to extract optional information:
    IDs, class labels, and confidences, filtered by specified IDs and classes.
    
    Args:
        results: The results object containing bounding boxes.
        model: The model providing class names.
        include_ids (bool): Whether to include bounding box IDs in the output.
        include_classes (bool): Whether to include class labels in the output.
        include_confidences (bool): Whether to include confidences in the output.
        ids (list or single int/float): Filter for specific bounding box IDs. Defaults to None.
        classes (list or single int/float): Filter for specific class IDs. Defaults to None.
    
    Returns:
        List of formatted strings in the form:
        "id: {id} | class: {class} | confidence: {confidence}".
    """
    # Ensure `classes` and `ids` are lists
    if isinstance(classes, (int, float)):
        classes = [classes]
    if isinstance(ids, (int, float)):
        ids = [ids]

    output = []
    for box in results[0].boxes:
        instance_id = int(box.id[0])
        class_id = int(box.cls[0])
        
        # Determine if the box matches the filters
        include_mask = (
            (instance_id in ids) or
            (class_id in classes) or
            (not classes and not ids)
        )
        
        if include_mask:
            id_str = f"id: {instance_id}" if include_ids else ""
            class_str = f"class: {model.names[class_id]}" if include_classes else ""
            conf_str = f"confidence: {float(box.conf[0]):.2f}" if include_confidences else ""
            
            # Build the formatted string with separators
            formatted = " | ".join(filter(None, [id_str, class_str, conf_str]))
            output.append(formatted.strip())
    
    return output


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


def process_image_to_contours(frame, threshold=127, max_value=255, color=(255, 255, 255), thickness=2):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Apply a binary threshold to get a binary image
    _, thresh = cv2.threshold(gray, threshold, max_value, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a black image with the same resolution as the input frame
    contour_frame = np.zeros_like(frame)

    # Draw the contours in white (for RGB image, use (255, 255, 255) for white)
    cv2.drawContours(contour_frame, contours, -1, color, thickness)

    return contour_frame


def start_results_to_tracks():
    # Store the track history
    track_history = defaultdict(lambda: [])

    def process_results_to_tacks(results, frame, max_tracks=30, color=(230, 230, 230), thickness=5, ids=[], classes=[]):
        """
        Annotates tracks of detected objects on the frame, filtered by IDs or classes.

        Args:
            results: The YOLO detection results object.
            frame (np.array): The frame to annotate.
            max_tracks (int, optional): Maximum number of historical points to draw. Defaults to 30.
            color (tuple, optional): RGB color for the track lines. Defaults to (230, 230, 230).
            thickness (int, optional): Thickness of the track lines. Defaults to 5.
            ids (list or int/float, optional): Filter for specific track IDs. Defaults to [].
            classes (list or int/float, optional): Filter for specific class IDs. Defaults to [].

        Returns:
            np.array: Annotated frame with object tracks drawn.
        """
        frame = frame.astype(np.uint8)
        
        # Ensure `classes` and `ids` are lists
        if isinstance(classes, (int, float)):
            classes = [classes]
        if isinstance(ids, (int, float)):
            ids = [ids]

        annotated_frame = np.zeros_like(frame)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        # Plot the tracks
        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x, y, w, h = box
            include_track = (
                (track_id in ids) or
                (class_id in classes) or
                (not classes and not ids)
            )
            if include_track:
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point

                # Only draw the last `max_tracks` points
                draw_points = track[-max_tracks:]
                if len(draw_points) > 1:
                    points = np.hstack(draw_points).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=color, thickness=thickness)

        return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    return process_results_to_tacks
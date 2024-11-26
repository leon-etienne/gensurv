import cv2
import numpy as np
import PIL
import skimage
import torch
from datetime import datetime
from IPython.display import Video
from tqdm import tqdm
from ultralytics import YOLO
import ffmpegcv

def get_video_frames(input_video_filename, start=0, end=None, width=None, height=None, displayVideo=True):
    """
    Loads a video, extracts frames within the specified time interval, and returns them as a list.
    
    Parameters:
        input_video_filename (str): The filename of the input video.
        start (float, optional): Start time in seconds for the subclip. Defaults to 0.
        end (float, optional): End time in seconds for the subclip. If None, defaults to the end of the video.
        width (int, optional): The desired width for resizing the video frames. Defaults to None (no resizing).
        height (int, optional): The desired height for resizing the video frames. Defaults to None (no resizing).
    
    Returns:
        list: A list of frames (each frame is a NumPy array in RGB format).
        tuple: A tuple containing the number of frames, frames per second (fps), and video duration.
    """
    
    # Open the video file
    cap = cv2.VideoCapture(input_video_filename)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Calculate start and end frame numbers
    start_frame = int(start * fps)
    end_frame = int(end * fps) if end is not None else total_frames
    
    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    video_frames = []
    frame_number = start_frame
    
    while cap.isOpened() and frame_number < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize the frame if dimensions are provided
        if width or height:
            frame = cv2.resize(frame, (width, height) if width and height else None)
        
        video_frames.append(frame)
        frame_number += 1
    
    cap.release()
    
    # Display video information
    print(f"Number of frames: {len(video_frames)}")
    print(f"Frames per second (fps): {fps}")
    print(f"Duration: {duration:.2f} seconds")
    
    if displayVideo:
        display(Video(input_video_filename, width=width or 640, height=height or 360))
    
    return video_frames, fps, duration



def save_video_frames(processed_frames, output_filename, fps, duration=0, displayVideo=True):
    """
    Save the processed frames as a new video file using OpenCV.
    
    Parameters:
        processed_frames (list of ndarray): List of processed frames to save.
        output_filename (str): The filename for the output video.
        fps (float): The frame rate of the original video.
        width (int): The width of the frames to save.
        height (int): The height of the frames to save.
    """

    height, width = processed_frames[0].shape[:2]
    # Generate a timestamp for the output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output_filename = f"{timestamp}_{output_filename}"
    
    # Define the codec and create a VideoWriter object
    out = ffmpegcv.VideoWriter(full_output_filename, "h264", fps, resize=(width, height))
    
    # Write each frame to the video file
    for frame in processed_frames:
        # Ensure the frame is the correct size
        if (frame.shape[1], frame.shape[0]) != (width, height):
            frame = cv2.resize(frame, (width, height))
        
        out.write(frame)
    
    # Release the VideoWriter object
    out.release()
    
    # Display video information
    print(f"Video saved to: {full_output_filename}")
    
    if displayVideo:
        # Display the processed video with specified dimensions
        display(Video(url=full_output_filename, width=640, height=360))
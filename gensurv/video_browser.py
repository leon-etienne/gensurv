import ipywidgets as widgets
from IPython.display import display
from os import listdir
from os.path import isfile, join, getmtime
from datetime import datetime
import time
import threading


# Function to get all video files (MP4, WEBM, MOV) sorted by modification date
def list_sorted_video_files(directory):
    # Supported video formats
    video_extensions = ('.mp4', '.webm', '.mov')
    files = [
        (f, getmtime(join(directory, f))) 
        for f in listdir(directory) 
        if isfile(join(directory, f)) and f.lower().endswith(video_extensions)
    ]
    
    # Sort by modification time (newest first)
    sorted_files = sorted(files, key=lambda x: x[1], reverse=True)
    
    # Format the options for the dropdown
    options = [
        (f"{file} ({datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')})", join(directory, file))
        for file, mtime in sorted_files
    ]
    
    return options

# Function to update the dropdown when new videos appear
def update_dropdown_options(directory, dropdown, video):
    options = list_sorted_video_files(directory)
    current_options = dropdown.options
    
    # Update only if there are changes to avoid resetting dropdown
    if options != current_options:
        dropdown.options = options
        
        # Automatically select the first video if available
        if options:
            dropdown.value = options[0][1]
            video.value = open(dropdown.value, "rb").read()

# Function to play the selected video
def play_video(change, video):
    if change['new']:  # Check if there's a new selected value
        video_path = change['new']
        if video_path:
            video.value = open(video_path, "rb").read()

# Main widget creation function
def create_video_browser(directory):
    # Create the video widget
    video = widgets.Video(
        value=b'',
        format='mp4',
        width=600,
        height=400,
        autoplay=True,  # Automatically play when a video is selected
        loop=False
    )
    
    # Create the dropdown menu for video selection
    dropdown = widgets.Dropdown(
        options=list_sorted_video_files(directory),
        description="Select Video:",
        layout=widgets.Layout(width='500px')
    )
    
    # Add event listener to play video when a new option is selected
    dropdown.observe(lambda change: play_video(change, video), names='value')
    
    # Initial load of video files and auto-play the first video if exists
    update_dropdown_options(directory, dropdown, video)
    
    # Display the interface
    display(widgets.VBox([dropdown, video]))
import os
import subprocess
from tkinter import Tk, filedialog
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def get_video_duration(video_path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path], capture_output=True, text=True)

    duration = float(result.stdout.strip())

    return duration

def split_video_with_interval(video_path, interval_duration):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    video_directory = os.path.dirname(video_path)

    output_folder = os.path.join(video_directory, f"{video_name}_split")
    os.makedirs(output_folder, exist_ok=True)

    # Define the duration of each split (in seconds)
    split_duration = 60 * 5

    # Split the video into smaller parts
    video_duration = get_video_duration(video_path)
    for i in range(0, int(video_duration), split_duration + interval_duration):
        start_time = i
        end_time = min(i + split_duration, int(video_duration))
        output_path = os.path.join(output_folder, f"{video_name}_{i}-{end_time}.mp4")

        # Extract the subclip using ffmpeg
        ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_path)

    print("Video splitting completed!")

# Open file manager dialog to choose a video file
Tk().withdraw()  # Hide the main tkinter window
video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4")])
if not video_path:
    print("No video file selected. Exiting...")
    exit()

interval_duration = 60 * 5 * 3

split_video_with_interval(video_path, interval_duration)

import cv2
import numpy as np
import os
import subprocess

# Set paths
image_path = "inversion_without_well_path.png"  # Replace with your image file
temp_video_path = "temp_video.avi"  # Temporary AVI file
final_output_path = "output_sam2.mp4"  # Final MP4 file

# Load image
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Check if image was loaded
if image is None:
    raise FileNotFoundError(f"Error: Unable to load image. Check if '{image_path}' exists.")

# Convert to RGB if image has transparency (RGBA to RGB)
if image.shape[-1] == 4:  
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

# Resize to 720p (1280x720) for SAM2 compatibility
target_width, target_height = 1280, 720
image_resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

# Video settings
fps = 30  # Standard frame rate
duration = 5  # Video duration in seconds
frame_count = fps * duration

# Define OpenCV video writer (temporary AVI file)
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Use XVID for AVI (better FFmpeg support)
video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (target_width, target_height))

# Write the same frame multiple times
for _ in range(frame_count):
    video_writer.write(image_resized)

# Release video writer
video_writer.release()

# Convert AVI to MP4 using FFmpeg
ffmpeg_cmd = [
    "ffmpeg", "-y",  # Overwrite output if exists
    "-i", temp_video_path,  # Input AVI file
    "-c:v", "libx264",  # H.264 encoding
    "-pix_fmt", "yuv420p",  # Ensure correct pixel format
    "-movflags", "+faststart",  # Optimize for fast playback
    final_output_path  # Output MP4 file
]

# Run FFmpeg conversion
subprocess.run(ffmpeg_cmd, check=True)

# Remove temporary file
os.remove(temp_video_path)

# Verify the output
if os.path.exists(final_output_path):
    print(f"SAM2-compatible MP4 video saved as {final_output_path}")
else:
    print("Error: Failed to create the video.")

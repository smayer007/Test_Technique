import cv2
import os

# Directories
video_dir = '/home/samer/Tasks/DataPart2'
output_dir = '/home/samer/Tasks/YOLO_Task/Train_Data'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to extract frames from video
def extract_frames(video_path, output_folder, max_frames=50):
    # Open video file
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    success, image = video_capture.read()

    while success and frame_count < max_frames:
        # Construct the filename and save the frame
        frame_filename = os.path.join(output_folder, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, image)
        frame_count += 1
        success, image = video_capture.read()

    # Release the video capture object
    video_capture.release()

# Process all videos in the directory
for filename in os.listdir(video_dir):
    if filename.endswith('.MP4'):  # Adjust if your videos have a different extension
        video_path = os.path.join(video_dir, filename)
        video_name = os.path.splitext(filename)[0]
        video_output_dir = os.path.join(output_dir, video_name)

        # Create a folder for each video
        os.makedirs(video_output_dir, exist_ok=True)
        
        print(f"Processing {filename}...")
        extract_frames(video_path, video_output_dir)
        print(f"Frames extracted and saved to {video_output_dir}")

print("Processing completed.")


import argparse
import cv2
import os

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, help="input mp4 path")
parser.add_argument("--output_name", type=str, help="output name for the directory of image frames")

args = parser.parse_args()
video_path = args.video_path
output_name = args.output_name

if '~' in video_path:
    video_path = video_path.replace("~", os.path.expanduser("~"))

if not os.path.exists(video_path):
    print(f"File '{video_path}' does not exist.")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

directory_name = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(directory_name, "dataset_list", f"{output_name}")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "orig"), exist_ok=True)

frame_count = 0
while True:
    ret, frame = cap.read() # Read a frame from the video

    if not ret: # If no more frames are returned, break the loop
        break

    # Construct the filename for the current frame
    frame_filename = os.path.join(output_dir, "orig", f"{output_name}_{str(frame_count)}.jpg")

    # Save the frame as an image file
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

print(f"Successfully extracted {frame_count} frames to {output_dir}")


cap.release()
cv2.destroyAllWindows() # Close any OpenCV windows (if you were displaying frames)


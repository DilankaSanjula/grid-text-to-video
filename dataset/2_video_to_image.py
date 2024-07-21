import cv2
import os
import json

def extract_frames(video_name, video_file_dir, train_image_output_dir):
    # Create the output folder based on the video name
    output_folder = os.path.join(train_image_output_dir, video_name)
    print(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Construct the video file path
    video_name = f"{video_name}.mp4"
    video_path = os.path.join(video_file_dir, video_name)
    print(video_path)
    
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    save_count = 0
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % 4 == 0:
            frame_filename = os.path.join(output_folder, f"frame_{save_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            save_count += 1  # Increment the save count

        frame_count += 1  # Increment the frame count

    video_capture.release()
    print(f"Extracted {frame_count} frames from {video_name}")


train_videos_path = 'train_list_new.txt'
test_videos_path = 'test_list_new.txt'

with open(train_videos_path, "r") as train_video_file:
    train_videos = train_video_file.readlines()

with open(test_videos_path, "r") as test_video_file:
    test_videos = test_video_file.readlines()


train_image_dir = "train_video_to_images/"
test_image_dir = "test_video_to_images/"
video_file_dir = "MSRVTT/videos/all"


# for count, train_video in enumerate(train_videos, 1):
#     print(count)
#     video_name = train_video.strip()
#     extract_frames(video_name, video_file_dir, train_image_dir)

for count, test_video in enumerate(test_videos, 1):
    print(count)
    video_name = test_video.strip()
    extract_frames(video_name, video_file_dir, test_image_dir)


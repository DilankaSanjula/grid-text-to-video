import cv2
import os
import json

def extract_frames(video_path, image_path):
    # Get the video name without the extension and create a folder with that name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(os.path.dirname(image_path), video_name)

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        # Read a single frame
        ret, frame = video_capture.read()
        if not ret:
            break

        # Save the frame as an image file
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1

    # Release the video capture object
    video_capture.release()

    print(f"Extracted {frame_count} frames from {video_path} to {output_folder}")


train_annotations_file = "train_annotations.json"
test_annotations_file = "test_annotations.json"

with open(train_annotations_file, 'r') as file:
    train_video_count = 0
    test_video_count = 0

    data = json.load(file)
    print(type(data))
    annotations = data["train_annotations"]
    
    for annotation in annotations:
        print(annotation)





# # Example usage
# train_image_dir = "train_video_to_images/"
# test_image_dir = "test_video_to_images/"


# video_file_dir =  "MSRVTT/videos/all"  # Replace with your video file path
# extract_frames(video_file_path)
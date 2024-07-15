import os
import cv2
import numpy as np
from PIL import Image

def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)

    frames = []
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            resized_frame = cv2.resize(frame, (128, 128))
            frames.append(resized_frame )
        if len(frames) == num_frames:
            break

    cap.release()
    return frames

def create_grid_image(frames, grid_size=(4, 4), output_path='grid_image.jpg'):
    frame_height, frame_width = frames[0].shape[:2]
    grid_height = grid_size[0] * frame_height
    grid_width = grid_size[1] * frame_width

    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        start_y = row * frame_height
        start_x = col * frame_width
        grid_image[start_y:start_y + frame_height, start_x:start_x + frame_width] = frame

    grid_image = cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(grid_image)
    img.save(output_path)
    print(f"Saved grid image as {output_path}")

def process_videos_in_folder(folder_path, num_frames=16, grid_size=(4, 4), output_folder='4x4_grid_images'):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(folder_path):
        try:
            if filename.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(folder_path, filename)
                frames = extract_frames(video_path, num_frames=num_frames)
                if frames:
                    output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_grid.jpg")
                    create_grid_image(frames, grid_size=grid_size, output_path=output_path)
        except:
            print("skipped")

# Example usage
folder_path = 'resized_videos'
process_videos_in_folder(folder_path)
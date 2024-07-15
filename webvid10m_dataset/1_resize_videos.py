import os
from PIL import Image
from moviepy.editor import VideoFileClip


def crop_center_square(clip):
    width, height = clip.size
    min_dim = min(width, height)
    crop_x = (width - min_dim) // 2
    crop_y = (height - min_dim) // 2
    return clip.crop(x1=crop_x, y1=crop_y, width=min_dim, height=min_dim)


def process_videos_in_folder(folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(folder_path):
        print(filename)
        try:
            if filename.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(folder_path, filename)
                clip = VideoFileClip(video_path)
                clip_cropped = crop_center_square(clip)
                clip_resized = clip_cropped.resize(newsize=(512, 512))
                resized_video_path = os.path.join(output_folder, filename)
                clip_resized.write_videofile(resized_video_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Example usage
folder_path = 'webvid-10m'
output_folder = 'resized_videos'
process_videos_in_folder(folder_path, output_folder)
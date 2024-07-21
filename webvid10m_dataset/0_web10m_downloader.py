from datasets import load_dataset
import re
import requests
from pytube import YouTube
from datetime import timedelta
import os
import pickle

# ds = load_dataset("TempoFunk/webvid-10M")

# train_split = ds["train"]

train_pickle_path = '/content/drive/MyDrive/webvid-10m-dataset/train_split.pkl'

# Load the training split from the pickle file
with open(train_pickle_path, 'rb') as f:
    train_split = pickle.load(f)
print("Training split loaded from pickle file")

print(train_split[0])
print(train_split[0]['duration'])

save_folder_train = '/content/drive/MyDrive/webvid-10m-dataset/train_videos'
save_folder_test = '/content/drive/MyDrive/webvid-10m-dataset/test_videos'

def duration_to_seconds(duration):
    pattern = re.compile(r'PT(\d+H)?(\d+M)?(\d+S)?')
    match = pattern.match(duration)
    if not match:
        return 0
    parts = match.groups()
    hours = int(parts[0][:-1]) if parts[0] else 0
    minutes = int(parts[1][:-1]) if parts[1] else 0
    seconds = int(parts[2][:-1]) if parts[2] else 0
    total_seconds = timedelta(hours=hours, minutes=minutes, seconds=seconds).total_seconds()
    return int(total_seconds)


def download_video_train(content_url, file_name):
    response = requests.get(content_url, stream=True)
    try:
        if response.status_code == 200:
            file_path = os.path.join(save_folder_train, f"{file_name.replace(' ', '_')}.mp4")
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print(f"Downloaded video as {file_name}")
        else:
            print("Failed to download video")
    except:
        pass

def download_video_test(content_url, file_name):
    response = requests.get(content_url, stream=True)
    try:
        if response.status_code == 200:
            file_path = os.path.join(save_folder_test, f"{file_name.replace(' ', '_')}.mp4")
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print(f"Downloaded video as {file_name}")
        else:
            print("Failed to download video")
    except:
        pass

# Path to save the current count
video_index_path = '/content/drive/MyDrive/webvid-10m-dataset/resume_index.txt'

# Initialize the count
count = 0

# Load count from file if it exists
if os.path.exists(video_index_path):
    with open(video_index_path, 'r') as f:
        count = int(f.read().strip())
    print(f"Resuming download from count: {count}")


for i, video in enumerate(train_split):
    # Skip videos up to the last processed count
    if i < count:
        continue

    # Get the duration in seconds
    duration_seconds = duration_to_seconds(video['duration'])
    file_name = video['name']
    print(f"Duration in seconds: {duration_seconds}")

    if duration_seconds < 20:
        download_video_train(video['contentUrl'], file_name)
    
    # Update the count
    count += 1
    with open(video_index_path, 'w') as f:
        f.write(str(i))

    print(f"Downloaded video count: {count}")


# count = 0
# for video in test_split:
#     # Get the duration in seconds
#     duration_seconds = duration_to_seconds(video['duration'])
#     file_name = video['name']
#     print(f"Duration in seconds: {duration_seconds}")

#     count += 1
#     print(count)
#     if duration_seconds < 20:
#         download_video_test(video['contentUrl'], file_name)

#         if count > 30000:
#             break

#     else:
#         continue
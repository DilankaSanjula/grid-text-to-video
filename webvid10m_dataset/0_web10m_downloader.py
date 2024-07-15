from datasets import load_dataset
import re
import requests
from pytube import YouTube
from datetime import timedelta
import os


ds = load_dataset("TempoFunk/webvid-10M")

train_split = ds["train"]

print(train_split[0])
print(train_split[0]['duration'])

save_folder = 'webvid-10m'

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


def download_video(content_url, file_name):
    response = requests.get(content_url, stream=True)
    try:
        if response.status_code == 200:
            file_path = os.path.join(save_folder, f"{file_name.replace(' ', '_')}.mp4")
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print(f"Downloaded video as {file_name}")
        else:
            print("Failed to download video")
    except:
        pass


count = 0
for video in train_split:
    # Get the duration in seconds
    duration_seconds = duration_to_seconds(video['duration'])
    file_name = video['name']
    #print(f"Duration in seconds: {duration_seconds}")

    count += 1
    if count < 100:
        download_video(video['contentUrl'], file_name)

    else:
        break
import os
import random
import re

import cv2
import pandas as pd
import yaml
from torchvision.datasets.utils import download_url


with open('config.yaml') as f:
    config = yaml.safe_load(f)
VIDEOS_PATH = config['paths']['VIDEOS_PATH']


def get_urls(num_samples=None):
    urls = pd.read_csv('data/templates.csv')
    urls = urls.preview_media_id.drop_duplicates().tolist()
    if num_samples is not None:
        urls = random.sample(urls, num_samples)
    base_url = "https://res.cloudinary.com/lightricks/video/upload/"
    urls = [base_url + s + '.mp4' for s in urls]
    return urls


def download_video(url, destination="data/video_data"):
    extenstion = re.sub("https://res.cloudinary.com/lightricks/video/upload/", "", url)
    download_url(
        url,
        destination,
        extenstion
    )


def download_multiple_videos(url_list):
    for url in url_list:
        download_video(url)


def delete_video(video_path):
    os.remove(video_path)


def delete_multiple_videos(videos_path):
    for video_path in videos_path:
        delete_video(video_path)


def get_existing_path():
    files = os.listdir(VIDEOS_PATH)
    paths = [VIDEOS_PATH + i for i in files]
    return paths


def load_video(video_path):
    video = cv2.VideoCapture(video_path)
    return video


def load_multiple_videos(videos_path):
    videos = []
    for video_path in videos_path:
        video = load_video(video_path)
        videos.append(video)
    return videos


def build_dataset(num_samples):
    # TODO: check for duplications
    urls = get_urls(num_samples)
    download_multiple_videos(urls)


def save_encoding(frames_df, save_path, visited_path):
    names = frames_df['name'].drop_duplicates()

    if not os.path.isfile(visited_path):
        names.to_csv(visited_path)
    elif os.path.isfile(visited_path):
        names = names.tolist()
        visited = pd.read_csv(visited_path)
        visited = visited['name'].tolist()
        set_a = set(visited)
        set_b = set(names)
        difference = set_b - set_a
        names = pd.DataFrame({'name': list(difference)})
        names.to_csv(visited_path, mode='a', header=False)

    if not os.path.isfile(save_path):
        frames_df.to_csv(save_path)
    elif os.path.isfile(save_path) and len(names) > 0:
        frames_df.to_csv(save_path, mode='a', header=False)
    elif len(names) == 0:
        print("no new video was found")


def get_center_clips(previews):
    df = pd.read_csv('data/templates.csv')
    df = df[df['feature_types'].str.lower().isin(['clip', 'mixer'])]
    df = df[df['preview_media_id'].isin(previews)]
    df['center_time'] = (df['start_times'] + df['feature_durations'])/2
    return df


if __name__ == '__main__':
    build_dataset(10)

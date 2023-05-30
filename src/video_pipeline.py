import json
import os
import re
import pandas as pd

import yaml

import src.encode_video as ev
import src.video_utils as vu
from src.video_model import processor_clip, model_clip

with open('config.yaml') as f:
    config = yaml.safe_load(f)
VIDEOS_PATH = config['paths']['VIDEOS_PATH']
VID_EMB_PATH = config['paths']['VID_EMB_PATH']
VISITED_PATH = config['paths']['VISITED_PATH']


class VideoPipeline:
    def __init__(self, url):
        self.url = url
        self.media_preview = re.sub("https://res.cloudinary.com/lightricks/video/upload/|.mp4", "", self.url)
        self.video_path = VIDEOS_PATH + "/" + self.media_preview + ".mp4"

    def pipeline(self, df):
        vu.download_video(self.url)
        video = vu.load_video(self.video_path)
        centers = vu.get_center_clips(df, [self.media_preview])['center_time'].tolist()
        ve = ev.VideoEncode(video=video, processor=processor_clip, model=model_clip, sample_method='times',
                            times=centers)
        df = ve.get_frames_df(os.path.basename(self.video_path))
        vu.save_encoding(df, VID_EMB_PATH, VISITED_PATH)
        vu.delete_video(self.video_path)


def run_all(urls):
    # df = pd.read_json('data/templates.json', lines=True)
    rows = []
    with open('data/templates.json', 'r') as file:
        for line in file:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df = df.explode(['feature_types', 'start_times', 'feature_durations']).reset_index(drop=True)
    for i, j in enumerate(urls):
        print(i)
        try:
            vp = VideoPipeline(j)
            vp.pipeline(df)
        except:
            pass


if __name__ == '__main__':
    urls = vu.get_urls()
    run_all(urls)

import os
import re

import yaml

import src.encode_video as ev
import src.video_utils as vu
from src.video_model import processor_blip, model_blip

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

    def pipeline(self):
        vu.download_video(self.url)
        video = vu.load_video(self.video_path)
        centers = vu.get_center_clips([self.media_preview])['center_time'].tolist()
        ve = ev.VideoEncode(video=video, processor=processor_blip, model=model_blip,
                            model_frames=16, sample_method='times', times=centers)
        df = ve.get_frames_df(os.path.basename(self.video_path))
        vu.save_encoding(df, VID_EMB_PATH, VISITED_PATH)
        vu.delete_video(self.video_path)


def run_all(urls):
    for i in urls:
        try:
            vp = VideoPipeline(i)
            vp.pipeline()
        except:
            pass


if __name__ == '__main__':
    urls = vu.get_urls(100)
    run_all(urls)

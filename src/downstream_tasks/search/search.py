# TODO: finish search system
from urllib.error import HTTPError

import numpy as np
import yaml

import src.video_utils as vu
from src.comparing import coverage
from src.encode_text import TextEmbeddings

with open('config.yaml') as f:
    config = yaml.safe_load(f)
VIDEOS_PATH = config['paths']['VIDEOS_PATH']


class SearchVideo(TextEmbeddings):
    def __init__(self, video_encoding, text, model, tokenizer, num_results):
        super().__init__(text, model, tokenizer)
        self.video_encoding = video_encoding
        self.num_results = num_results
        self.path_list = self.matching_frames()

    def matching_frames(self):
        """
        Matches text embeddings with video encodings to find similar videos.

        Returns:
            path_list (list): List of paths to matching videos.
        """
        search_embeddings = self.get_embeddings().detach().numpy()
        similarities = coverage(search_embeddings, self.video_encoding.drop('name', axis=1).values)
        similarities = similarities[0].T
        top_indices = np.argsort(similarities)[::-1]
        distances = np.sort(similarities)[::-1]
        video_names = self.video_encoding.iloc[top_indices]['name'].drop_duplicates().tolist()

        # Ensure num_results of unique videos
        path_list = []
        seen_videos = set()
        for i, video_name in enumerate(video_names):
            if video_name not in seen_videos:
                path = video_name
                distance = distances[i]
                path_list.append((path, distance))
                seen_videos.add(video_name)
                if len(path_list) == self.num_results:
                    break

        return path_list

    def download_results(self):
        urls = ["https://res.cloudinary.com/lightricks/video/upload/" + i for i, _ in self.path_list]
        distances = [i for _, i in self.path_list]
        for i, url in enumerate(urls):
            try:
                vu.download_video(url)
                print(distances[i])
            except HTTPError as e:
                if e.code == 404:
                    print("The requested page was not found.")
                else:
                    print("An HTTP error occurred with status code:", e.code)

    def delete_results(self):
        delete_list = [VIDEOS_PATH + i for i in self.path_list]
        try:
            vu.delete_multiple_videos(delete_list)
        except ValueError:
            pass

# TODO: finish search system
import numpy as np
import pandas as pd

from src.comparing import coverage
import src.video_utils as vu
from src.encode_text import TextEmbeddings
import yaml

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
        search_embeddings = self.get_embeddings().detach().numpy()
        similarities = coverage(search_embeddings, self.video_encoding.iloc[:, 3:].values)
        similarities = similarities[0].T
        top_indices = np.argpartition(similarities, -self.num_results)[-self.num_results:]
        top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
        path_list = self.video_encoding.iloc[top_indices]['name'].drop_duplicates().tolist()
        return path_list

    def download_results(self):
        urls = ["https://res.cloudinary.com/lightricks/video/upload/" + i for i in self.path_list]
        vu.download_multiple_videos(urls)

    def delete_results(self):
        delete_list = [VIDEOS_PATH + i for i in self.path_list]
        try:
            vu.delete_multiple_videos(delete_list)
        except ValueError:
            pass


from src.text_model import text_model, tokenizer
embs = pd.read_csv('data/frames_encoding.csv')
# embs = df.iloc[:, 3:]
sv = SearchVideo(embs, "gaming", text_model, tokenizer, 2)
sv.download_results()

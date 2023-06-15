import pandas as pd
import streamlit as st
import yaml


from src.downstream_tasks.search.search import SearchVideo
from src.text_model import text_model, tokenizer

with open('config.yaml') as f:
    config = yaml.safe_load(f)
VIDEOS_PATH = config['paths']['VIDEOS_PATH']


def main():
    num_results = 10

    def _show_video(video_num, ids):
        assert video_num <= num_results, "the number of downloaded results is {}".format(num_results)
        video_paths = VIDEOS_PATH + '/' + ids[video_num][0]
        video_file = open(video_paths, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

    # load embeddings
    embs = pd.read_csv('data/frames_encoding.csv')
    embs = embs.iloc[:, [1] + list(range(3, len(embs.columns)))]
    embs_avg = embs.groupby('name').mean().reset_index()
    search_term = st.text_input('enter your search', 'Vacation in Greece')
    sv = SearchVideo(embs_avg, search_term, text_model, tokenizer, num_results)
    ids = sv.matching_frames()

    if st.button('Search'):
        sv.download_results()
    video_num = st.number_input('result number (out of {})'.format(num_results), min_value=0,
                                max_value=num_results)
    _show_video(video_num, ids)


main()





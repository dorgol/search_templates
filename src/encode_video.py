import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from scenedetect import detect, ContentDetector
from transformers.models.clip.modeling_clip import CLIPOutput


class VideoPreprocess:
    def __init__(self, video: cv2.VideoCapture, model_frames, sample_method="random",
                 starting_frame=None, video_path=None, times=None):
        self.video = video
        self.model_frames = model_frames
        self.sample_method = sample_method
        self.starting_frame = starting_frame
        self.video_path = video_path
        self.times = times
        self.indices = self.sample_frame_indices(video_frames=int(self.video.get(7)),
                                                 num_frames=self.model_frames,
                                                 sample_method=self.sample_method,
                                                 times=self.times)

    def sample_frame_indices(self, video_frames, num_frames, sample_method="linear", starting_frame=None, times=None):
        indices = None
        if sample_method == "linear":
            indices = np.round(np.linspace(0, video_frames, num=num_frames))
        elif sample_method == "random":
            indices = np.random.choice(video_frames, num_frames, replace=False)
            indices.sort(axis=0)
        elif sample_method == "consecutive":
            indices = np.array(range(starting_frame, starting_frame+self.model_frames))
        elif sample_method == "from_scene":
            scene_list = detect(self.video_path, ContentDetector(), show_progress=True)
            if len(scene_list) > 0:
                key_frame_ind = []
                for scene in scene_list:
                    first_frame = scene[0].get_frames()
                    last_frame = scene[1].get_frames()
                    index = np.random.randint(first_frame, last_frame, 1)
                    key_frame_ind.append(index[0])
                indices = np.array(key_frame_ind)
            elif len(scene_list) == 0:
                indices = [int(self.video.get(7)/2)]
        elif sample_method == "times":
            frame_rate = self.video.get(cv2.CAP_PROP_FPS)
            indices = []
            for time_stamp in times:
                frame_number = int(time_stamp * frame_rate)
                indices.append(frame_number)
            indices = np.array(indices)

        else:
            raise Exception("'sample_method' must be 'linear', 'random', 'consecutive', 'from_scene' or 'times")

        return indices

    def get_video_slice(self):
        def _get_frame(frame_num):
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            _, frame = self.video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return torch.tensor(frame)
        video = torch.stack([_get_frame(i) for i in self.indices])
        return video

    def show(self):
        fig = px.imshow(self.get_video_slice(), binary_string=True, facet_col=0, facet_col_wrap=8)
        fig.show()


class VideoEncode(VideoPreprocess):
    def __init__(self, video: cv2.VideoCapture, processor, model, model_frames, sample_method, times, tags=None):
        super().__init__(video, model_frames, sample_method, times)
        self.model = model
        self.processor = processor
        self.tags = tags

    def encode(self):
        video = self.get_video_slice()
        inputs = self.processor(
            images=list(video),
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

    def get_visual_embedding(self, outputs=None):
        if outputs is None:
            outputs = self.encode()
        return outputs.image_embeds

    def get_frames_df(self, video_name, outputs=None):
        if outputs is None:
            outputs = self.encode()
        embeddings = pd.DataFrame(self.get_visual_embedding(outputs).detach().numpy())
        video_name_list = [video_name] * self.model_frames
        df = pd.DataFrame({'name': video_name_list, 'indices': self.indices})
        frames_df = pd.concat([df, embeddings], axis=1)
        return frames_df

    def get_probability(self, outputs: CLIPOutput, frame_num: int):
        probs = outputs.logits_per_image.softmax(dim=1).tolist()[frame_num]
        probability = pd.DataFrame({'tags': self.tags, 'probability': probs})
        probability = probability.sort_values(by='probability', ascending=False)
        return probability

    def get_tags_per_frame(self, outputs):
        df = pd.DataFrame(self.tags)
        df_tags = df.iloc[outputs.logits_per_image.argmax(dim=1).tolist(), :]
        return df_tags

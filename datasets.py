import os
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch.utils.data
from PIL import Image
from torch.utils.data.dataset import T_co

import utils


class VideoDataset(torch.utils.data.IterableDataset):

    def __init__(self, root, im_w=64, im_h=64, nb_frame=2):
        self._root = Path(root)
        self._num_classes = 2

        self.im_w = im_w
        self.im_h = im_h
        self.nb_frame = nb_frame
        self.update_files()

    def update_files(self, max_num_video=None):
        self._files = []
        for c in range(self._num_classes):
            class_dir = self._root / str(c)
            files = list(sorted(class_dir.iterdir()))
            if max_num_video is not None and len(files) > max_num_video:
                old_files = files[:-max_num_video]
                files = files[-max_num_video:]
                for f in old_files:
                    os.remove(f)
            self._files.append(files)

    def _sample(self):
        video_1 = random.choice(self._files[0])
        video_2 = random.choice(self._files[1])

        video_1 = np.load(video_1)[0]
        video_2 = np.load(video_2)[0]
        t1 = random.randint(0, video_1.shape[0] - self.nb_frame)
        t2 = np.random.randint(0, video_2.shape[0] - self.nb_frame)
        video_1 = video_1[t1:t1 + self.nb_frame]
        video_2 = video_2[t2:t2 + self.nb_frame]
        if tuple(video_1.shape[1:3]) != (self.im_h, self.im_w):
            video_1 = VideoDataset.resize(video_1, self.im_w, self.im_h)
        if tuple(video_2.shape[1:3]) != (self.im_h, self.im_w):
            video_2 = VideoDataset.resize(video_2, self.im_w, self.im_h)

        step_1 = np.concatenate(list(video_1), axis=2).transpose(2, 0, 1)  # h x w x 3*nb
        step_2 = np.concatenate(list(video_2), axis=2).transpose(2, 0, 1)

        return step_1, step_2

    @staticmethod
    def resize(video, im_w, im_h):
        frame_list = []
        for t in range(video.shape[0]):
            frame = Image.fromarray(video[t])
            frame = np.array(frame.resize((im_w, im_h), Image.BICUBIC), dtype=np.float32)
            frame_list.append(frame)
        frame_list = np.stack(frame_list)
        return frame_list

    @staticmethod
    def sample_from_dir(video_dir, episode_len=None):
        if episode_len is not None:
            episode_len += 1
        else:
            episode_len = -1

        video_dir = Path(video_dir)
        files = list(video_dir.iterdir())
        video_i = np.load(random.choice(files))[0, :episode_len]
        return video_i

    @staticmethod
    def transform_frames(frames, im_w, im_h):
        if tuple(frames.shape[1:3]) != (im_h, im_w):
            frames = VideoDataset.resize(frames, im_w, im_h)
        return frames


    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield self._sample()

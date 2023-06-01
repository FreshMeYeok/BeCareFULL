import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
from torchvision.transforms import ToPILImage
import numpy as np


class CustomDataset(Dataset):

    def __init__(self, metadata_file, frames_folder, transform=None, timesteps=4):
        self.transform = transform
        self.metadata_file = metadata_file
        self.frames_folder = frames_folder
        self.timesteps = timesteps
        self.data = []

        # Load metadata from the JSON file
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        len_meta = len(metadata)
        for k, obj in enumerate(metadata):
            if k == len_meta//3:
                break
            obj_name = obj
            num_frames = metadata[obj]['num_frames']
            anomaly_start = metadata[obj]['anomaly_start']
            anomaly_end = metadata[obj]['anomaly_end']

        # 이미지 프레임 불러오기

            sequence_length = 4

            for frame_idx in range(num_frames - sequence_length+1):
                if anomaly_start <= frame_idx <= anomaly_end:
                    break
                frame_paths = []
                for step in range(sequence_length):
                    frame_path = f'{frames_folder}/{obj_name}/images/' + '{:06d}'.format(frame_idx+step) + '.jpg'
                    # frame = Image.open(frame_path).convert("RGB")
                    frame_paths.append(frame_path)
                    # print("수행중")
                is_anomaly = 1 if anomaly_start <= frame_idx+sequence_length <= anomaly_end else 0
                self.data.append((frame_paths, is_anomaly))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_paths, is_anomaly = self.data[idx]
        frame_imgs = []

        # for frame_path in frame_paths:
        #     # frame_img = torch.FloatTensor(Image.open(frame_path).convert("RGB"), is_anomaly)
        #     frame_img = Image.open(frame_path).convert("RGB")
        #     frame_imgs.append(frame_img)
        for frame_path in frame_paths:
            frame_img = Image.open(frame_path).convert("RGB")
            # if self.transform:
            #     frame_img = self.transform(frame_img)
            frame_imgs.append(frame_img)
        # frame_imgs = self.transform(frame_imgs)
        # concatenated_image = torch.stack([transforms.ToTensor()(image) for image in frame_imgs], dim=0)
        concatenated_image = torch.stack([self.transform(image) for image in frame_imgs], dim=0)
        # concatenated_image = torch.cat(frame_imgs, dim=0)
        # is_anomaly = torch.FloatTensor(is_anomaly)
        return concatenated_image, is_anomaly



        # frames = [self.transform(frame) for frame in self.x_data[idx]]
        # is_anomaly = torch.tensor(self.y_data[idx])
        #
        # return frames, is_anomaly

        # # Load and transform the image
        # image = Image.open(frame)
        # if self.transform:
        #     image = self.transform(image)
        #
        # return image, is_anomaly
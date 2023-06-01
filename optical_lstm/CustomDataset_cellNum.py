import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import cv2
import numpy as np


class CustomDataset(Dataset):

    def __init__(self, metadata_file, frames_folder, transform=None):
        """
        :param metadata_file: anomaly start와 end부분을 추출하기 위해 json파일 경로
        :param frames_folder: image를 불러오기위해 경로를 지정해줌
        :param transform: image를 그대로 사용할 순 없으니 transform을 하여 크기를 줄이거나 tensor형태로 만드는 등 수정
        :var data: 각 video마다 2개의 frame을 하나로 묶어 저장
        """

        self.metadata_file = metadata_file
        self.frames_folder = frames_folder
        self.transform = transform
        self.data = []

        # JSON file 읽어오기
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)

        split_num = 50
        len_metadata = len(metadata)
        print("metadata 갯수")
        print(len_metadata//split_num)
        # 읽어온 JSON file로 데이터 전처리
        for k, obj in enumerate(metadata):
            if k == len_metadata//split_num:
                break
            obj_name = obj
            num_frames = metadata[obj]['num_frames']
            anomaly_start = metadata[obj]['anomaly_start']
            anomaly_end = metadata[obj]['anomaly_end']
            # Optical Flow를 계산해야해서 Image 2개씩 묶음
            read_image_count = 7
            if(anomaly_start < read_image_count):
                continue



            # frame_path_list = []
            is_anomaly_list = []

            for frame_idx in range(anomaly_start - anomaly_start//2, anomaly_end - read_image_count):
                frame_path_list = []
                # two_frame = []
                for step in range(read_image_count):
                    frame_path = f'{frames_folder}/{obj_name}/images/' + '{:06d}'.format(frame_idx+step) + '.jpg'
                    # two_frame.append(frame_path)
                    frame_path_list.append(frame_path)
                is_anomaly = 1 if anomaly_start <= frame_idx+read_image_count < anomaly_end else 0
                # frame_path_list.append(two_frame)
                is_anomaly_list.append(is_anomaly)

                self.data.append((frame_path_list, is_anomaly))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_paths, is_anomaly = self.data[idx]
        frame_img_list = []

        # for frame_path in frame_paths:
        #     # frame_img = Image.open(frame_path).convert("RGB")
        #     frame_img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        #     frame_img_list.append(frame_img)

        sequence_len = 7
        frame_imgs = []
        flows = []
        results = []
        for i in range(sequence_len):
            frame_imgs.append(cv2.imread(frame_paths[i], cv2.IMREAD_GRAYSCALE))

        for i in range(sequence_len - 1):
            flows.append(cv2.calcOpticalFlowFarneback(frame_imgs[i], frame_imgs[i+1], None, 0.5, 3, 15, 3, 5, 1.1, 0))

        for i in range(sequence_len-1):
            mag, ang = flows[i][:, :, 0], flows[i][:,:,1]
            mag = self.transform(Image.fromarray(mag, mode='L'))
            ang = self.transform(Image.fromarray(ang, mode='L'))
            result = torch.cat([mag, ang], dim = 0)
            result = torch.flatten(result, 1)
            results.append(result)

        x = torch.cat(results, dim=0)

        return x, is_anomaly

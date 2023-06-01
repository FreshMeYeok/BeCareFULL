import torch
from OpticalLSTM import OpticalLSTM
from PIL import Image
import numpy as np
import os
import cv2
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((180, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.229]),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO 이 부분은 사용가능
def load_dataset(frames_folder, read_frames):
    """
    :param frames_folder: 사고 영상 이미지들이 있는 경로
    :param read_frames: 몇 개의 프레임을 한번에 읽을 지 정하는거
    :return: data : 한 영상에 read_frames개수 만큼 이미지가 들어있는 리스트 반환
    """

    data = []

    files = os.listdir(frames_folder)
    num_frames = len(files)

    for frame_idx in range(num_frames - read_frames):
        frame_path_list = []
        for step in range(read_frames):
            frame_path = frames_folder + '%06d.jpg' % (frame_idx + step)
            frame_path_list.append(frame_path)

        data.append(frame_path_list)

    return data

def main():
    # TODO 이 부분은 테스트 해보고 싶은 영상 위치
    frames_folder = '/home/dnc/Desktop/mnt/workspace/datasets/DoTA_dataset/frames/0RJPQ_97dcs_000307/images/'

    # 하나의 사고 영상을 가져온다. read_frames만큼 분할해서
    # TODO read_frames 같은 경우 각자 모델에서 학습할때 몇 프레임을 읽었는지에 따라 변함
    read_frames = 10
    val_dataset = load_dataset(frames_folder, read_frames)

    # 기존에 있던 모델 가져오기
    # TODO 이 이부분 각자 모델에 맞게 고치셈 #
    # TODO -------- 여기부터 ---------- #
    model_path = '/home/dnc/Desktop/test/load_data/output/model_epoch6.pt'
    hidden_size = 128
    num_layers = 2
    model = OpticalLSTM(180*320, hidden_size, num_layers)
    # TODO -------- 여기까지 ---------- #
    pretrained_dict = torch.load(model_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)

    # Load the modified state_dict into the model
    model.load_state_dict(model_dict)
    model.to(device)

    print(model)

    # 검증 단계
    model.eval()
    output_list = [] # 결과 값 저장
    output_images = [] # 이미지 저장
    with torch.no_grad():
        for data in val_dataset:
            frame_imgs = []
            flows = []
            results = []
            for i in range(read_frames):
                frame_imgs.append(cv2.imread(data[i], cv2.IMREAD_GRAYSCALE))
                if i == read_frames-1:
                    output_img = cv2.imread(data[i])
                    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                    output_images.append(output_img)

            for i in range(read_frames - 1):
                flows.append(
                    cv2.calcOpticalFlowFarneback(frame_imgs[i], frame_imgs[i + 1], None, 0.5, 3, 15, 3, 5, 1.1, 0))

            for i in range(read_frames - 1):
                mag, ang = flows[i][:, :, 0], flows[i][:, :, 1]
                mag = transform(Image.fromarray(mag, mode='L'))
                ang = transform(Image.fromarray(ang, mode='L'))
                result = torch.cat([mag, ang], dim=0)
                result = torch.flatten(result, 1)
                results.append(result)

            x = torch.cat(results, dim=0)
            x = x.unsqueeze(0)

            output_list.append(model(x))

        print('hi') # 이건 디버깅용

if __name__ == '__main__':
    main()

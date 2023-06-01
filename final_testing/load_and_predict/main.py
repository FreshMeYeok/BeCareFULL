import torch
from R2plus1D import R2plus1D
from PIL import Image
import numpy as np
import torch.nn.functional as F
import argparse
import os
import cv2
from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder

from flask import Flask
from flask_socketio import SocketIO
import random
import threading
from flask import Flask, request, send_from_directory
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
import numpy as np
from flask import jsonify
import json



app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*')
UPLOAD_FOLDER = "./uploads"  # 동영상 파일을 저장할 폴더 경로
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO 2개 함수 load_image랑 extract_optical_flow는
# TODO RAFT라는 Optical Flow 구해주는거 사용해서 추가적으로 붙었음
# TODO RAFT사용 안했으면 밑에 두 함수는 필요없음
# def load_image(imfile):
#     img = np.array(Image.open(imfile)).astype(np.uint8)
#     img = torch.from_numpy(img).permute(2, 0, 1).float()
#     return img[None].to(device)

def extract_optical_flow(frame1, frame2, args):
    """
    :param frame1: RGB Image1
    :param frame2: RGB Image2
    :return: 두 이미지 간의 Optical Flow
    """

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(device)
    model.eval()

    with torch.no_grad():
        # frame1 = load_image(frame1)
        # frame2 = load_image(frame2)

        padder = InputPadder(frame1.shape)
        frame1, frame2 = padder.pad(frame1, frame2)

        flow_low, flow_up = model(frame1, frame2, iters=20, test_mode=True)

    return flow_up

# TODO 이 부분은 사용가능
# def load_dataset(frames_folder, read_frames):
#     """
#     :param frames_folder: 사고 영상 이미지들이 있는 경로
#     :param read_frames: 몇 개의 프레임을 한번에 읽을 지 정하는거
#     :return: data : 한 영상에 read_frames개수 만큼 이미지가 들어있는 리스트 반환
#     """
#
#     data = []
#
#     files = os.listdir(frames_folder)
#     num_frames = len(files)
#
#     for frame_idx in range(num_frames - read_frames):
#         frame_path_list = []
#         for step in range(read_frames):
#             frame_path = frames_folder + '%06d.jpg' % (frame_idx + step)
#             frame_path_list.append(frame_path)
#
#         data.append(frame_path_list)
#
#     return data

def video2frame(video):
    cap = cv2.VideoCapture(video)

    frame_list = []

    while cap.isOpened():
        # 프레임 읽기
        ret, frame = cap.read()

        # 비디오가 끝나면 종료
        if not ret:
            break

        # 각 프레임의 이미지를 RAFT 형식에 맞게 변환
        img = np.array(frame).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = img[None].to(device)

        frame_list.append(img)
    # 비디오 캡처 객체 해제
    cap.release()

    return frame_list


def calc_anomaly(file_name):
    # TODO parser부분도 RAFT사용하려고 있는거임 안썼으면 무시하셈
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default="../RAFT/models/raft-kitti.pth")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    # TODO 이 부분은 테스트 해보고 싶은 영상 위치
    frames_folder = '/home/mot/DOTA/frames/0RJPQ_97dcs_000307/images/'

    # tmp = video2frame('/home/mot/DOTA/video/test1.mp4')

    # 하나의 사고 영상을 가져온다. read_frames만큼 분할해서
    # TODO read_frames 같은 경우 각자 모델에서 학습할때 몇 프레임을 읽었는지에 따라 변함
    read_frames = 4
    # val_dataset = load_dataset(frames_folder, read_frames)
    val_dataset = video2frame(f'./video/{file_name}')

    # 기존에 있던 모델 가져오기
    # TODO 이 이부분 각자 모델에 맞게 고치셈 #
    # TODO -------- 여기부터 ---------- #
    model_path = './output/first_model/model_epoch5.pt'
    hidden_size = 256
    num_layers = 2
    model = R2plus1D(270*480, hidden_size, num_layers)
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
    output_list = []
    for i in range(read_frames-1):
        output_list.append(0.0)

    with torch.no_grad():
        for n, data in enumerate(val_dataset):
            # TODO 이 부분은 CustomDataset에서 __getitem__부분을 가져옴 각자 모델 입맛에 맞게 작성
            optical_flow_list = []
            # for i in range(len(data)-1):
            if n > len(val_dataset) - read_frames - 1:
                continue
            for i in range(n, n + read_frames):
                optical_flow = extract_optical_flow(val_dataset[i], val_dataset[i+1], args)
                optical_flow = F.interpolate(optical_flow, size=(270, 480), mode='bilinear')
                optical_flow_list.append(optical_flow)

            result = torch.cat([optical_flow_list[0], optical_flow_list[1]], dim=1)

            for i in range(2, len(optical_flow_list)):
                result = torch.cat([result, optical_flow_list[i]], dim=1)
            # TODO 여기까지

            output_list.append(model(result).cpu().numpy()[0][1])

        # print('hi') # 이건 디버깅용
    print(output_list)

    return output_list






# Flask Server 파트
@socketio.on('connect')
def handle_connect():
    print('클라이언트가 연결되었습니다!')


def send_anomaly_probabilities(anomaly_probabilities):
    socketio.emit('anomaly_probabilities', anomaly_probabilities)
    print(anomaly_probabilities)


# 매 초마다 랜덤한 숫자를 보내는 함수
# # @socketio.on('connect')
# def send_random_number():
#     anomaly_value = np.sin(np.arange(100))
#     # for i in range(len(anomaly_value)):
#     while True:
#         number = random.randint(1, 10)  # 1부터 100까지의 랜덤한 숫자 생성
#         # 클라이언트에게 'random_number' 이벤트와 함께 숫자 전송
#
#         """Model 삽입
#         h x w x c
#         model.predict(data = images)
#
#
#
#         """
#         # number = anomaly_value[i]
#         print(number)
#         socketio.emit('random_number', number)
#         socketio.sleep(0.1)  # 1초 대기


# Flask 애플리케이션 실행 시 스레드를 시작하는 함수


def start_thread():
    socketio.start_background_task(send_random_number)


def anomaly_value():
    anomaly_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return anomaly_value


@app.route("/upload", methods=["POST"])
def upload():
    if "video" in request.files:
        video = request.files["video"]
        filename = secure_filename(video.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        video.save(filepath)

        # 모델을 통과하여 각 초에서의 이상탐지 확률 리스트를 얻어옴
        # anomaly_probabilities = model.predict(video_path)
        print("anomaly_probabilities 계산 중")
        anomaly_probabilities = calc_anomaly(filename)

        # anomaly_probabilities = [1,2,3,4,5]

        # float_anomaly_list = np.array([int(float(value)*100) for value in anomaly_probabilities])
        # max_anomaly = np.max(float_anomaly_list)
        # min_anomaly = np.min(float_anomaly_list)
        # float_anomaly_list = (float_anomaly_list-min_anomaly)/(max_anomaly-min_anomaly) * 100
        # re_list = []
        # for data in float_anomaly_list:
        #     re_list.append(int(data))
        re_list = [int(float(value)*100) for value in anomaly_probabilities]
        json_anomaly = json.dumps(re_list)

        print("anomaly_probabilities 계산 완료")
        # 각 초에서의 이상탐지 확률 리스트를 클라이언트에게 전송
        # for probability in anomaly_probabilities:
        #     print("출력중")
        #     send_anomaly_probabilities(probability)
        #     socketio.sleep(0.1)

        # 업로드한 영상의 URL 생성
        video_url = f"http://localhost:8888/download/{filename}"
        return jsonify({"url": video_url, "anomaly_probabilities": json_anomaly})

    return "No video file provided.", 400


@app.route("/download/<filename>", methods=["GET"])
def download(filename):
    print("전송 시작")
    print("전송 완료")
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, mimetype='video/mp4')















if __name__ == '__main__':
    print("실행")
    # start_thread()
    socketio.run(app, host='0.0.0.0', port=8888, allow_unsafe_werkzeug=True)

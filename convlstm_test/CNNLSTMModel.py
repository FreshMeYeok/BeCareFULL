# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
#
# class CNNLSTMModel(nn.Module):
#     def __init__(self, hidden_size, num_layers):
#         super(CNNLSTMModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         # self.lstm = nn.LSTM(32 * 320 * 180, hidden_size, num_layers, batch_first=True)
#         # self.fc = nn.Linear(hidden_size, 2)  # 2는 사고 여부를 나타내는 클래스 수입니다.
#
#         # LSTM layers
#         self.lstm = nn.LSTM(32 * 160 * 90, hidden_size, num_layers, batch_first=True)
#
#         # Fully connected layers
#         # self.fc = nn.Sequential(
#         #     nn.Linear(hidden_size, 2),
#         #     nn.Softmax(dim=1)
#         # )
#
#     def forward(self, x):
#         # print(x)
#         # print(x.size())
#         batch_size, timesteps, C, H, W = x.size()
#         x = x.view(batch_size * timesteps, C, H, W)
#         cnn_output = self.cnn(x)
#         cnn_output = cnn_output.view(batch_size, timesteps, -1)
#         lstm_output, _ = self.lstm(cnn_output)
#         lstm_output = lstm_output[:, -1, :]
#         output = self.fc(lstm_output)
#         return output
#
# # 모델 인스턴스 생성
# # hidden_size = 256
# # num_layers = 2
# # model = CNNLSTMModel(hidden_size, num_layers)
#
# # hidden_size = 128
# # num_layers = 1
# # model = CNNLSTMModel(hidden_size, num_layers)
#
# # 예측 수행
# # input_frames = torch.randn(1, 4, 3, 1280, 720)  # 입력 영상의 프레임
# # output = model(input_frames)

import torch
import torch.nn as nn
import torchvision.models as models

class CNNLSTMModel(nn.Module):
    def __init__(self, hidden_size, num_classes, timesteps=4, pretrained=True):
        super(CNNLSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.timesteps = timesteps

        self.cnn = models.resnet18(pretrained=pretrained)
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(num_ftrs, 512)
        self.cnn.fc = nn.Linear(512, 256)

        # Temporal pooling layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # LSTM layers
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=2, batch_first=True)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)

        cnn_output = self.cnn(x)
        cnn_output = cnn_output.view(batch_size, timesteps, -1)

        lstm_output, _ = self.lstm(cnn_output)
        lstm_output = lstm_output[:, -1, :]

        output = self.fc(lstm_output)
        return output
import torch
import torch.nn as nn
import torchvision.models as models

class OpticalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(OpticalLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.layer_norm1 = nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=3, batch_first=True, bidirectional=True)
        self.fc2 = nn.Linear(hidden_size*2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.layer_norm2(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Get the output of the last time step
        x = self.fc2(x)
        x = self.softmax(x)
        return x

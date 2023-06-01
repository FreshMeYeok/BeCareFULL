import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.video as video

class R2plus1D(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(R2plus1D, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # self.raft = video.
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, dropout=0.25, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        batch_size, sequence_length, rows, cols = x.shape
        x = x.view(batch_size, sequence_length, -1)

        lstm_output, _ = self.lstm(x)
        lstm_output = lstm_output[:,-1,:]
        output = self.fc(lstm_output)
        output = self.softmax(output)

        return output

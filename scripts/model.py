import torch.nn as nn, torch

class WakewordRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(input_size=40, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x og shape -> batch, n_mels, time
        x = x.transpose(1, 2) # x new shape -> batch, time, n_mels
        _, h = self.rnn(x)
        out = self.fc(h[-1])
        return torch.sigmoid(out)
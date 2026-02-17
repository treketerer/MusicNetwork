import torch
import torch.nn as nn
from torch import tensor

class ConductorLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size:int, output_size: int):
        super(ConductorLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layer_dim = 2

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.layer_dim,
            batch_first=True,
            dropout=0.2
        )

        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_vector, h0=None, c0=None):
        if h0 is None and c0 is None:
            out, (hn, cn) = self.lstm(input_vector)
        else:
            out, (hn, cn) = self.lstm(input_vector, (h0, c0))

        logits = self.linear(out)
        return logits, hn, cn
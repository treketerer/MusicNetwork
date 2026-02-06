import torch
import torch.nn as nn
from torch import tensor

class ConductorLSTM(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(ConductorLSTM, self).__init__()

        self.output_size = output_size
        self.hidden_state = input_size
        self.layer_dim = 3

        self.lstm = nn.LSTM(
            input_size=self.emb_length,
            hidden_size=self.hidden_state,
            num_layers=self.layer_dim,
            batch_first=True,
            dropout=0.2
        )
        self.linear = nn.Linear(self.hidden_state, self.output_size)

    def forward(self, input_vector, h0=None, c0=None):
        if h0 is None and c0 is None:
            h0 = input_vector.repeat(self.layer_dim, 1, 1).contiguous()
            c0 = input_vector.repeat(self.layer_dim, 1, 1).contiguous()

        out, (hn, cn) = self.lstm(input_vector, (h0, c0))
        logits = self.linear(out)
        return logits, hn, cn
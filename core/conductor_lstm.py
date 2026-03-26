import torch
import torch.nn as nn
from torch import tensor

class ConductorLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size:int, output_size: int):
        super(ConductorLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layer_dim = 2

        self.fusion_linear = nn.Linear(input_size, self.hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=self.layer_dim,
            batch_first=True,
            dropout=0.15
        )

        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, vibe_vector, instruments_vector, backloop_vectors, h0=None, c0=None):
        constant_vector = torch.cat((vibe_vector, instruments_vector), dim=1)

        # Получение всех векторов для инструментов
        if backloop_vectors.size(1) > 1:
            zeros = torch.zeros_like(backloop_vectors[:, :1, :])
            backloop_input = torch.cat((zeros, backloop_vectors[:, :-1, :]), dim=1)
        else:
            backloop_input = backloop_vectors

        constant_vector = constant_vector.unsqueeze(1).expand(-1, backloop_input.size(dim=1), -1)
        concatenated_vectors = torch.cat((constant_vector, backloop_input), dim=2)

        compressed_vector = self.fusion_linear(concatenated_vectors)

        if h0 is None and c0 is None:
            out, (hn, cn) = self.lstm(compressed_vector)
        else:
            out, (hn, cn) = self.lstm(compressed_vector, (h0, c0))

        logits = self.linear(out)
        return logits, hn, cn
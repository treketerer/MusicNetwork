import torch
import torch.nn as nn
from torch import tensor

class DecoderLSTM(nn.Module):
    def __init__(self, input_size: int, alphabet_size: int):
        super(DecoderLSTM, self).__init__()

        self.alphabet_size = alphabet_size

        self.emb_length = 256
        self.midi_embeddings = nn.Embedding(alphabet_size, self.emb_length)

        self.hidden_state = input_size
        self.layer_dim = 1
        self.lstm = nn.LSTM(
            input_size=self.emb_length,
            hidden_size=self.hidden_state,
            num_layers=self.layer_dim,
            batch_first=True
        )
        self.linear = nn.Linear(self.hidden_state, self.alphabet_size)

    def forward(self, context, midis_idx, h0=None, c0=None):

        if h0 is None and c0 is None:
            h0 = context.unsqueeze(0).contiguous() #torch.zeros(self.layer_dim, context.size(0), self.hidden_state)
            c0 = context.unsqueeze(0).contiguous() #torch.zeros(self.layer_dim, context.size(0), self.hidden_state)

        x = self.midi_embeddings(midis_idx)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        logits = self.linear(out)
        return logits, hn, cn
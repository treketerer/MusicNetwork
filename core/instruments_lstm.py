import torch
import torch.nn as nn
from torch import tensor

class InstrumentsLSTM(nn.Module):
    def __init__(self, conductor_context_size: int, instruments_count: int, instruments_emb_size: int, midi_emb_size: int):
        super(InstrumentsLSTM, self).__init__()

        self.instruments_count = instruments_count

        self.midi_embeddings = nn.Embedding(self.instruments_count, midi_emb_size)

        self.input_size = conductor_context_size + instruments_emb_size + midi_emb_size
        self.hidden_state = self.input_size
        self.layer_dim = 3
        self.lstm = nn.LSTM(
            input_size=self.hidden_state,
            hidden_size=self.hidden_state,
            num_layers=self.layer_dim,
            batch_first=True,
            dropout=0.2
        )

        self.midi_out = nn.Sequential(
            nn.Linear(self.hidden_state, self.instruments_count),
            nn.Sigmoid()
        )

    def forward(self, conductor_context, instruments_emb, last_note_idx, h0=None, c0=None):
        last_note_emb = self.midi_embeddings(last_note_idx)
        input_embedding = torch.cat([conductor_context, instruments_emb, last_note_emb], dim=-1)

        if h0 is None and c0 is None:
            h0 = input_embedding.repeat(self.layer_dim, 1, 1).contiguous() #torch.zeros(self.layer_dim, context.size(0), self.hidden_state)
            c0 = input_embedding.repeat(self.layer_dim, 1, 1).contiguous() #torch.zeros(self.layer_dim, context.size(0), self.hidden_state)

        out, (hn, cn) = self.lstm(input_embedding, (h0, c0))
        logits = self.midi_out(out)
        return logits, hn, cn
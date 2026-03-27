import torch
import torch.nn as nn
from torch.nn import Embedding

class SongInstrumentsLinearParser(nn.Module):
    def __init__(self, input_dim: int, inner_dim: int, output_dim: int, instruments_embeddings: Embedding):
        super(SongInstrumentsLinearParser, self).__init__()

        self.instruments_embeddings = instruments_embeddings
        self.input_dim = input_dim
        self.parser = nn.Sequential(
            nn.Linear(input_dim, inner_dim),
            nn.ReLU(),
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, output_dim)
        )

    def forward(self, instruments_idx):
        instruments_emb = self.instruments_embeddings(instruments_idx)
        x = self.parser(instruments_emb)
        return x

class ConductorInstrumentsParser(nn.Module):
    def __init__(self, h_dim: int, inner_dim: int, instruments_count:int):
        super(ConductorInstrumentsParser, self).__init__()

        self.instruments_count = instruments_count
        self.parser = nn.Sequential(
            nn.Linear(h_dim, inner_dim),
            nn.ReLU(),
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, self.instruments_count),
        )

    def forward(self, conductor_h):
        x = self.parser(conductor_h)
        return x

class BackloopEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(BackloopEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )

        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, notes_emb):
        bd, td, ind, nd, ed = notes_emb.shape

        flat_notes = notes_emb.view(bd * td * ind, nd, ed)
        _, h = self.gru(flat_notes)

        h_last_layer = h[-1]
        h_reshaped = h_last_layer.view(bd, td, ind, -1)

        tact_context = h_reshaped.mean(dim=2)
        x = self.linear(tact_context)
        return x  # [bd, td, output_dim]
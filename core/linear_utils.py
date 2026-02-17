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
            nn.Sigmoid()
        )

    def forward(self, conductor_h):
        x = self.parser(conductor_h)
        return x
    
class BackloopLinearEncoder(nn.Module):
    def __init__(self, input_dim: int, inner_dim: int, output_dim: int):
        super(BackloopLinearEncoder, self).__init__()

        self.parser =  nn.Sequential(
            nn.Linear(input_dim, inner_dim),
            nn.ReLU(),
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, output_dim)
        )

    def forward(self, notes_sum_emb):
        x = self.parser(notes_sum_emb)
        # print(x.shape)
        return x.sum(2)
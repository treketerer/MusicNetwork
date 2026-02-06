import torch
import torch.nn as nn

class InstrumentsMultiHotLinearParser(nn.Module):
    def __init__(self, input_dim: int, inner_dim: int, output_dim: int):
        super(InstrumentsMultiHotLinearParser, self).__init__()

        self.input_dim = input_dim
        self.parser = nn.Sequential(
            nn.Linear(input_dim, inner_dim),
            nn.ReLU(),
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, output_dim)
        )

    def forward(self, instruments_idx):
        zeros = torch.zeros(self.input_dim)
        multi_hot = zeros.scatter_(0, instruments_idx, 1)
        x = self.parser(multi_hot)
        return x

class ConductorInstrumentsParser(nn.Module):
    def __init__(self, instruments_count:int, h_dim: int, inner_dim: int, output_dim: int):
        super(ConductorInstrumentsParser, self).__init__()

        self.instruments_count = instruments_count
        self.parser = nn.Sequential(
            nn.Linear(h_dim, inner_dim),
            nn.Sigmoid(),
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, output_dim),
            nn.Sigmoid(),
            nn.Threshold(0.3, 1)
        )

    def forward(self, conductor_h):
        x = self.parser(conductor_h)
        multi_hot = torch.zeros(self.instruments_count).scatter_(0, x, 1)
        return multi_hot
    
class BackloopLinearEncoder(nn.Module):
    def __init__(self, input_dim: int, inner_dim: int, output_dim: int):
        super(BackloopLinearEncoder, self).__init__()

        self.parser =  nn.Sequential(
            nn.Linear(input_dim, inner_dim),
            nn.ReLU(),
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, output_dim)
        )

    def forward(self, sums_h):
        x = self.parser(sums_h)
        return x
import torch
import torch.nn as nn
from torch import tensor

class TextEncoderGRU(nn.Module):
    def __init__(self, alphabet_size: int, text_emb_dim:int, inner_context_size: int, output_dim: int):
        super(TextEncoderGRU, self).__init__()

        self.embeddings_layer = nn.Embedding(
            alphabet_size,
            text_emb_dim,
            padding_idx=0
        )

        self.gru = nn.GRU(
            input_size=text_emb_dim,
            hidden_size=inner_context_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.linear = nn.Sequential(
            nn.Linear(inner_context_size, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        x = self.embeddings_layer(x)
        _, h = self.gru(x)
        x = self.linear(h[-1])
        return x
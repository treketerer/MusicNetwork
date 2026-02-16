import torch
import torch.nn as nn
from torch import tensor

class EncoderLinear(nn.Module):
    def __init__(self, alphabet_size: int, text_emb_dim:int, inner_context_size: int, output_dim: int):
        super(EncoderLinear, self).__init__()

        self.embeddings_layer = nn.Embedding(
            alphabet_size,
            text_emb_dim
        )

        self.encoder = nn.Sequential(
            nn.Linear(text_emb_dim, inner_context_size),
            nn.ReLU(),
            nn.Linear(inner_context_size, inner_context_size),
            nn.ReLU(),
            nn.LayerNorm(inner_context_size),
            nn.Linear(inner_context_size, output_dim),
        )

    def forward(self, x):
        x = self.embeddings_layer(x)
        x = x.view(x.size(0), -1)

        x = self.encoder(x)
        return x
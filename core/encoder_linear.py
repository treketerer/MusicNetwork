import torch
import torch.nn as nn
from torch import tensor

class EncoderLinear(nn.Module):
    def __init__(self, text_emb_length:int, inner_context_size: int, output_dim: int, alphabet_size: int):
        super(EncoderLinear, self).__init__()

        self.embeddings_layer = nn.Embedding(num_embeddings=alphabet_size, embedding_dim=text_emb_length)

        self.encoder = nn.Sequential(
            nn.Linear(text_emb_length, inner_context_size),
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
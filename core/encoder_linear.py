import torch
import torch.nn as nn
from torch import tensor

class EncoderLinear(nn.Module):
    def __init__(self, inner_context_size: int, alphabet_size: int):
        super(EncoderLinear, self).__init__()

        self.alphabet_emb = None
        self.alphabet_words = None

        self.middle_dim = 1024
        self.emb_length = 512
        self.output_length = inner_context_size

        self.alphabet_size = alphabet_size

        self.embeddings_layer = nn.Embedding(num_embeddings=self.alphabet_size, embedding_dim=self.emb_length)

        self.encoder = nn.Sequential(
            nn.Linear(self.emb_length, self.middle_dim),
            nn.ReLU(),
            nn.Linear(self.middle_dim, self.middle_dim),
            nn.ReLU(),
            nn.LayerNorm(self.middle_dim),
            nn.Linear(self.middle_dim, self.output_length),
        )

    def forward(self, x):
        x = self.embeddings_layer(x)
        x = x.view(x.size(0), -1)

        x = self.encoder(x)
        return x
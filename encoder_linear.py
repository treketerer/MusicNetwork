import torch
import torch.nn as nn
from torch import tensor

class EncoderLinear(nn.Module):
    def __init__(self, inner_context_size: int, alphabet_size: int):
        super(EncoderLinear, self).__init__()

        self.alphabet_emb = None
        self.alphabet_words = None

        self.emb_length = 256
        self.max_input_emb_length = 10
        self.output_length = inner_context_size

        self.alphabet_size = alphabet_size

        self.embeddings_layer = nn.Embedding(num_embeddings=self.alphabet_size, embedding_dim=self.emb_length)
        self.fc = nn.Linear(self.emb_length, self.output_length)

    def forward(self, x):
        input_emb = self.embeddings_layer(x)
        pooled = torch.mean(input_emb, dim=1)

        x = torch.relu(self.fc(pooled))
        return x
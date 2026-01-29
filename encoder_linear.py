import torch
import torch.nn as nn
from torch import tensor

class EncoderLinear(nn.Module):
    def __init__(self, inner_context_size: int, alphabet_size: int):
        super(EncoderLinear, self).__init__()

        self.alphabet_emb = None
        self.alphabet_words = None

        self.middle_count = 512
        self.emb_length = 128
        self.max_prompt_len = 7
        self.output_length = inner_context_size

        self.alphabet_size = alphabet_size

        self.embeddings_layer = nn.Embedding(num_embeddings=self.alphabet_size, embedding_dim=self.emb_length)
        self.fc1 = nn.Linear(self.emb_length * self.max_prompt_len, self.middle_count)
        self.fc2 = nn.Linear(self.middle_count, self.output_length)

    def forward(self, x):
        if x.size(1) < self.max_prompt_len:
            pad = torch.zeros(x.size(0), self.max_prompt_len - x.size(1), dtype=torch.long, device=x.device)
            x = torch.cat([x, pad], dim=1)
        else:
            x = x[:, :self.max_prompt_len]
        input_emb = self.embeddings_layer(x)
        flat_embeds = input_emb.view(input_emb.size(0), -1)

        x = torch.relu(self.fc1(flat_embeds))
        x = self.fc2(x)
        return x
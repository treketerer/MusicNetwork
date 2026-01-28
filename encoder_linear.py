import torch
import torch.nn as nn
from torch import tensor
from learndata import data as learn_data

class EncoderLinear(nn.Module):
    def __init__(self, inner_context_size, batch_size = 1):
        super(EncoderLinear, self).__init__()

        self.alphabet_emb = None
        self.alphabet_words = None

        self.emb_length = 64
        self.max_input_emb_length = 10
        self.output_length = 256

        self.batch_size = inner_context_size

        self.init_embeddings(learn_data)

        self.fc1 = nn.Linear(self.max_input_emb_length * self.emb_length, 50)
        self.fc2 = nn.Linear(50, self.output_length)

    def parse_learn_couple(self, input_string):
        split_string = input_string.split()

        inp_part = split_string[:-1]
        learn_word = split_string[-1]

        inp_len = len(inp_part)

        if inp_len > self.max_input_emp_length:
            return ' '. join(inp_part[-5:]), learn_word

        for i in range(self.max_input_emp_length - inp_len):
            inp_part.append('<unk>')
        return ' '.join(inp_part), learn_word

    def parse_words_idx(self, x):
        inp_idx = []
        for word in x.split():
            if word in self.alphabet_words:
                inp_idx.append(self.alphabet_words.index(word))
            else:
                inp_idx.append(self.alphabet_words.index("<unk>"))

        return tensor(inp_idx)

    def forward(self, x):
        input_emb = self.alphabet_emb(x)
        x_batch = input_emb.view(x.size(0), -1)

        x = torch.relu(self.fc1(x_batch))
        x = self.fc2(x)
        return x
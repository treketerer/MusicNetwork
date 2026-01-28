import torch
import torch.nn as nn
from torch import tensor
from learndata import data as learn_data

class DecoderLSTM(nn.Module):
    def __init__(self, input_size, batch_size = 1):
        super(DecoderLSTM, self).__init__()

        self.midi_alphabet_words = None

        self.batch_size = batch_size

        self.output_length = -1
        self.hidden_state = 100
        self.layer_dim = 1
        self.lstm = nn.LSTM(input_size, self.hidden_state, self.layer_dim, True)
        self.linear = nn.Linear(self.hidden_state, self.output_length)

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

    def forward(self, x, h0=None, c0=None):
        if h0 is None and c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_state)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_state)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out, hn, cn
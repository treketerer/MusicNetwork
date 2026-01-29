import torch
import torch.nn as nn

from encoder_linear import EncoderLinear
from decoder_lstm import DecoderLSTM

class MusicNN(nn.Module):
    def __init__(self, text_alphabet_size, midi_alphabet_size):
        super(MusicNN, self).__init__()

        self.inner_context_size = 256

        self.encoder_model = EncoderLinear(self.inner_context_size, text_alphabet_size)
        self.decoder_model = DecoderLSTM(self.inner_context_size, midi_alphabet_size)

    def forward(self, prompt_idx, midis_idx, h0 = None, c0 = None):
        context = self.encoder_model(prompt_idx)
        answer, hn, cn = self.decoder_model(context, midis_idx, h0, c0)

        return answer, hn, cn
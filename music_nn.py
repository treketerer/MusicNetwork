import torch
import torch.nn as nn

from encoder_linear import EncoderLinear
from decoder_lstm import DecoderLSTM

class MusicNN(nn.Module):
    def __init__(self, batch_size = 1):
        super(MusicNN, self).__init__()

        self.inner_context_size = 256

        self.encoder_model = EncoderLinear(self.inner_context_size, batch_size)
        self.decoder_model = DecoderLSTM(self.inner_context_size, batch_size)

        self.h0 = None
        self.c0 = None
        self.raw_h0 = None
        self.raw_c0 = None

    def forward(self, x):
        context = self.encoder_model(x)
        answer, h0, c0 = self.decoder_model(context)

        self.raw_h0 = h0
        self.raw_c0 = c0

        return answer

    def bake_memory(self):
        self.h0, self.c0 = self.raw_h0.detach(), self.raw_c0.detach()
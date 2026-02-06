import torch
import torch.nn as nn

from .encoder_linear import EncoderLinear
from .conductor_lstm import ConductorLSTM
from .instruments_lstm import InstrumentsLSTM
from .linear_untils import *

class MusicNN(nn.Module):
    def __init__(self, text_alphabet_size, instruments_counts = 128, learning=True):
        super(MusicNN, self).__init__()

        self.backloop_vector = None

        self.instruments_counts = instruments_counts
        self.instruments_embedding_size = 512

        self.midi_embeddings_size = 512
        self.inner_context_size = 1024

        self.instruments_embeddings = nn.Embedding(self.instruments_counts, self.instruments_embedding_size)

        self.encoder_model = EncoderLinear(self.inner_context_size, text_alphabet_size)

        self.conductor_lstm = ConductorLSTM(self.inner_context_size, self.inner_context_size)
        self.instruments_linear_parser = InstrumentsMultiHotLinearParser(self.instruments_counts, self.inner_context_size, self.inner_context_size)
        self.instruments_lstm = InstrumentsLSTM(self.inner_context_size, self.instruments_counts, self.midi_embeddings_size)

        backloop_input_size = self.inner_context_size + self.midi_embeddings_size + self.instruments_counts
        self.backloop_encoder = BackloopLinearEncoder(backloop_input_size, backloop_input_size, int(backloop_input_size/2))

    def learn_nn(self):
        pass

    def use_nn(self):
        pass


    def forward(self, prompt_idx, instruments_idx, midis_idx, h0 = None, c0 = None):


        input_vector = None
        if self.backloop_vector is None:
            instruments_vector = self.instruments_linear_parser(instruments_idx)
            vibe_vector = self.encoder_model(prompt_idx)
            input_vector = torch.cat((vibe_vector, instruments_vector), dim = 1)
            self.backloop_vector = input_vector
        if midis_idx is None:
            dim = prompt_idx.size(-1)
            midis_idx = [1] * dim

        answer, hn, cn = self.conductor_lstm(input_vector, h0, c0)
        instruments_multi_hot = self.instruments_linear_parser(answer)
        inst_weights = torch.sigmoid(instruments_multi_hot)
        instruments_vectors = self.instruments_embeddings.weight * inst_weights.unsqueeze(-1)

        instruments_answer, hn, cn = self.instruments_lstm(answer, instruments_vectors, midis_idx)
        backloop_vec = instruments_answer.sum(dim = 1)
        backloop_vec = torch.nn.functional.normalize(backloop_vec, dim=1)
        self.backloop_vector = self.backloop_encoder(backloop_vec)

        return instruments_answer, hn, cn
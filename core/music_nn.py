import torch
import torch.nn as nn

from .encoder_linear import EncoderLinear
from .conductor_lstm import ConductorLSTM
from .instruments_lstm import InstrumentsLSTM
from .linear_utils import *

class MusicNN(nn.Module):
    def __init__(self, text_alphabet_size, instruments_counts = 128, learning=True):
        super(MusicNN, self).__init__()

        self.is_training = learning
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

    def learn_nn(self, prompt_idx, instruments_idx, all_tacts):
        instruments_vector = self.instruments_linear_parser(instruments_idx)
        vibe_vector = self.encoder_model(prompt_idx)
        input_vector = torch.cat((vibe_vector, instruments_vector), dim=1)

        first_vector = torch.cat((input_vector, torch.zeros(input_vector.size(-1))), dim = 1)

        tacts_hots = torch.oneho
        backloop_vectors = self.backloop_encoder(all_tacts)
        final_vibes = torch.cat((first_vector, backloop_vectors))
        repeated_inp = input_vector.repeat(input_vector.size(-1))
        all_conductors_inp = torch.cat((repeated_inp, final_vibes))

        conductor_h = self.conductor_lstm(all_conductors_inp)
        instruments = self.instruments_linear_parser(conductor_h)

        instruments_conductor_vectors = self.instruments_embeddings.weight * instruments

        empty_note = torch.tensor([2])
        first_notes = all_tacts[:, :-1, :1]
        final_notes = torch.cat((empty_note, first_notes))
        self.instruments_lstm(conductor_h, instruments_conductor_vectors, final_notes)

        return None

    def use_nn(self, prompt_idx, instruments_idx, last_notes, h0, c0):
        input_vector = None
        if self.backloop_vector is None:
            instruments_vector = self.instruments_linear_parser(instruments_idx)
            vibe_vector = self.encoder_model(prompt_idx)
            input_vector = torch.cat((vibe_vector, instruments_vector), dim=1)
            self.backloop_vector = input_vector
        if last_notes is None:
            dim = prompt_idx.size(-1)
            midis_idx = [1] * dim

        answer, hn, cn = self.conductor_lstm(input_vector, h0, c0)
        instruments_multi_hot = self.instruments_linear_parser(answer)
        inst_weights = torch.sigmoid(instruments_multi_hot)
        instruments_vectors = self.instruments_embeddings.weight * inst_weights.unsqueeze(-1)

        instruments_answer, hn, cn = self.instruments_lstm(answer, instruments_vectors, midis_idx)
        backloop_vec = instruments_answer.sum(dim=1)
        backloop_vec = torch.nn.functional.normalize(backloop_vec, dim=1)
        self.backloop_vector = self.backloop_encoder(backloop_vec)

        return instruments_answer, hn, cn


    def forward(self, prompt_idx, instruments_idx, midis_idx, h0 = None, c0 = None):
        if self.is_training:
            learn = self.learn_nn(prompt_idx, instruments_idx, midis_idx)
            return learn
        else:
            user = self.use_nn(prompt_idx, instruments_idx, midis_idx, h0, c0)
            return user

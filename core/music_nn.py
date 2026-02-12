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

        backloop_output_size = int(self.midi_embeddings_size / 2)

        self.instruments_embeddings = nn.Embedding(self.instruments_counts, self.instruments_embedding_size)

        self.encoder_model = EncoderLinear(self.inner_context_size, text_alphabet_size)

        conductor_input_size = self.inner_context_size + self.inner_context_size + backloop_output_size
        self.conductor_lstm = ConductorLSTM(conductor_input_size, self.inner_context_size)
        self.instruments_linear_parser = InstrumentsMultiHotLinearParser(self.instruments_counts + 1, self.inner_context_size, self.inner_context_size)
        self.instruments_lstm = InstrumentsLSTM(self.inner_context_size, self.instruments_counts, self.midi_embeddings_size)

        self.backloop_encoder = BackloopLinearEncoder(self.inner_context_size, self.inner_context_size, backloop_output_size)

    def learn_nn(self, prompt_idx:list, full_instr_list:torch.Tensor, tacts_instr:torch.Tensor, tacts_data:torch.Tensor):
        # Прогон через Backloop для получения векторов в начало
        inst_embs = self.instruments_lstm.midi_embeddings(tacts_data)
        backloop_vectors = self.backloop_encoder(inst_embs.sum(dim=(2, 3)))

        # Получение половины вектора для инструментов
        instruments_vector = self.instruments_linear_parser(full_instr_list)
        vibe_vector = self.encoder_model(prompt_idx)
        constant_vector = torch.cat((vibe_vector, instruments_vector), dim=1)
        f, s = constant_vector.shape

        # Получение всех векторов для инструментов
        zeros = torch.zeros_like(backloop_vectors[:, :1, :])
        backloop_input = torch.cat((zeros, backloop_vectors[:, :-1, :]), dim=1)

        constant_vector_expanded = constant_vector.unsqueeze(1).expand(-1, backloop_vectors.shape[1], -1)
        all_input_vectors = torch.cat((constant_vector_expanded, backloop_input), dim=2)

        # Прогон через дирижера, получение предсказания инструментов
        conductor_h = self.conductor_lstm(all_input_vectors)
        instruments_logits = self.instruments_linear_parser(conductor_h)

        # Получение нот
        instruments_conductor_vectors = self.instruments_embeddings(tacts_instr)
        notes_logits, hn, cn = self.instruments_lstm(conductor_h, instruments_conductor_vectors, tacts_data)

        return notes_logits, instruments_logits

    def use_nn(self, prompt_idx:list, full_instr_list:torch.Tensor, backloop_vec = None, h0=None, c0=None, temperature=0.9, short_notes_coef=0.75, top_k=50):
        tacts_data: torch.Tensor

        # Получение половины вектора для инструментов
        instruments_vector = self.instruments_linear_parser(full_instr_list)
        vibe_vector = self.encoder_model(prompt_idx)
        constant_vector = torch.cat((vibe_vector, instruments_vector), dim=1)

        if backloop_vec is None:
            backloop_vec = torch.zeros(constant_vector.size(-1))

        # Получение всех векторов для инструментов
        input_vector = torch.cat((constant_vector, backloop_vec))

        # Прогон через дирижера, получение предсказания инструментов
        conductor_h = self.conductor_lstm(input_vector)
        instruments_logits = self.instruments_linear_parser(conductor_h)

        # Получение нот
        instruments_conductor_vectors = self.instruments_embeddings.weight * instruments_logits

        output_tact_data = {}
        tact_data = []

        # Получаем самые вероятные инструменты
        threshold = 0.35
        active_mask = instruments_conductor_vectors > threshold
        instruments_indices = torch.where(active_mask)[0]

        for idx in instruments_indices:
            tact_data.append([1])

        hn, cn = None, None

        for i in range(100):
            notes_logits, hn, cn = self.instruments_lstm(conductor_h.unsqueeze(0), instruments_conductor_vectors.unsqueeze(0).unsqueeze(0), torch.tensor(tact_data).unsqueeze(0).unsqueeze(0), h0=hn, c0=cn)

            for i, note_logits in enumerate(notes_logits):
                next_token_logits = note_logits[0, -1, :] / temperature
                next_token_logits[110:130] /= (2 - short_notes_coef)

                next_token_logits = torch.nan_to_num(next_token_logits, nan=0.0, posinf=10.0, neginf=-10.0)

                threshold = torch.topk(next_token_logits, top_k).values[-1]
                next_token_logits[next_token_logits < threshold] = -float('Inf')

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                if next_token == 2:
                    output_tact_data[instruments_indices[i]] = tact_data[i]
                    del tact_data[i]
                else:
                    tact_data[i].append(next_token)

        backloop_vector = self.backloop_encoder(torch.sum(self.instruments_lstm.midi_embeddings(tact_data)))
        return tact_data, backloop_vector


    def forward(self, prompt_idx:list, full_instr_list:torch.Tensor, tacts_instr:torch.Tensor = None, tacts_data:torch.Tensor = None, backloop_vec = None, h0 = None, c0 = None, temperature=0.9, short_notes_coef=0.75, top_k=50):
        if self.is_training:
            tact_data, instruments_logits = self.learn_nn(prompt_idx, full_instr_list, tacts_instr, tacts_data)
            return tact_data, instruments_logits
        else:
            tact_data, backloop_vector = self.use_nn(prompt_idx, full_instr_list, backloop_vec=backloop_vec, h0=h0, c0=c0, temperature=temperature, short_notes_coef=short_notes_coef, top_k=top_k)
            return tact_data, backloop_vector

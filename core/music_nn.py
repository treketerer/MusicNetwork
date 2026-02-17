import torch
import torch.nn as nn

from .encoder_linear import EncoderLinear
from .conductor_lstm import ConductorLSTM
from .instruments_lstm import InstrumentsLSTM
from .linear_utils import *

class MusicNN(nn.Module):
    def __init__(self, text_alphabet_size:int, midi_alphabet_size:int, instruments_counts = 129, inner_context_size=512, learning=True):
        super(MusicNN, self).__init__()

        self.is_training = learning
        self.instruments_counts = instruments_counts

        self.text_embeddings_size = 512
        self.midi_embeddings_size = 512
        self.instruments_embedding_size = 512

        self.inner_context_size = inner_context_size
        self.backloop_output_size = self.inner_context_size

        self.instruments_embeddings = nn.Embedding(
            self.instruments_counts,
            self.instruments_embedding_size
        )

        self.instruments_linear_parser = SongInstrumentsLinearParser(
             self.instruments_embedding_size,
             self.inner_context_size,
             self.inner_context_size,
             self.instruments_embeddings
        )

        self.encoder_model = EncoderLinear(
            text_alphabet_size,
            self.text_embeddings_size,
            self.inner_context_size,
            self.inner_context_size,
        )

        conductor_input_size = self.inner_context_size + self.inner_context_size + self.backloop_output_size
        self.conductor_lstm = ConductorLSTM(
            conductor_input_size,
            self.inner_context_size,
            self.inner_context_size
        )
        self.conductor_need_instruments_parser = ConductorInstrumentsParser(
            self.inner_context_size,
            self.inner_context_size,
            self.instruments_counts
        )

        self.instruments_lstm_input_size = self.inner_context_size + self.instruments_embedding_size + self.midi_embeddings_size
        self.instruments_lstm = InstrumentsLSTM(
            self.instruments_lstm_input_size,
            midi_alphabet_size,
            self.midi_embeddings_size
        )

        self.backloop_encoder = BackloopLinearEncoder(
            self.midi_embeddings_size,
            self.inner_context_size,
            self.backloop_output_size
        )

    def learn_nn(self, prompt_idx:torch.Tensor, full_instr_list:torch.Tensor, tacts_instr:torch.Tensor, tacts_data:torch.Tensor):
        # Прогон через Backloop для получения векторов в начало
        # print("\nSTART EMBEDDINGS")
        notes_emb = self.instruments_lstm.midi_embeddings(tacts_data).sum(dim=3)
#         print("ISNT EMV", tacts_data.shape, notes_emb.shape)
        backloop_vectors = self.backloop_encoder(notes_emb)
#         print("BACKLOOP", backloop_vectors.shape)

        # Получение половины вектора для инструментов
        instruments_vector = self.instruments_linear_parser(full_instr_list)
#         print("\ninstruments_vector", instruments_vector.shape)
        instruments_vector = instruments_vector.sum(dim=1)
#         print("sum instruments_vector", instruments_vector.shape)
        vibe_vector = self.encoder_model(prompt_idx)
#         print("vibe_vector", vibe_vector.shape)
        vibe_vector = vibe_vector.sum(dim=1)
#         print("sum vibe_vector", vibe_vector.shape)
        constant_vector = torch.cat((vibe_vector, instruments_vector), dim=1)
#         print("constant_vector", constant_vector.shape)

        # Получение всех векторов для инструментов
        zeros = torch.zeros_like(backloop_vectors[:, :1, :])
#         print("\nzeros", zeros.shape)
        backloop_input = torch.cat((zeros, backloop_vectors[:, :-1, :]), dim=1)
#         print("backloop_input", backloop_input.shape)

        constant_vector = constant_vector.unsqueeze(1).expand(-1, backloop_input.size(dim=1), -1)
#         print("constant_vector", constant_vector.shape)
        all_input_vectors = torch.cat((constant_vector, backloop_input), dim=2)
#         print("\nall_input_vectors", all_input_vectors.shape)

        # Прогон через дирижера, получение предсказания инструментов
        conductor_h, hn, cn = self.conductor_lstm(all_input_vectors)
#         print("\nconductor_h", conductor_h.shape)
        instruments_logits = self.conductor_need_instruments_parser(conductor_h)
#         print("instruments_logits", instruments_logits.shape)

        # Получение нот
        instruments_conductor_vectors = self.instruments_embeddings(tacts_instr)
#         print("\ninstruments_conductor_vectors", instruments_conductor_vectors.shape)
        notes_logits, hn, cn = self.instruments_lstm(conductor_h, instruments_conductor_vectors, tacts_data)
#         print("notes_logits", notes_logits.shape)
#         print(notes_logits.shape, instruments_logits.shape)
        return notes_logits, instruments_logits

    def use_nn(self, prompt_idx:list, full_instr_list:torch.Tensor, backloop_vec = None, h0=None, c0=None, temperature=0.9, short_notes_coef=0.75, top_k=50):
        tacts_data: torch.Tensor

        # Получение половины вектора для инструментов
        instruments_vector = self.instruments_linear_parser(full_instr_list).sum(dim=1)
        print("instruments_vector", instruments_vector.shape)
        vibe_vector = self.encoder_model(prompt_idx).sum(dim=1)
        print("vibe_vector", instruments_vector.shape)
        constant_vector = torch.cat((vibe_vector, instruments_vector), dim=1)
        print("constant_vector", constant_vector.shape)

        if backloop_vec is None:
            backloop_vec = torch.zeros((1, self.backloop_output_size))
        print("backloop_vec", backloop_vec.shape)

        # Получение всех векторов для инструментов
        input_vector = torch.cat((constant_vector, backloop_vec), dim=1)
        print("input_vector", input_vector.shape)

        # Прогон через дирижера, получение предсказания инструментов
        conductor_h, hn, cn = self.conductor_lstm(input_vector)
        print("conductor_h", conductor_h.shape)
        instruments_logits = self.conductor_need_instruments_parser(conductor_h)
        print("instruments_logits", instruments_logits.shape)
        print("instruments_logits", instruments_logits)

        # Получаем самые вероятные инструменты
        threshold = 0.2
        active_mask = instruments_logits > threshold
        instruments_indices = torch.where(active_mask)[0]
        print("INST", instruments_indices)

        hn, cn = None, None

        # Получение нот
        instruments_conductor_vectors = self.instruments_embeddings(instruments_indices)
        print("instruments_conductor_vectors", instruments_conductor_vectors.shape)

        conductor_h = conductor_h.unsqueeze(0)
        instruments_conductor_vectors = instruments_conductor_vectors.unsqueeze(0).unsqueeze(0)

        current_tact_data = [[1] for i in range(len(instruments_indices))]
        finished_instruments = [False] * len(instruments_indices)

        max_iters = 30
        for i in range(max_iters):
            tact_data_tensor = torch.tensor(current_tact_data, dtype=torch.long)
            tact_data_tensor = tact_data_tensor.unsqueeze(0).unsqueeze(0)

            print(conductor_h.shape, instruments_conductor_vectors.shape, tact_data_tensor.shape)
            notes_logits, hn, cn = self.instruments_lstm(conductor_h, instruments_conductor_vectors, tact_data_tensor, h0=hn, c0=cn)

            for inst_idx in range(len(notes_logits)):
                if finished_instruments[inst_idx]:
                    continue

                next_token_logits = notes_logits[0, 0, inst_idx, -1, :].flatten()
                next_token_logits = next_token_logits / temperature
                next_token_logits[110:130] /= (2 - short_notes_coef)

                next_token_logits = torch.nan_to_num(next_token_logits, nan=0.0, posinf=10.0, neginf=-10.0)

                threshold = torch.topk(next_token_logits, top_k).values[-1]
                next_token_logits[next_token_logits < threshold] = -float('Inf')

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                if not finished_instruments[inst_idx]:
                    current_tact_data[inst_idx].append(next_token)
                if next_token == 2 or i == max_iters:
                    print("Finish")
                    finished_instruments[inst_idx] = True

        final_tact_data = {}

        for instrument in instruments_indices:
            final_tact_data[instrument] = current_tact_data[instrument].copy()

        emb = self.instruments_lstm.midi_embeddings(torch.tensor(current_tact_data))
        print(emb.shape)
        sum_notes = torch.sum(emb, dim=1).unsqueeze(0).unsqueeze(0)
        backloop_vector = self.backloop_encoder(sum_notes)
        backloop_vector = backloop_vector.squeeze(0)
        return final_tact_data, backloop_vector


    def forward(self, prompt_idx:torch.Tensor, full_instr_list:torch.Tensor, tacts_instr:torch.Tensor = None, tacts_data:torch.Tensor = None, backloop_vec = None, h0 = None, c0 = None, temperature=0.9, short_notes_coef=0.75, top_k=50):
        if self.is_training:
            tact_data, instruments_logits = self.learn_nn(prompt_idx, full_instr_list, tacts_instr, tacts_data)
            return tact_data, instruments_logits
        else:
            tact_data, backloop_vector = self.use_nn(prompt_idx, full_instr_list, backloop_vec=backloop_vec, h0=h0, c0=c0, temperature=temperature, short_notes_coef=short_notes_coef, top_k=top_k)
            return tact_data, backloop_vector

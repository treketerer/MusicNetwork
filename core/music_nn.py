import torch
import torch.nn as nn

from .text_encoder import TextEncoderGRU
from .conductor_lstm import ConductorLSTM
from .instruments_lstm import InstrumentsLSTM
from .model_utils import *

class MusicNN(nn.Module):
    def __init__(self, text_alphabet_size:int, midi_alphabet_size:int,
            instruments_counts = 129, inner_context_size=768,
            text_embeddings_size = 256, midi_embeddings_size = 128,
            instruments_embedding_size = 128, learning=True):

        super(MusicNN, self).__init__()

        self.is_training = learning
        self.instruments_counts = instruments_counts

        self.text_embeddings_size = text_embeddings_size
        self.midi_embeddings_size = midi_embeddings_size
        self.instruments_embedding_size = instruments_embedding_size

        self.inner_context_size = inner_context_size
        self.backloop_output_size =  int(self.inner_context_size * 1.5)

        self.instruments_embeddings = nn.Embedding(
            self.instruments_counts + 1,
            self.instruments_embedding_size,
            padding_idx=129
        )

        self.instruments_linear_parser = SongInstrumentsLinearParser(
             self.instruments_embedding_size,
             self.inner_context_size,
             self.inner_context_size,
             self.instruments_embeddings
        )

        self.encoder_model = TextEncoderGRU(
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
        self.style_projector = nn.Sequential(
            nn.Linear(self.inner_context_size, self.inner_context_size),
            nn.ReLU()
        )

        self.cond_size = self.inner_context_size + self.inner_context_size
        self.instruments_lstm_input_size = (
                self.cond_size +
                self.instruments_embedding_size +
                self.midi_embeddings_size
        )

        self.instruments_lstm = InstrumentsLSTM(
            self.instruments_lstm_input_size,
            self.inner_context_size,
            self.cond_size,
            midi_alphabet_size,
            self.midi_embeddings_size,
            self.instruments_embedding_size
        )

        self.backloop_encoder = BackloopEncoder(
            self.midi_embeddings_size,
            self.inner_context_size,
            self.backloop_output_size
        )

    def learn_nn(self, prompt_idx:torch.Tensor, full_instr_list:torch.Tensor, tacts_instr:torch.Tensor, tacts_data:torch.Tensor):
        # Прогон через Backloop для получения векторов в начало
        notes_emb = self.instruments_lstm.midi_embeddings(tacts_data)
        backloop_vectors = self.backloop_encoder(notes_emb)

        # Получение половины вектора для инструментов
        instruments_vector = self.instruments_linear_parser(full_instr_list)
        instruments_vector = instruments_vector.mean(dim=1)
        vibe_vector = self.encoder_model(prompt_idx)

        # Прогон через дирижера, получение предсказания инструментов
        conductor_h, hn, cn = self.conductor_lstm(vibe_vector, instruments_vector, backloop_vectors)
        instruments_logits = self.conductor_need_instruments_parser(conductor_h)

        # Функция потеря удержания вайба в h
        # h_projected = self.style_projector(conductor_h)
        # vibe_expanded = vibe_vector.unsqueeze(1).expand(-1, conductor_h.size(1), -1)
        # loss_style = nn.MSELoss()(h_projected, vibe_expanded)

        # Получение нот
        instruments_conductor_vectors = self.instruments_embeddings(tacts_instr)
        notes_logits, hn, cn = self.instruments_lstm(conductor_h, vibe_vector, instruments_conductor_vectors, tacts_data)
        return notes_logits, instruments_logits

    def use_nn(self, prompt_idx:list, full_instr_list:torch.Tensor, backloop_vec = None, max_tokens=100, temperature=0.9, short_notes_coef=0.75, top_k=50, conductor_h=None, conductor_c=None):
        device = next(self.parameters()).device

        # Получение половины вектора для инструментов
        instruments_vector = self.instruments_linear_parser(full_instr_list).mean(dim=1)
        vibe_vector = self.encoder_model(prompt_idx)#.sum(dim=1)
        if backloop_vec is None:
            backloop_vec = torch.zeros((1, 1, self.backloop_output_size), device=device)

        # Прогон через дирижера, получение предсказания инструментов
        conductor_h, cond_h, cond_c = self.conductor_lstm(vibe_vector, instruments_vector, backloop_vec, h0=conductor_h, c0=conductor_c)
        instruments_logits = self.conductor_need_instruments_parser(conductor_h)

        # Получаем самые вероятные инструменты
        instruments_probs = torch.sigmoid(instruments_logits)

        threshold = 0.8
        active_mask = instruments_probs > threshold
        instruments_indices = torch.where(active_mask)[2]
        print(instruments_indices)

        hn, cn = None, None

        # Получение нот
        instruments_conductor_vectors = self.instruments_embeddings(instruments_indices)

        instruments_conductor_vectors = instruments_conductor_vectors.unsqueeze(0).unsqueeze(0)

        current_tact_data = [[1] for i in range(len(instruments_indices))]
        finished_instruments = [False] * len(instruments_indices)
        last_tokens = torch.ones((len(instruments_indices), 1), dtype=torch.long).to(device)

        for i in range(max_tokens):
            last_notes_data_tensor = last_tokens.unsqueeze(0).unsqueeze(0)
            notes_logits, hn, cn = self.instruments_lstm(conductor_h, vibe_vector, instruments_conductor_vectors, last_notes_data_tensor, h0=hn, c0=cn)

            for inst_idx in range(len(instruments_indices)):
                if finished_instruments[inst_idx]:
                    current_tact_data[inst_idx].append(0)
                    last_tokens[inst_idx] = 0
                    continue

                next_token_logits = notes_logits[0, 0, inst_idx, -1, :].flatten()
                next_token_logits = next_token_logits / temperature
                # next_token_logits[110:130] /= (2 - short_notes_coef)

                next_token_logits = torch.nan_to_num(next_token_logits, nan=0.0, posinf=10.0, neginf=-10.0)

                threshold = torch.topk(next_token_logits, top_k).values[-1]
                next_token_logits[next_token_logits < threshold] = -float('Inf')

                # if len(current_tact_data[inst_idx]) < 10:  # Хотим минимум 10 токенов
                #     next_token_logits[2] = -float('Inf')

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                if not finished_instruments[inst_idx]:
                    current_tact_data[inst_idx].append(next_token)
                    last_tokens[inst_idx] = next_token
                if next_token == 2 or i == max_tokens - 1:
                    finished_instruments[inst_idx] = True

        final_tact_data = {}

        for i, instrument in enumerate(instruments_indices):
            clean_notes = [n for n in current_tact_data[i] if n != 0]
            final_tact_data[instrument] = clean_notes

        emb = self.instruments_lstm.midi_embeddings(torch.tensor(current_tact_data, device=device)).to(device)
        sum_notes = emb.unsqueeze(0).unsqueeze(0)
        backloop_vector = self.backloop_encoder(sum_notes)

        return final_tact_data, backloop_vector, cond_h, cond_c


    def forward(self, prompt_idx:torch.Tensor, full_instr_list:torch.Tensor, tacts_instr:torch.Tensor = None, tacts_data:torch.Tensor = None, backloop_vec = None, temperature=0.9, short_notes_coef=0.75, top_k=50, conductor_h=None, conductor_c=None):
        if self.is_training:
            tact_data, instruments_logits = self.learn_nn(prompt_idx, full_instr_list, tacts_instr, tacts_data)
            return tact_data, instruments_logits
        else:
            tact_data, backloop_vector, cond_h, cond_c = self.use_nn(prompt_idx, full_instr_list, backloop_vec=backloop_vec, temperature=temperature, short_notes_coef=short_notes_coef, top_k=top_k, conductor_h=conductor_h, conductor_c=conductor_c)
            return tact_data, backloop_vector, cond_h, cond_c

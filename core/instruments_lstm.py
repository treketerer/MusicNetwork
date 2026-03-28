import torch
import torch.nn as nn
from torch import tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InstrumentsLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, cond_size: int, midi_alphabet_size: int, midi_emb_dim: int, instruments_emb_dim: int):
        super(InstrumentsLSTM, self).__init__()

        self.midi_embeddings = nn.Embedding(midi_alphabet_size, midi_emb_dim, padding_idx=0)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_dim = 2

        self.h0_linear = nn.Linear(cond_size, self.layer_dim * self.hidden_size)
        self.c0_linear = nn.Linear(cond_size, self.layer_dim * self.hidden_size)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim= cond_size + instruments_emb_dim,
            num_heads=8,
            batch_first=True
        )

        self.fusion_linear = nn.Linear(self.input_size, self.hidden_size)
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.layer_dim,
            batch_first=True,
            dropout=0.25
        )

        self.midi_out = nn.Linear(self.hidden_size, midi_alphabet_size)

    def forward(self, conductor_context: torch.Tensor, vibe_vector: torch.Tensor, instruments_emb: torch.Tensor, tacts_notes: torch.Tensor, h0=None, c0=None):
        """
        conductor_context - (batch, tacts, conductor_out_dim)
        instruments_emb - (batch, tacts, instruments, instruments_emb_dim)
        tacts_notes - (batch, tacts, instruments, notes)
        """

        bd, tb, ib, nb = tacts_notes.shape

        # Инициализация векторов вайба и дирижера
        vibe_expanded = vibe_vector.unsqueeze(1).expand(-1, tb, -1)
        constant_vector = torch.cat((conductor_context, vibe_expanded), dim=2)
        cond_vec = constant_vector.unsqueeze(2).expand(bd, tb, ib, -1) # .unsqueeze(3)
        inst_vec = instruments_emb.expand(bd, tb, ib, -1) # .unsqueeze(3)

        # Подготовка для внимания
        cond_and_inst = torch.cat([cond_vec, inst_vec], dim=-1)
        cond_and_inst_flat  = cond_and_inst.view(bd * tb, ib, -1)

        # Внимание по инструментам в тактах
        attn_output, _ = self.multihead_attn(cond_and_inst_flat, cond_and_inst_flat, cond_and_inst_flat )
        attn_output = cond_and_inst_flat + attn_output
        attn_output = attn_output.view(bd, tb, ib, 1, -1).expand(-1, -1, -1, nb, -1)

        # Подготовка к lstm
        notes_vec = self.midi_embeddings(tacts_notes)
        input_vec = torch.cat([attn_output, notes_vec], dim=-1)
        input_flat = input_vec.view(bd * tb * ib, nb, -1)

        if h0 is None and c0 is None:
            flat_constant = constant_vector.unsqueeze(2).expand(bd, tb, ib, -1).reshape(bd * tb * ib, -1)

            h0 = self.h0_linear(flat_constant).view(bd * tb * ib, self.layer_dim, self.hidden_size).permute(1, 0, 2).contiguous()
            c0 = self.c0_linear(flat_constant).view(bd * tb * ib, self.layer_dim, self.hidden_size).permute(1, 0, 2).contiguous()

        compressed_vector = self.fusion_linear(input_flat)

        out, (hn, cn) = self.lstm(compressed_vector, (h0, c0))

        logits = self.midi_out(out)
        out_logits = logits.view(bd, tb, ib, nb, -1)
        return out_logits, hn, cn
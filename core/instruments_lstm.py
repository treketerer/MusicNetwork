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

        self.note_dropout = nn.Dropout(0.2)

        self.attn_fusion_linear = nn.Linear(cond_size + instruments_emb_dim, self.hidden_size)
        self.attn_fusion_act = nn.GELU()

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim = self.hidden_size,
            num_heads=8,
            batch_first=True
        )

        self.attn_norm = nn.LayerNorm(self.hidden_size)
        self.attn_dropout = nn.Dropout(0.1)

        self.fusion_norm = nn.LayerNorm(self.hidden_size + midi_emb_dim)

        self.lstm = nn.LSTM(
            input_size=self.hidden_size + midi_emb_dim,
            hidden_size=self.hidden_size,
            num_layers=self.layer_dim,
            batch_first=True,
            dropout=0.25
        )

        self.midi_out = nn.Linear(self.hidden_size, midi_alphabet_size)

    def forward(self, conductor_context: torch.Tensor, vibe_vector: torch.Tensor, instruments_emb: torch.Tensor, tacts_notes: torch.Tensor, tacts_instr: torch.Tensor, h0=None, c0=None):
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
        cond_and_inst_flat  = cond_and_inst.reshape(bd * tb, ib, -1)

        # Внимание по инструментам в тактах
        key_padding_mask = (tacts_instr == 129).reshape(bd * tb, ib)

        # Фикс от None
        bad_rows = key_padding_mask.all(dim=1)
        if bad_rows.any():
            key_padding_mask[bad_rows, 0] = False

        fusion_attn_vector = self.attn_fusion_linear(cond_and_inst_flat)
        fusion_attn_vector = self.attn_fusion_act(fusion_attn_vector)

        attn_output, _ = self.multihead_attn(
            fusion_attn_vector,
            fusion_attn_vector,
            fusion_attn_vector,
            key_padding_mask=key_padding_mask
        )

        attn_output = self.attn_dropout(attn_output)
        attn_output = self.attn_norm(fusion_attn_vector + attn_output)

        attn_output = attn_output.reshape(bd, tb, ib, 1, -1).expand(-1, -1, -1, nb, -1)

        # Подготовка к lstm
        notes_vec = self.note_dropout(self.midi_embeddings(tacts_notes))
        input_vec = torch.cat([attn_output, notes_vec], dim=-1)
        input_flat = input_vec.reshape(bd * tb * ib, nb, -1)

        if h0 is None and c0 is None:
            flat_constant = constant_vector.unsqueeze(2).expand(bd, tb, ib, -1).reshape(bd * tb * ib, -1)

            h0 = self.h0_linear(flat_constant).view(bd * tb * ib, self.layer_dim, self.hidden_size).permute(1, 0, 2).contiguous()
            c0 = self.c0_linear(flat_constant).view(bd * tb * ib, self.layer_dim, self.hidden_size).permute(1, 0, 2).contiguous()

        norm_vector = self.fusion_norm(input_flat)

        out, (hn, cn) = self.lstm(norm_vector, (h0, c0))

        logits = self.midi_out(out)
        out_logits = logits.view(bd, tb, ib, nb, -1)
        return out_logits, hn, cn
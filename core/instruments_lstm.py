import torch
import torch.nn as nn
from torch import tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InstrumentsLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, midi_alphabet_size: int, midi_emb_dim: int):
        super(InstrumentsLSTM, self).__init__()

        self.midi_embeddings = nn.Embedding(midi_alphabet_size, midi_emb_dim, padding_idx=0)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_dim = 2

        self.fusion_linear = nn.Linear(self.input_size, self.hidden_size)
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.layer_dim,
            batch_first=True,
            dropout=0.15
        )

        self.midi_out = nn.Linear(self.hidden_size, midi_alphabet_size)

    def forward(self, conductor_context: torch.Tensor, vibe_vector: torch.Tensor, instruments_emb: torch.Tensor, tacts_notes: torch.Tensor, h0=None, c0=None):
        """
        conductor_context - (batch, tacts, conductor_out_dim)
        instruments_emb - (batch, tacts, instruments, instruments_emb_dim)
        tacts_notes - (batch, tacts, instruments, notes)
        """

        bd, tb, ib, nb = tacts_notes.shape
        vibe_expanded = vibe_vector.unsqueeze(1).expand(-1, tb, -1)
        constant_vector = torch.cat((conductor_context, vibe_expanded), dim=2)
        notes_vec = self.midi_embeddings(tacts_notes)
        cond_vec = constant_vector.unsqueeze(2).unsqueeze(3).expand(bd, tb, ib, nb, -1)
        inst_vec = instruments_emb.unsqueeze(3).expand(bd, tb, ib, nb, -1)

        input_vec = torch.cat([cond_vec, inst_vec, notes_vec], dim=-1)
        input_flat = input_vec.view(bd * tb * ib, nb, -1)

        if h0 is None and c0 is None:
            h0 = torch.zeros(self.layer_dim, bd * tb * ib, self.hidden_size).to(device)
            c0 = torch.zeros(self.layer_dim, bd * tb * ib, self.hidden_size).to(device)

        compressed_vector = self.fusion_linear(input_flat)

        out, (hn, cn) = self.lstm(compressed_vector, (h0, c0))
        logits = self.midi_out(out)
        out_logits = logits.view(bd, tb, ib, nb, -1)
        return out_logits, hn, cn
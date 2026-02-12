import torch
import torch.nn as nn
from torch import tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InstrumentsLSTM(nn.Module):
    def __init__(self, input_size: int, instruments_count: int, midi_emb_size: int):
        super(InstrumentsLSTM, self).__init__()

        self.instruments_count = instruments_count

        self.midi_embeddings = nn.Embedding(self.instruments_count, midi_emb_size)

        self.input_size = input_size
        self.hidden_state = self.input_size
        self.layer_dim = 3
        self.lstm = nn.LSTM(
            input_size=self.hidden_state,
            hidden_size=self.hidden_state,
            num_layers=self.layer_dim,
            batch_first=True,
            dropout=0.2
        )

        self.midi_out = nn.Linear(self.hidden_state, self.hidden_state)

    def forward(self, conductor_context: torch.Tensor, instruments_emb: torch.Tensor, tacts_notes: torch.Tensor, h0=None, c0=None):
        """
        conductor_context - (batch, tacts, conductor_out_dim)
        instruments_emb - (batch, tacts, instruments, instruments_emb_dim)
        tacts_notes - (batch, tacts, instruments, notes)
        """

        bd, tb, ib, nb = tacts_notes.shape

        notes_vec = self.midi_embeddings(tacts_notes)
        cond_vec = conductor_context.unsqueeze(2).unsqueeze(3).expand(bd, tb, ib, nb, -1)
        inst_vec = instruments_emb.expand(nb, 2).expand(bd, tb, ib, nb, -1)

        input_vec = torch.cat([cond_vec, inst_vec, notes_vec], dim=-1)
        input_flat = input_vec.view(bd * tb * ib, nb, -1)

        if h0 is None and c0 is None:
            h0 = torch.zeros(self.layer_dim, bd * tb * ib, self.hidden_state).to(device)
            c0 = torch.zeros(self.layer_dim, bd * tb * ib, self.hidden_state).to(device)

        out, (hn, cn) = self.lstm(input_flat, (h0, c0))
        logits = self.midi_out(out)
        out_logits = logits.view(bd, tb, ib, nb, -1)
        return out_logits, hn, cn
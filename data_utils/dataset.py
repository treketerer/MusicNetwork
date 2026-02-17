import random
import re

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
import json

from data.keywords import ban_list, all_translations, all_instruments

class MusicStreamingDataset(IterableDataset):
    def __init__(self, parsed_midi_path, words_alphabet_path, midi_alphabet_path,
                 buffer_size=100, max_tacts=20, max_token_in_tact=64):

        self.max_tacts = max_tacts
        self.max_token_in_tact = max_token_in_tact

        self.parsed_midi_path = parsed_midi_path

        self.words_alphabet_idx = None
        self.words_alphabet_words = None
        self.midi_alphabet_idx = None
        self.midi_alphabet_midi = None
        self.midi_alphabet_len = None

        self.dataset_len = -1

        self.max_buffer_len = buffer_size

        self.all_instruments = all_instruments
        self.reverse_all_instruments = {value: key for value, key in all_instruments.items()}

        with open(words_alphabet_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) >= 2:
                self.words_alphabet_idx = json.loads(lines[0])
                self.words_alphabet_words = json.loads(lines[1])
            else:
                print("ФАЙЛ АЛФАВИТА ПУСТ!!!")

        with open(midi_alphabet_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) >= 2:
                self.midi_alphabet_idx = json.loads(lines[0])
                self.midi_alphabet_midi = json.loads(lines[1])
                self.midi_alphabet_len = len(self.midi_alphabet_idx)
            else:
                print("ФАЙЛ MIDI АЛФАВИТА ПУСТ!!!")

    def __len__(self):
        if self.dataset_len == -1:
            with open(self.parsed_midi_path, 'r', encoding='utf-8') as f:
                self.dataset_len = sum(1 for _ in f)
        return self.dataset_len

    def get_words_alphabet_len(self):
        return len(self.words_alphabet_words)

    def get_midi_alphabet_len(self):
        return len(self.midi_alphabet_midi)

    def midi_to_idx(self, input_midi):
        answer = []
        for midi in input_midi.split():
            if midi in self.midi_alphabet_midi:
                answer.append(self.midi_alphabet_midi.get(midi))
        return answer

    def idx_to_midi(self, input_idx):
        answer = []
        alphabet_len = len(self.midi_alphabet_idx)
        for idx in input_idx:
            if idx < alphabet_len:
                answer.append(self.midi_alphabet_idx.get(idx))
        return answer

    def words_to_idx(self, input_words) -> tuple[list, list]:
        input_words = " ".join([word for word in input_words.split() if word not in ban_list])
        instruments_idx, input_words = self.parse_instruments(input_words)

        words_idx = []
        for word in input_words.split():
            if word in self.words_alphabet_words:
                words_idx.append(self.words_alphabet_words.get(word.lower()))
            else:
                words_idx.append(self.words_alphabet_words.get("<unk>"))
        return words_idx, instruments_idx

    def parse_instruments(self, input_words: str) -> tuple[list, str]:
        input_words = re.sub(r'[^\w\s]', '', input_words).strip()
        instruments_arr = []

        for i, key in enumerate(list(all_translations.keys())[:128]):
            find = False
            if key in input_words:
                input_words = input_words.replace(key, '')
                find = True

            for name in all_translations[key]:
                if name in input_words:
                    input_words = input_words.replace(name, '')
                    find = True
                    break
            if find:
                instruments_arr.append(i)
        return instruments_arr, input_words


    def idx_to_words(self, input_idx):
        answer = []
        alphabet_len = len(self.words_alphabet_idx)
        for idx in input_idx:
            if idx < alphabet_len:
                answer.append(self.words_alphabet_idx.get(idx))
        return answer


    def collate_fn(self, batch):
        # Фильтруем None (битые данные)
        batch = [item for item in batch if item is not None]
        if not batch: return None

        prompts = [item['idx_prompt'] for item in batch]
        instruments = [item['instruments'] for item in batch]

        batch_size = len(batch)
        max_tacts = 0
        max_inst = 0
        max_notes = 0

        for item in batch:
            # [batch, song, tact, instruments]
            t_len, i_len = item['tacts_instruments'].shape
            # [batch, song, tact, instruments, notes]
            n_len = item['tacts_data'].shape[2]

            max_tacts = max(max_tacts, t_len)
            max_inst = max(max_inst, i_len)
            max_notes = max(max_notes, n_len)

        max_notes = max(max_notes, self.max_token_in_tact)

        tacts_instruments_padded = torch.zeros((batch_size, max_tacts, max_inst), dtype=torch.long)
        tacts_data_padded = torch.zeros((batch_size, max_tacts, max_inst, max_notes), dtype=torch.long)

        for i, item in enumerate(batch):
            t_len, i_len = item['tacts_instruments'].shape
            n_len = item['tacts_data'].shape[2]

            tokens_slise = min(n_len, max_notes)
            tacts_instruments_padded[i, :t_len, :i_len] = item['tacts_instruments']
            tacts_data_padded[i, :t_len, :i_len, :n_len] = item['tacts_data'][:t_len, :i_len, :tokens_slise]

        # Паддинг (заполняем нулями короткие последовательности)
        prompts_padded = pad_sequence(prompts, batch_first=True, padding_value=0)
        instruments =  pad_sequence(instruments, batch_first=True, padding_value=0)

        return {'idx_prompts': prompts_padded, 'tacts_instruments': tacts_instruments_padded, 'tacts_data': tacts_data_padded, 'instruments': instruments}

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        buffer = []

        with open(self.parsed_midi_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i % num_workers != worker_id:
                    continue

                track_data = json.loads(line)

                if len(buffer) < self.max_buffer_len:
                    buffer.append(track_data)
                else:
                    random_idx = random.randrange(0, self.max_buffer_len)
                    if buffer[random_idx] is None: continue
                    yield self.parse_data_tensors(buffer[random_idx])
                    buffer[random_idx] = track_data

            random.shuffle(buffer)
            for item in buffer:
                if item is None: continue
                yield self.parse_data_tensors(item)

    def parse_data_tensors(self, data):
        res = data.copy()
        res['idx_prompt'] = torch.tensor(res['idx_prompt'], dtype=torch.long)
        res['instruments'] = torch.tensor(res['instruments'], dtype=torch.long)

        max_instruments = 0
        max_data = 0

        tacts = res.get('tacts')
        tacts_len = len(tacts)

        if tacts_len > self.max_tacts:
            rand_start = random.randint(0, tacts_len - self.max_tacts)
            tacts = tacts[rand_start : rand_start+self.max_tacts]

        tacts_len = len(tacts)

        for tact in tacts:
            for k_idx, key in enumerate(list(tact.keys())):
                if max_instruments < k_idx + 1:
                    max_instruments = k_idx + 1
                for n_key, note in enumerate(tact[key]):
                    if max_data < n_key + 1:
                        max_data = n_key + 1

        tacts_instruments = torch.zeros((tacts_len, max_instruments), dtype=torch.long)
        tacts_data = torch.zeros((tacts_len, max_instruments, max_data), dtype=torch.long)

        for tact_idx, tact in enumerate(tacts):
            for k_idx, key in enumerate(list(tact.keys())):
                int_key = int(key)
                tacts_instruments[tact_idx, k_idx] = int_key

                for n_key, note in enumerate(tact[key]):
                    tacts_data[tact_idx, k_idx, n_key] = tact[key][n_key]

        res['tacts_instruments'] = tacts_instruments
        res['tacts_data'] = tacts_data
        return res
import random

import torch
from torch import tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
import json

from content.project.data.midi_alphabet import midi_alph_midi, midi_alph_idx

class MusicStreamingDataset(IterableDataset):
    def __init__(self, word_prompts_path, idx_prompts_path, words_alphabet_path, parsed_midi_path, buffer_size=1000):
        self.parsed_midi_path = parsed_midi_path

        self.midi_alph_midi = midi_alph_midi
        self.midi_alph_idx = midi_alph_idx

        self.tracks_metadata = {}

        self.words_alphabet_idx = None
        self.words_alphabet_words = None

        self.max_buffet_len = buffer_size

        with open(words_alphabet_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) >= 2:
                self.words_alphabet_idx = json.loads(lines[0])
                self.words_alphabet_words = json.loads(lines[1])
            else:
                print("ФАЙЛ АЛФАВИТА ПУСТ!!!")

        with open(word_prompts_path, 'r', encoding='utf-8') as f_words, \
            open(idx_prompts_path, 'r', encoding='utf-8') as f_idx:
            for word_line, idx_line in zip(f_words, f_idx):
                words_data = json.loads(word_line)
                idx_data = json.loads(idx_line)

                now_words_md5 = words_data.get('md5')
                if now_words_md5 not in self.tracks_metadata:
                    self.tracks_metadata[now_words_md5] = {'md5': now_words_md5}
                self.tracks_metadata[now_words_md5]['words_prompt'] = words_data.get('prompt_keys')

                now_idx_md5 = idx_data.get('md5')
                if now_idx_md5 not in self.tracks_metadata:
                    self.tracks_metadata[now_idx_md5] = {'md5': now_idx_md5}
                self.tracks_metadata[now_idx_md5]['idx_prompt'] = idx_data.get('prompt_keys')

        print(self.words_alphabet_idx, self.words_alphabet_words)
    def get_words_alphabet_len(self):
        return len(self.words_alphabet_words)

    def get_midi_alphabet_len(self):
        return len(self.midi_alph_midi)

    def midi_to_idx(self, input_midi):
        answer = []
        for midi in input_midi.split():
            if midi in self.midi_alph_midi:
                answer.append(self.midi_alph_midi.get(midi))
            # else:
            #     answer.append(self.midi_alph_dict.index("<unk>"))
        return answer

    def idx_to_midi(self, input_idx):
        answer = []
        alph_len = len(self.midi_alph_idx)
        for idx in input_idx:
            if idx < alph_len:
                answer.append(self.midi_alph_idx.get(idx))
        return answer

    def words_to_idx(self, input_words):
        answer = []
        for word in input_words.split():
            if word in self.words_alphabet_words:
                answer.append(self.words_alphabet_words.get(word.lower()))
            else:
                answer.append(self.words_alphabet_words.get("<unk>"))
        return answer

    def idx_to_words(self, input_idx):
        answer = []
        alph_len = len(self.words_alphabet_idx)
        for idx in input_idx:
            if idx < alph_len:
                answer.append(self.words_alphabet_idx.get(idx))
        return answer

    def parse_line(self, line):
        parsed = json.loads(line)
        md5 = parsed.get("md5")
        tokens = parsed.get('tokens')

        MAX_SEQ_LEN = 512
        if len(tokens) > MAX_SEQ_LEN:
            tokens = tokens[:MAX_SEQ_LEN]

        basedata = self.tracks_metadata.get(md5)

        if basedata:
            return {**basedata, 'tokens': tokens}
        return None

    def collate_fn(self, batch):
        # Фильтруем None (битые данные)
        batch = [item for item in batch if item is not None]
        if not batch: return None

        prompts = [item['idx_prompt'] for item in batch]
        tokens = [item['tokens'] for item in batch]

        # Паддинг (заполняем нулями короткие последовательности)
        prompts_padded = pad_sequence(prompts, batch_first=True, padding_value=0)
        tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)

        return {'idx_prompts': prompts_padded, 'tokens': tokens_padded}

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

                try:
                    parsed_data = self.parse_line(line)
                except:
                    continue

                if len(buffer) < self.max_buffet_len:
                    buffer.append(parsed_data)
                else:
                    random_idx = random.randrange(0, self.max_buffet_len)
                    yield self.parse_data_tensors(buffer[random_idx])
                    buffer[random_idx] = parsed_data

            random.shuffle(buffer)
            for item in buffer:
                yield self.parse_data_tensors(item)

    def parse_data_tensors(self, data):
        res = data.copy()
        res['idx_prompt'] = torch.tensor(res['idx_prompt'], dtype=torch.long)
        res['tokens'] = tensor(res.get('tokens'), dtype=torch.long)
        return res
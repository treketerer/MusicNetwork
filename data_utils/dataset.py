import random
import re

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
import json

from data.keywords import ban_list, all_translations, all_instruments


class MusicStreamingDataset(IterableDataset):
    def __init__(self, parsed_midi_path, word_prompts_path, idx_prompts_path, words_alphabet_path, midi_alphabet_path,
                 buffer_size=1000):
        self.parsed_midi_path = parsed_midi_path

        self.tracks_metadata = {}

        self.words_alphabet_idx = None
        self.words_alphabet_words = None
        self.midi_alphabet_idx = None
        self.midi_alphabet_midi = None

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
            else:
                print("ФАЙЛ MIDI АЛФАВИТА ПУСТ!!!")

        with open(word_prompts_path, 'r', encoding='utf-8') as f_words, \
                open(idx_prompts_path, 'r', encoding='utf-8') as f_idx:
            for word_line, idx_line in zip(f_words, f_idx):
                # words_data = json.loads(word_line)
                idx_data = json.loads(idx_line)

                # now_words_md5 = words_data.get('md5')
                # if now_words_md5 not in self.tracks_metadata:
                #     self.tracks_metadata[now_words_md5] = {'md5': now_words_md5}
                # self.tracks_metadata[now_words_md5]['words_prompt'] = words_data.get('prompt_keys')

                now_idx_md5 = idx_data.get('md5')
                if now_idx_md5 not in self.tracks_metadata:
                    self.tracks_metadata[now_idx_md5] = {'md5': now_idx_md5}
                self.tracks_metadata[now_idx_md5]['idx_prompt'] = idx_data.get('prompt_keys')
                self.tracks_metadata[now_idx_md5]['instruments'] = idx_data.get('instruments')

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
        input_words = re.sub(r'[^\w\s-]', '', input_words)
        instruments_arr = []

        for key in all_translations.keys()[:129]:
            find = False

            if key in input_words:
                input_words = input_words.replace(key, '')
                find = True

            for name in all_translations[key]:
                if name in input_words:
                    input_words = input_words.replace(name, '')
                    find = True

            if find and key in self.reverse_all_instruments:
                instruments_arr.append(self.reverse_all_instruments[key])

        return instruments_arr, input_words


    def idx_to_words(self, input_idx):
        answer = []
        alphabet_len = len(self.words_alphabet_idx)
        for idx in input_idx:
            if idx < alphabet_len:
                answer.append(self.words_alphabet_idx.get(idx))
        return answer

    def parse_line(self, line):
        parsed = json.loads(line)
        md5 = parsed.get("md5")
        tokens = parsed.get('tacts')

        MAX_SEQ_LEN = 756
        if len(tokens) > MAX_SEQ_LEN:
            tokens = tokens[:MAX_SEQ_LEN]

        basedata = self.tracks_metadata.get(md5)

        if basedata:
            return {**basedata, 'tacts': tokens}
        return None

    def collate_fn(self, batch):
        # Фильтруем None (битые данные)
        batch = [item for item in batch if item is not None]
        if not batch: return None

        prompts = [item['idx_prompt'] for item in batch]
        tacts = [item['tacts'] for item in batch]
        instruments = [item['instruments'] for item in batch]

        # Паддинг (заполняем нулями короткие последовательности)
        prompts_padded = pad_sequence(prompts, batch_first=True, padding_value=0)
        tacts_padded = pad_sequence(tacts, batch_first=True, padding_value=0)
        instruments =  pad_sequence(instruments, batch_first=True, padding_value=0)

        return {'idx_prompts': prompts_padded, 'tacts': tacts_padded, 'instruments': instruments}

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

                if len(buffer) < self.max_buffer_len:
                    buffer.append(parsed_data)
                else:
                    random_idx = random.randrange(0, self.max_buffer_len)
                    yield self.parse_data_tensors(buffer[random_idx])
                    buffer[random_idx] = parsed_data

            random.shuffle(buffer)
            for item in buffer:
                yield self.parse_data_tensors(item)

    def parse_data_tensors(self, data):
        res = data.copy()
        res['idx_prompt'] = torch.tensor(res['idx_prompt'], dtype=torch.long)
        res['instruments'] = torch.tensor(res['instruments'], dtype=torch.long)
        res['tacts'] = torch.tensor(res.get('tacts'), dtype=torch.long)
        return res
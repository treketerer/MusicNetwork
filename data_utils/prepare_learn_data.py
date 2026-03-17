import json
import random
import os
import re
from collections import Counter

from numpy.ma.core import equal

from data.keywords import all_translations
from metaparser import get_captions_tags
from multiprocessing import cpu_count
import concurrent

from miditok import REMI, TokenizerConfig
from symusic import Score
import logging
import sys

from zipfile import ZipFile
import csv

from urllib.parse import unquote

import io

# Сначала отключаем логирование, ДО любых импортов библиотек обработки
# os.environ['SYMUSIC_LOG_LEVEL'] = '0' # Иногда 0, parfois OFF, depends on version
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Настройка логирования python
logging.getLogger("miditok").setLevel(logging.ERROR)

# --- Глобальная переменная для воркера ---
worker_tokenizer = None
worker_zips = {}

class MidiParser:
    def parse_midi_tokens(self, parsed_midi_meta: list[tuple]):
        print(len(parsed_midi_meta))

        to_process: list[tuple] = []

        processed_midi = set()

        if os.path.exists("../data/parsed_midi.jsonl"):
            with open("../data/parsed_midi.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    line_data = json.loads(line)
                    md5 = line_data.get('md5')
                    if md5 is not None:
                        processed_midi.add(md5)

        for track in parsed_midi_meta:
            md5 = track[0]
            if md5 not in processed_midi:
                to_process.append(track)

        total_files = len(to_process)
        print(f"К обработке: {len(to_process)} файлов")

        workers = max(1, cpu_count() - 1)

        with open("../data/parsed_midi.jsonl", "a", encoding="utf-8") as f:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=self.init_worker) as executor:
                results_iterator = executor.map(self.process_single_midi, to_process, chunksize=12)

                for i, result in enumerate(results_iterator):
                    if result:
                        f.write(json.dumps(result) + "\n")

                    if i % 1000 == 0:
                        print(f"Готово: {i}/{total_files} ({(i / total_files) * 100:.2f}%)")
                        f.flush()

    def init_worker(self):
        global worker_tokenizer

        # --- БЛОК ПОДАВЛЕНИЯ ВЫВОДА ---
        # Открываем "null" устройство (в никуда)
        devnull = open(os.devnull, 'w')

        # Перенаправляем stdout и stderr текущего процесса в null
        # Мы сохраняем оригинальные дескрипторы, если вдруг понадобятся (но в worker'е они не нужны)
        try:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
        except Exception:
            pass

        """Инициализация токенизатора один раз для каждого процесса"""
        # Создаем конфиг внутри процесса
        config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
        worker_tokenizer = REMI(config)
        worker_zips = {}

    def parse_tokens_in_ids_arr(self, ids, start_programs_token: int, now_instrument_id: int):
        tacts = []
        tact = {'prepare': []}

        for token in ids:
            # Когда новый такт
            if token == 4 and len(tact) > 1:
                for instrument in tact:
                    tact[instrument].append(2)
                    tact[instrument].insert(0, 1)

                del tact['prepare']
                tacts.append(tact.copy())

                del tact
                tact = {}
                tact['prepare'] = []
                now_instrument_id = -2
            if token >= start_programs_token:
                now_instrument_id = token - start_programs_token
                print(now_instrument_id)

                if now_instrument_id not in tact:
                    tact[now_instrument_id] = (tact['prepare']).copy()

            print(token, start_programs_token, token >= start_programs_token)
            if token >= start_programs_token:
                continue
            if now_instrument_id != -2:
                tact[now_instrument_id].append(token)
            elif token != 4:
                tact['prepare'].append(token)

        if len(tact) > 1:
            if 'prepare' in tact: del tact['prepare']  # если нужно
            tacts.append(tact.copy())

        return tacts

    def process_single_midi(self, track_data: dict):
        md5, path, indexes, instruments, prompt, tags = track_data
        global worker_tokenizer

        start_programs_token = 282
        now_instrument_id = -2

        try:
            if not os.path.exists(path):
                return None
            midi = Score(path)
            tokens = worker_tokenizer(midi)
            ids = []
            if hasattr(tokens, 'ids'):
                ids = tokens.ids
            elif isinstance(tokens, list):
                ids = tokens  # Если это уже список
            elif isinstance(tokens, dict) and 'ids' in tokens:
                ids = tokens['ids']
            else:
                return {"error": f"UNKNOWN_TOKEN_FORMAT: {type(tokens)}", "md5": md5}

            if not ids:
                return {"error": "EMPTY_TOKENS", "md5": md5}

            tacts = self.parse_tokens_in_ids_arr(ids, start_programs_token, now_instrument_id)

            json_line = {
                'file_path': path,
                "md5": md5,
                'prompt': indexes,
                'instruments': instruments,
                'tacts': tacts,
                'prompt_words': prompt,
                'prompt_tags': tags
            }
            return json_line

        except Exception as e:
            return {"error": f"ERROR: {str(e)}", "md5": md5}


class CsvParser:
    def __init__(self):
        self.artists_ban_list = {'Trad'}

        maxInt = sys.maxsize
        while True:
            try:
                csv.field_size_limit(maxInt)
                break
            except OverflowError:
                maxInt = int(maxInt / 10)

        self.styles_counter = Counter()
        self.artists_counter = Counter()

        self.tempo_re = re.compile(r'qpm=([\d.]+)')

        self.sorted_translation = sorted(list(all_translations.keys()), key=len, reverse=True)


    def parse_csv(self):
        prompt_words_indexes = self.init_alphabets()

        all_rows_list: list[tuple] = []
        counter = 0
        for i, row in enumerate(self.read_file('../data/meta/GigaMIDI-Dataset.csv')):
            if self.is_raw_valid(row):
                tags = self.get_music_tokens(row)
                prompt, indexes = self.get_midi_prompt(tags, prompt_words_indexes)

                split_path = str(row['file_path']).split('/')

                data_category = split_path[2].split('-')[0]
                drums_category = split_path[3]

                path = f"../data/{split_path[1]}/{data_category}-{drums_category}/{'/'.join(split_path[3:])}"

                md5 = str(row['md5'])

                type = int(row['instrument_category: drums-only: 0, all-instruments-with-drums: 1,no-drums: 2'])
                if type > 0:
                    instruments = str(row['NOMML'])
                else:
                    instruments = '[128]'

                all_rows_list.append(
                    ( md5, path, indexes, instruments, prompt, tags )
                )

                counter += 1
                if counter % 50000 == 0:
                    print(f"{counter} ({i})", prompt, indexes, tags, path, instruments, '', sep="\n")

        return all_rows_list

    def is_raw_valid(self, row: dict):
        if (len(row['music_style_scraped']) == 0 or
                len(row['title']) == 0 or
                int(row['num_tracks']) == 0 or
                int(row['total_notes']) < 100):
            return False
        return True

    def get_music_tokens(self, row):
        music_style_raw = unquote(str(row['music_style_scraped']))
        music_styles_tags = music_style_raw.split(',')
        self.styles_counter.update(music_styles_tags)

        artist = str(row['artist'])
        artist = ( artist
                  .replace('\n', ' ')
                  .replace(",", "")
                  .replace(".", "")
                  .replace(" ", "-")
                  .replace("'", "") )
        # artist = artist.encode('ascii', 'ignore').decode('ascii')  # удалит ударения (á -> a) и BOM

        self.artists_counter.update([artist])

        avg_velocity = float(row['avg_velocity']) # 68.2734375
        drums_type = str(row['Type']) # no-drums drums-only all-instruments-with-drums

        tempo_info = str(row['tempo']) #Tempo(time=0, qpm=118.99992463338107, mspq=504202, ttype='Tick')
        tempo_search = self.tempo_re.search(tempo_info)
        tempo = tempo_search.group(0).split("=")[1] if tempo_search is not None else -1
        tempo_val = float(tempo)

        avg_note_duration = float(row['avg_note_duration'])  # 308.4344005021971

        # velocity
        if avg_velocity < 40:
            velocity_tag = "very-quiet"
        elif avg_velocity < 60:
            velocity_tag = "quiet"
        elif avg_velocity < 85:
            velocity_tag = "moderate-loudness"
        elif avg_velocity < 105:
            velocity_tag = "loud"
        else:
            velocity_tag = "very-loud"

        # tempo
        if tempo_val < 65:
            tempo_tag = "very-slow-tempo"
        elif tempo_val < 90:
            tempo_tag = "slow-tempo"
        elif tempo_val < 125:
            tempo_tag = "moderate-tempo"
        elif tempo_val < 160:
            tempo_tag = "fast-tempo"
        else:
            tempo_tag = "very-fast-tempo"

        if avg_note_duration < 150:
            note_durations_tag = "staccato-style"
        elif avg_note_duration < 350:
            note_durations_tag = "short-notes"
        elif avg_note_duration < 600:
            note_durations_tag = "medium-length-notes"
        else:
            note_durations_tag = "long-legato-notes"

        prompt_tags = [
            velocity_tag,
            tempo_tag,
            note_durations_tag,
            drums_type,
            *music_styles_tags
        ]
        if artist not in self.artists_ban_list:
            prompt_tags.append(artist)

        filtered_tags = [item for item in prompt_tags if random.randint(0, 10) > 2]

        if len(filtered_tags) < 2:
            filtered_tags = prompt_tags

        return filtered_tags

    def get_midi_prompt(self, prompt_tags: list, prompt_words_indexes: dict[str, int]) -> tuple[str, list[str]]:
        random.shuffle(prompt_tags)
        exit_prompt = []
        for i, tag in enumerate(prompt_tags):
            clean_tag = tag.replace("-", " ").lower()

            if clean_tag in self.sorted_translation:
                exit_prompt.append(random.choice(all_translations[clean_tag]))
            else:
                parts = clean_tag.split()
                for part in parts:
                    if part in self.sorted_translation:
                        exit_prompt.append( random.choice(all_translations[part]) )

        prompt = " ".join(exit_prompt)

        indexes_prompt = []
        for word in prompt.split():
            if word in prompt_words_indexes:
                indexes_prompt.append(prompt_words_indexes[word])
        return prompt, indexes_prompt

    def init_alphabets(self) -> dict[str, int]:
        words_jsonl = "../data/words_alphabet.jsonl"
        midi_jsonl = "../data/midi_alphabet.jsonl"

        if os.path.exists(words_jsonl) and os.path.exists(midi_jsonl):
            with open(words_jsonl, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    return json.loads(lines[1])
            print("ФАЙЛ АЛФАВИТА ПУСТ!!!")

        alphabet_words = set()

        for key, values in all_translations.items():
            alphabet_words.add(key.lower())
            value_words = " ".join(values).split()
            for word in value_words:
                alphabet_words.add(word.lower())
        alphabet_words.add("<unk>")

        alph_list = list(alphabet_words)
        alph_list.sort()

        alph_dict = dict(enumerate(alph_list))
        reverse_alph_dict = {v: k for k, v in alph_dict.items()}

        with open(words_jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps(alph_dict, ensure_ascii=False) + "\n")
            f.write(json.dumps(reverse_alph_dict, ensure_ascii=False) + "\n")

        config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
        tokenizer = REMI(config)
        vocab = tokenizer.vocab
        revers_vocab = dict(zip(vocab.values(), vocab.keys()))

        with open(midi_jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps(vocab, ensure_ascii=False) + "\n")
            f.write(json.dumps(revers_vocab, ensure_ascii=False) + "\n")

        return reverse_alph_dict

    # '../data/meta/GigaMIDI-Dataset.csv'
    def read_file(self, path):
        with open(path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
            for row in reader:
                yield row

def main():
    print("ЗАПУЩЕН ПРОЦЕСС ПРЕДПОДГОТОВКИ ДАННЫХ")
    print("\nПарсинг метаданных из CSV файла")
    metadata = CsvParser().parse_csv()
    print("\nПарсинг токенов из midi файлов")
    MidiParser().parse_midi_tokens(metadata)
    print("Парсинг данных завершен")

if __name__ == "__main__":
    main()

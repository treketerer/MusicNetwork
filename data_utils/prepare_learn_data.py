import json
import random
import os
import re
from collections import Counter

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

# Сначала отключаем логирование, ДО любых импортов библиотек обработки
# os.environ['SYMUSIC_LOG_LEVEL'] = '0' # Иногда 0, parfois OFF, depends on version
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Настройка логирования python
logging.getLogger("miditok").setLevel(logging.ERROR)

# --- Глобальная переменная для воркера ---
worker_tokenizer = None

def init_worker():
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

def get_midi_prompt(prompts_tags):
    midi_prompt = {}
    # with open('../data/midi_prompt_meta_keys.jsonl', 'r', encoding='utf-8') as f:
    print(len(prompts_tags))
    for i, items in enumerate(prompts_tags):
        track = items.get('md5')
        keys = items.get('prompt_keys')
        prompt_keys = keys.replace("_"," ").split()
        instruments_arr = json.loads(keys[keys.index("inst:")+5:])
        random.shuffle(prompt_keys)

        exit_prompt = []

        for key in prompt_keys:
            prompt_form = ""
            if key in all_translations:
                prompt_form = random.choice(all_translations[key])
            exit_prompt.append(prompt_form)
        midi_prompt[track] = (track, ' '.join(exit_prompt).lower(), instruments_arr)

        if i % 1000000 == 0:
            print(f"{i}/{len(prompts_tags)}")

    return midi_prompt

def init_alphabets():
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

    with open("../data/words_alphabet.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(alph_dict, ensure_ascii=False) + "\n")
        f.write(json.dumps(reverse_alph_dict, ensure_ascii=False) + "\n")

    config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
    tokenizer = REMI(config)
    vocab = tokenizer.vocab
    revers_vocab = dict(zip(vocab.values(), vocab.keys()))

    with open("../data/midi_alphabet.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(vocab, ensure_ascii=False) + "\n")
        f.write(json.dumps(revers_vocab, ensure_ascii=False) + "\n")

    return alph_list

def parse_tokens_in_ids_arr(ids, start_programs_token: int, now_instrument_id: int):
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

def process_single_midi(data: list):
    md5, _, prompt_idx, instruments_arr = data
    global worker_tokenizer

    start_programs_token = 282
    now_instrument_id = -2

    try:
        path = f"../data/GigaMIDI.zip/"
        if not os.path.exists(path):
            return None

        with ZipFile(path) as myzip:
            with myzip.open('') as midifile:

                midi = Score(midifile.read())
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

                tacts = parse_tokens_in_ids_arr(ids, start_programs_token, now_instrument_id)
                json_line = {"md5": md5, "tokens": tacts}
                return json_line

    except Exception as e:
        print(e)
        return {"error": f"UNHANDLED_EXCEPTION: {e}", "md5": md5}


class CsvParser:
    def __init__(self):
        maxInt = sys.maxsize
        while True:
            try:
                csv.field_size_limit(maxInt)
                break
            except OverflowError:
                maxInt = int(maxInt / 10)

        self.counter = Counter()

        for i, line in enumerate(self.read_file('../data/meta/GigaMIDI-Dataset.csv')):
            if self.is_raw_valid(line):
                self.get_music_tokens(line)
                if i % 100000 == 0:
                    print(i)
        print(self.counter)


    def is_raw_valid(self, row: dict):
        if (len(row['music_style_scraped']) == 0 or
                len(row['title']) == 0 or
                int(row['num_tracks']) == 0 or
                int(row['total_notes']) < 100):
            return False
        return True

    def get_music_tokens(self, row):
        file_path = str(row['file_path'])
        training_type = file_path.split('/')[2].split('-')[0]  # training-V1.1-80% validation-V1.1-10% test-V1.1-10%

        music_style_raw = unquote(str(row['music_style_scraped']))
        music_styles_tags = music_style_raw.split(',')
        self.counter.update(music_styles_tags)

        artist = str(row['artist'])

        avg_velocity = float(row['avg_velocity']) # 68.2734375
        drums_type = str(row['Type']) # no-drums drums-only all-instruments-with-drums

        tempo_info = str(row['tempo']) #Tempo(time=0, qpm=118.99992463338107, mspq=504202, ttype='Tick')
        tempo_search = re.search(r'qpm=([\d.]+)', tempo_info)
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

        prompt_tags = f'{artist.replace(",", "").replace(" ", "-")} {velocity_tag} {tempo_tag} {note_durations_tag} {drums_type} {" ".join(music_styles_tags)}'

        return prompt_tags

    # '../data/meta/GigaMIDI-Dataset.csv'
    def read_file(self, path):
        with open(path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
            for row in reader:
                yield row

    #
    # def parse_midi_tokens(md5_paths, midi_prompts):
    #     parsed_files = set()
    #
    #
    #
    #     base_path = "C:/Project/MusicNetwork/midi/"
    #     to_process = []
    #
    #     for md5 in md5_paths:
    #         if md5 not in parsed_files and os.path.exists(f"{base_path}{md5}.mid"):
    #             to_process.append((md5, *midi_prompts[md5]))
    #
    #     total_files = len(to_process)
    #     print(f"К обработке: {len(to_process)} файлов")
    #
    #     workers = max(1, cpu_count() - 1)
    #
    #
    #
    #
    #
    #
    #     with open("../data/parsed_midi.jsonl", "a", encoding="utf-8") as f:
    #         with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:
    #             results_iterator = executor.map(process_single_midi, to_process, chunksize=10)
    #
    #             for i, result in enumerate(results_iterator):
    #                 if result:
    #                     f.write(json.dumps(result) + "\n")
    #
    #                 if i % 1000 == 0:
    #                     print(f"Готово: {i}/{total_files} ({(i / total_files) * 100:.2f}%)")
    #                     f.flush()


def main():
    print("ЗАПУЩЕН ПРОЦЕСС ПРЕДПОДГОТОВКИ ДАННЫХ")
    CsvParser()



    # all_md5, prompts_tags = get_captions_tags()
    #
    # all_md5 = set()
    # prompts_tags = []
    # with open("../data/prompt_tags.jsonl", "r", encoding="utf-8") as f:
    #     for i, line in enumerate(f):
    #         data = json.loads(line)
    #         all_md5.add(data.get('md5'))
    #         # prompts_tags.append(data)
    #         if i >= 70000:
    #             break
    #
    # print("MIDI ПРОМПТ ТЕГИ ИНИЦИАЛИЗИРОВАНЫ!")
    #
    # init_alphabets()
    # print("АЛФАВИТЫ СЛОВ и MIDI ТЭГОВ ИНИЦИАЛИЗИРОВАНЫ!")
    #
    # print("\nЗапущен процесс составления промптов!")
    # midi_prompts = get_midi_prompt(prompts_tags)
    # del prompts_tags
    # print("ПРОМПТЫ СОСТАВЛЕНЫ!")
    # # save_prompts(midi_prompts)
    # # print("ПРОМПТЫ СОХРАНЕНЫ В ФАЙЛ!")
    #
    # all_md5_list = list(all_md5)[:70000]
    # del all_md5
    # print("Запуск токенизации MIDI файлов!")
    # parse_midi_tokens(all_md5, midi_prompts)
    # print("MIDI ФАЙЛЫ ТОКЕНИЗИРОВАННЫ!")


if __name__ == "__main__":
    main()

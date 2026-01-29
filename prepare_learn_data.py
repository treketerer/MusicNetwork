import json
import random

from data.keywords import prompts_by_key
from meta.metaparser import get_prompts_tags

import os

from multiprocessing import cpu_count
import concurrent

# Сначала отключаем логирование, ДО любых импортов библиотек обработки
# os.environ['SYMUSIC_LOG_LEVEL'] = '0' # Иногда 0, parfois OFF, depends on version
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from miditok import REMI, TokenizerConfig
from symusic import Score
import logging
import sys

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
    for items in prompts_tags:
        track = items.get('md5')

        prompt_keys = items.get('prompt_keys').split()
        random.shuffle(prompt_keys)

        exit_prompt = []

        for key in prompt_keys:
            prompt_form = ""
            if key in prompts_by_key:
                prompt_form = random.choice(prompts_by_key[key])
            exit_prompt.append(prompt_form)
        midi_prompt[track] = ' '.join(exit_prompt).lower()
    return midi_prompt


def save_prompts(midi_prompt):
    words_prompts_file = open("./data/midi_words_prompts.jsonl", 'w', encoding='utf-8')
    idx_prompts_file = open("./data/midi_idx_prompts.jsonl", 'w', encoding='utf-8')

    alphabet_file = open("./data/words_alphabet.jsonl", 'r', encoding='utf-8')
    alphabet_file.readline()
    raw_words_alphabet = alphabet_file.readline()
    words_alphabet = json.loads(raw_words_alphabet)
    alphabet_file.close()

    for md5, prompt in midi_prompt.items():
        json_word_line = {
            "md5": md5,
            "prompt_keys": prompt
        }

        idx_prompt = []
        for word in prompt.split():
            if word in words_alphabet:
                idx_prompt.append(words_alphabet[word])
            else:
                idx_prompt.append(words_alphabet["<unk>"])

        json_idx_line = {
            "md5": md5,
            "prompt_keys": idx_prompt
        }

        words_prompts_file.write(json.dumps(json_word_line, ensure_ascii=False) + "\n")
        idx_prompts_file.write(json.dumps(json_idx_line, ensure_ascii=False) + "\n")


def init_embeddings():
    alphabet_words = set()

    for key, values in prompts_by_key.items():
        # print(key, values)
        alphabet_words.add(key.lower())
        value_words = " ".join(values).split()
        for word in value_words:
            alphabet_words.add(word.lower())
    alphabet_words.add("<unk>")

    alph_list = list(alphabet_words)
    alph_list.sort()

    alph_dict = dict(enumerate(alph_list))
    reverse_alph_dict = {v: k for k, v in alph_dict.items()}

    with open("./data/words_alphabet.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(alph_dict, ensure_ascii=False) + "\n")
        f.write(json.dumps(reverse_alph_dict, ensure_ascii=False) + "\n")

    return alph_list

def process_single_midi(md5):
    global worker_tokenizer
    try:
        path = f"C:/Project/MusicNetwork/midi/{md5}.mid"
        if not os.path.exists(path):
            return None

        midi = Score(path)
        tokens = worker_tokenizer(midi)
        ids = None
        if hasattr(tokens, 'ids'):
            ids = tokens.ids
        elif isinstance(tokens, list):
            ids = tokens  # Если это уже список
        elif isinstance(tokens, dict) and 'ids' in tokens:
            ids = tokens['ids']
        else:
            # Пытаемся понять, что вернул токенизатор
            return {"error": f"UNKNOWN_TOKEN_FORMAT: {type(tokens)}", "md5": md5}

        if not ids:
            return {"error": "EMPTY_TOKENS", "md5": md5}

        json_line = {"md5": md5, "tokens": ids}
        return json_line
    except Exception as e:
        return {"error": f"UNHANDLED_EXCEPTION: {str(e)}", "md5": md5}

def parse_midi_tokens(md5_paths):
    parsed_files = set()
    if os.path.exists("./data/parsed_midi.jsonl"):
        with open("./data/parsed_midi.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                parsed_files.add(data.get('md5'))
            f.close()

    to_process = [md5 for md5 in md5_paths if md5 not in parsed_files]
    total_files = len(to_process)
    print(f"К обработке: {len(to_process)} файлов")

    workers = max(1, cpu_count() - 1)

    with open("./data/parsed_midi.jsonl", "a", encoding="utf-8") as f:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:
            results_iterator = executor.map(process_single_midi, to_process, chunksize=10)

            for i, result in enumerate(results_iterator):
                if result:
                    f.write(json.dumps(result) + "\n")

                if i % 1000 == 0:
                    print(f"Готово: {i}/{total_files} ({(i / total_files) * 100:.2f}%)")
                    f.flush()

def main():
    print("ЗАПУЩЕН ПРОЦЕСС ПРЕДПОДГОТОВКИ ДАННЫХ")
    all_md5, prompts_tags = get_prompts_tags()
    print("MIDI ПРОМПТ ТЕГИ ИНИЦИАЛИЗИРОВАНЫ!")
    init_embeddings()
    print("АЛФАВИТЫ СЛОВ ИНИЦИАЛИЗИРОВАНЫ!")
    midi_prompts = get_midi_prompt(prompts_tags)
    print("ПРОМПТЫ СОСТАВЛЕНЫ!")
    save_prompts(midi_prompts)
    print("ПРОМПТЫ СОХРАНЕНЫ В ФАЙЛ!")

    # parse_midi_tokens(all_md5)
    # print("MIDI ФАЙЛЫ ПРЕОБРАЗОВАНЫ В АЛФАВИТ!")

if __name__ == "__main__":
    main()
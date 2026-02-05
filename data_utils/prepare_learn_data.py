import json
import random
import os

from IPython.terminal.shortcuts.auto_match import brackets

from data.keywords import all_translations
from metaparser import get_captions_tags
from multiprocessing import cpu_count
import concurrent

from miditok import REMI, TokenizerConfig
from symusic import Score
import logging
import sys


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
    midi_prompt = []
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
        midi_prompt.append((track, ' '.join(exit_prompt).lower(), instruments_arr))

        if i % 1000000 == 0:
            print(f"{i}/{len(prompts_tags)}")

    return midi_prompt


def save_prompts(midi_prompt):
    words_prompts_file = open("../data/midi_words_prompts.jsonl", 'w', encoding='utf-8')
    idx_prompts_file = open("../data/midi_idx_prompts.jsonl", 'w', encoding='utf-8')

    alphabet_file = open("../data/words_alphabet.jsonl", 'r', encoding='utf-8')
    alphabet_file.readline()
    raw_words_alphabet = alphabet_file.readline()
    words_alphabet = json.loads(raw_words_alphabet)
    alphabet_file.close()

    for md5, prompt, instruments in midi_prompt:
        json_word_line = {
            "md5": md5,
            "prompt_keys": ' '.join(prompt.split())
        }

        idx_prompt = []
        for word in prompt.split():
            if word in words_alphabet:
                idx_prompt.append(words_alphabet[word])
            else:
                idx_prompt.append(words_alphabet["<unk>"])

        json_idx_line = {
            "md5": md5,
            "prompt_keys": idx_prompt,
            "instruments": instruments
        }

        words_prompts_file.write(json.dumps(json_word_line, ensure_ascii=False) + "\n")
        idx_prompts_file.write(json.dumps(json_idx_line, ensure_ascii=False) + "\n")


def init_alphabets():
    alphabet_words = set()

    for key, values in all_translations.items():
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

def process_single_midi(md5):
    global worker_tokenizer
    start_programs_token = 282
    tacts = []
    tact = {'prepare': []}
    now_instrument_id = -2

    try:
        path = f"../midi/{md5}.mid"
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

        print(ids)

        for token in ids:
            # Когда новый такт
            if token == 4 and len(tact) > 1:
                del tact['prepare']
                tacts.append(tact.copy())

                del tact
                tact = {}
                tact['prepare'] = []
                now_instrument_id = -2
            if token >= start_programs_token:
                now_instrument_id = token - start_programs_token
                print(now_instrument_id)
                if now_instrument_id == 128:
                    now_instrument_id = -1

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

        json_line = {"md5": md5, "tokens": tacts}
        return json_line
    except Exception as e:
        print(e)
        return {"error": f"UNHANDLED_EXCEPTION: {e} {tact} {tacts}", "md5": md5}

def parse_midi_tokens(md5_paths):
    parsed_files = set()
    if os.path.exists("../data/parsed_midi.jsonl"):
        with open("../data/parsed_midi.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                parsed_files.add(data.get('md5'))
            f.close()

    base_path = "C:/Project/MusicNetwork/midi/"
    to_process = []

    for md5 in md5_paths:
        if md5 not in parsed_files and os.path.exists(f"{base_path}{md5}.mid"):
            to_process.append(md5)

    total_files = len(to_process)
    print(f"К обработке: {len(to_process)} файлов")

    workers = max(1, cpu_count() - 1)

    with open("../data/parsed_midi.jsonl", "a", encoding="utf-8") as f:
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
    # all_md5, prompts_tags = get_captions_tags()

    all_md5 = set()
    prompts_tags = []
    with open("../data/prompt_tags.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            all_md5.add(data.get('md5'))
            # prompts_tags.append(data)
            if i >= 70000:
                break

    # print("MIDI ПРОМПТ ТЕГИ ИНИЦИАЛИЗИРОВАНЫ!")
    #
    # init_alphabets()
    # print("АЛФАВИТЫ СЛОВ и MIDI ТЭГОВ ИНИЦИАЛИЗИРОВАНЫ!")
    #
    # print("\nЗапущен процесс составления промптов!")
    # midi_prompts = get_midi_prompt(prompts_tags)
    # del prompts_tags
    # print("ПРОМПТЫ СОСТАВЛЕНЫ!")
    # save_prompts(midi_prompts)
    # print("ПРОМПТЫ СОХРАНЕНЫ В ФАЙЛ!")
    #
    # all_md5_list = list(all_md5)[:70000]
    # del midi_prompts
    # del all_md5
    print("Запуск токенизации MIDI файлов!")
    parse_midi_tokens(all_md5)
    print("MIDI ФАЙЛЫ ТОКЕНИЗИРОВАННЫ!")

if __name__ == "__main__":
    main()
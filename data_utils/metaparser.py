import json
from collections import Counter
import re

# from data.keywords import prompts_by_key

# ФАЙЛ ДЛЯ ПРЕОБРАЗОВАНИЯ МЕТА ДАННЫХ В ТЕГИ ПРОМПТОВ

# Функция для ручного отбора тегов промптов

def get_bigrams(split_string):
    # Создаем пары: (слово1, слово2), (слово2, слово3)
    bigrams = []
    ban_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'and', 'in', 'on', 'a', 'the']
    for i in range(len(split_string) - 1):
        if (split_string[i] not in ban_list) and (split_string[i + 1]not in ban_list):
            bigrams.append(f"{split_string[i]} {split_string[i + 1]}")
    return bigrams

def get_unique_words():
    min_frequency = 30

    counts = Counter()

    print("Старт get_unique_words")
    with open('../data/meta/identified_midis_data.jsonl', 'r', encoding='utf-8') as f:
    # with open('../data/meta/all_midis_text_captions.jsonl', 'r', encoding='utf-8') as f:

        for i, line in enumerate(f):
            items = json.loads(line)
            now_item = items.get('music_description')
            now_item += f" {items.get('genre')}"
            now_item += f" {items.get('style')}"
            now_item += f" {items.get('mood')}"
            now_item = re.sub(r'[^\w\s-]', '', now_item)
            now_item = now_item.lower().split()

            # Считаем пары слов
            counts.update(get_bigrams(now_item))

            # for tag in now_item:
            #     # if len(tag) > 20: continue
            #     if not tag: continue
            #     counts[tag] += 1

            if i % 100000 == 0:
                print(i, "строк обработано")

        final_vocab = [word for word, freq in counts.items() if freq >= min_frequency]
        print(f"Стало слов: {len(final_vocab)}")
        return sorted(final_vocab)

print(get_unique_words())

def get_base_meta():
    prompts_by_key = {}
    wordslist = {}
    with open('../data/meta/all_midis_meta_data.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            now_line = set()
            items = json.loads(line)
            track = items.get('md5')
            pre_item = items.get('metadata').split('\n')[2:]
            now_item = " ".join(pre_item).split()

            for word in now_item:
                if word in prompts_by_key:
                    now_line.add(word)
            wordslist[track] = " ".join(now_line)
    return wordslist

def get_acoustic_meta():
    wordslist = {}
    with open('../data/meta/all_midis_averages.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            items = json.loads(line)
            track = items.get('md5')
            avgs = items.get('avgs')

            tempo = avgs[0]
            velocity = avgs[1]

            if tempo < 80:
                word_tempo = 'Tempo_Slow'
            elif tempo < 120:
                word_tempo = "Tempo_Normal"
            elif tempo < 160:
                word_tempo = "Tempo_Fast"
            else:
                word_tempo = "Tempo_VeryFast"

            if velocity < 30:
                word_velocity = 'Velocity_Pianissimo'
            elif velocity < 60:
                word_velocity = "Velocity_Piano"
            elif velocity < 90:
                word_velocity = "Velocity_Forte"
            else:
                word_velocity = "Velocity_MezzoForte"

            wordslist[track] = [word_tempo, word_velocity]
    return wordslist

def get_prompts_tags():
    # get_unique_words()

    base_meta = get_base_meta()
    print("ПОЛУЧЕНЫ БАЗОВЫЕ МЕТАДАННЫЕ")
    ac_meta = get_acoustic_meta()
    print("ПОЛУЧЕНЫ АКУСТИЧЕСКИЕ МЕТАДАННЫЕ")

    prompts_keys_for_all_midi = []

    all_md5 = []
    for md5, key in base_meta.items():
        bm = base_meta.get(md5)
        am = ac_meta.get(md5)

        prompt_keys = bm
        if am is not None:
            prompt_keys += f" {am[0]} {am[1]}"

        prompt_keys = prompt_keys.strip()

        if not prompt_keys:
            continue

        json_line = {
            "md5": md5,
            "prompt_keys": prompt_keys
        }
        all_md5.append(md5)
        prompts_keys_for_all_midi.append(json_line)
        # f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    return all_md5, prompts_keys_for_all_midi
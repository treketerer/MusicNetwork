import json
from collections import Counter
import re

from content.project.data.keywords import prompts_by_key

# ФАЙЛ ДЛЯ ПРЕОБРАЗОВАНИЯ МЕТА ДАННЫХ В ТЕГИ ПРОМПТОВ

# очистка данных
def clean_vocabulary(all_tags_list, min_frequency=10):
    cleaned_list = []
    for tag in all_tags_list:
        # Убираем теги короче 2 символов и длиннее 20
        if len(tag) > 20: continue
        # Убираем теги, где есть перемешанные цифры и буквы (похоже на ID)
        if re.search(r'\d', tag) and re.search(r'[a-zA-Z]', tag): continue
        # Убираем расширения файлов
        if tag.endswith(('.mxl', '.mid', '.midi')): continue
        cleaned_list.append(tag)

    # 2. Считаем частоту каждого слова
    counts = Counter(cleaned_list)

    # 3. Оставляем только те, что встречаются хотя бы N раз
    # Это автоматически выкинет редких авторов и опечатки
    final_vocab = [word for word, freq in counts.items() if freq >= min_frequency]

    print(f"Было слов: {len(set(all_tags_list))}")
    print(f"Стало слов: {len(final_vocab)}")

    return sorted(final_vocab)

# Функция для ручного отбора тегов промптов
def get_unique_words():
    wordslist = []
    with open('./meta/all_midis_meta_data.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            items = json.loads(line)
            now_item = items.get('metadata').split('\n')[2:]
            for word in now_item:
                wordslist.append(word)

    print("FINISHED")
    print(clean_vocabulary(wordslist, 50))

def get_base_meta():
    wordslist = {}
    with open('./meta/all_midis_meta_data.jsonl', 'r', encoding='utf-8') as f:
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
    with open('./meta/all_midis_averages.jsonl', 'r', encoding='utf-8') as f:
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
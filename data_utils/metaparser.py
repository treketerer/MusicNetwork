import datetime
import json
from collections import Counter
import re
from itertools import zip_longest
from wsgiref.util import request_uri

from data.keywords import ban_list, all_instruments, all_translations

from multiprocessing import cpu_count, Pool
import concurrent.futures

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

            for tag in now_item:
                # if len(tag) > 20: continue
                if not tag: continue
                counts[tag] += 1

            if i % 100000 == 0:
                print(i, "строк обработано")

        final_vocab = [word for word, freq in counts.items() if freq >= min_frequency]
        print(f"Стало слов: {len(final_vocab)}")
        return sorted(final_vocab)



def join_tags(words, instruments):
    caps = []
    i = 0
    n = len(words[:-1])
    while i < n:
        if f'{words[i]} {words[i + 1]}' in all_translations:
            caps.append(f'{words[i]}_{words[i + 1]}')
            i += 1
        elif words[i] == "features" and i + 2 < n:
            caps.append(f'{words[i]}_{words[i + 1]}_{words[i + 2]}')
            i += 2
        elif words[i] in all_translations:
            caps.append(words[i])
        i += 1

    if len(caps) == 0:
        return None

    caps = [word for word in caps if word not in ban_list]

    if caps[0] == "a":
        caps = caps[1:]

    return " ".join(caps) + f" inst:{instruments}"

def get_line_json(line):
    try:
        json_line = json.loads(line)
        if json_line.get('genre') is not None:
            md5 = json_line.get('md5')
            now_desc = json_line.get('music_description')

            genre = json_line.get('genre')
            style = json_line.get('style')
            mood = json_line.get('mood')

            if genre:
                genre = genre.replace(' ', '_')
            if style:
                style = style.replace(' ', '_')
            if mood:
                mood = mood.replace(' ', '_')
                now_desc += f" {mood}"
            if genre != style:
                now_desc += f" {genre} {style}"

            return (md5, now_desc)

        # Второй файл
        if json_line:
                return json_line.get('md5'), json_line.get('caps')
        return None
    except:
        return None


def __parse_tags__(task):
    md5, caps = get_line_json(task)

    parsed_cap = re.sub(r'[^a-z0-9\s#()]', '', caps.lower())

    instruments = []
    for i in range(len(all_instruments)):
        if all_instruments[i].lower() in parsed_cap:
            instruments.append(i)
            parsed_cap = parsed_cap.replace(all_instruments[i], '')

    words = parsed_cap.split()
    tags = join_tags(words, instruments)

    if tags is None:
        return None

    return {
        "md5": md5,
        "prompt_keys": tags
    }

def parse_files(file1, file2, batch_size = 5000):
    start_global = datetime.datetime.now()

    batch = []
    batch_counts = 0
    with open(file1, 'r', encoding='utf-8') as f1, \
          open(file2, 'r', encoding='utf-8') as f2:
        start = datetime.datetime.now()
        for line1, line2 in zip_longest(f1, f2, fillvalue=None):
            if len(batch) >= batch_size:
                # print(batch)
                finish = datetime.datetime.now()
                print(f"Обработано {batch_size*(batch_counts+1)} строк, на {batch_size} ушло {str(finish - start)}")
                start = datetime.datetime.now()

                yield batch
                batch = []
                batch_counts += 1

            if line1:
                batch.append(line1)
            if line2:
                batch.append(line2)

    if batch:
        yield batch
    finish_global = datetime.datetime.now()
    print("Обработка завершена", str(finish_global - start_global))

# функция создания воркеров для обратки строк файлов
def get_captions_tags():
    file1 = '../data/meta/identified_midis_data.jsonl'
    file2 = '../data/meta/all_midis_text_captions.jsonl'
    output_file = '../data/prompt_tags.jsonl'

    num_workers = 10

    print("Старт потоковой обработки...")

    all_md5 = set()
    all_tags = []

    with open(output_file, 'w', encoding='utf-8') as out_f:
        with Pool(num_workers) as pool:
            for batch in parse_files(file1, file2, batch_size=100000):
                results = pool.imap(
                    __parse_tags__,
                    batch,
                    chunksize=10000
                )
                if results is not None:
                    for result in results:
                        if result is not None and result['md5'] not in all_md5:
                            all_md5.add(result['md5'])
                            all_tags.append(result)
                            out_f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Готово!")
    return all_md5, all_tags

if __name__ == "__main__":
    get_captions_tags()
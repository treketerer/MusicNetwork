import json
import random

from data.keywords import prompts_by_key
from meta.metaparser import get_prompts_tags

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
    words_prompts_file = open("../data/midi_words_prompts.jsonl", 'w', encoding='utf-8')
    idx_prompts_file = open("../data/midi_idx_prompts.jsonl", 'w', encoding='utf-8')

    alphabet_file = open("../data/words_alphabet.jsonl", 'r', encoding='utf-8')
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
        alphabet_words.add(key)
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

    return alph_list

def main():
    print("ЗАПУЩЕН ПРОЦЕСС ПРЕДПОДГОТОВКИ ДАННЫХ")
    prompts_tags = get_prompts_tags()
    print("MIDI ПРОМПТ ТЕГИ ИНИЦИАЛИЗИРОВАНЫ!")
    init_embeddings()
    print("АЛФАВИТЫ СЛОВ ИНИЦИАЛИЗИРОВАНЫ!")
    midi_prompts = get_midi_prompt(prompts_tags)
    print("ПРОМПТЫ СОСТАВЛЕНЫ!")
    save_prompts(midi_prompts)
    print("ПРОМПТЫ СОХРАНЕНЫ В ФАЙЛ!")

if __name__ == "__main__":
    main()
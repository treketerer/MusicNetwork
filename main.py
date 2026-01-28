import datetime
import json
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor
import random

torch.set_num_threads(1)

from music_nn import MusicNN

def generate_batches(batch_size):
    all_md5s = []
    all_words = []
    all_idx = []

    with open("../data/midi_words_prompts.jsonl", 'r', encoding='utf-8') as f_words, \
        open("../data/midi_idx_prompts.jsonl", 'r', encoding='utf-8') as f_idx:
        for word_line, idx_line in zip(f_words, f_idx):
            words_data = json.loads(word_line)
            idx_data = json.loads(idx_line)

            if words_data['md5'] == idx_data['md5']:
                all_md5s.append(words_data['md5'])
                all_words.append(words_data['prompt_keys'])
                all_idx.append(idx_data['prompt_keys'])

    print(f"Всего загружено {len(all_md5s)} пар данных.")

    # Перемешивание данных
    indices = list(range(len(all_md5s)))
    random.shuffle(indices)

    # Применяем перемешанные индексы к нашим данным
    shuffled_md5s = [all_md5s[i] for i in indices]
    shuffled_word_prompts = [all_words[i] for i in indices]
    shuffled_idx_prompts = [all_idx[i] for i in indices]

    # Нарезка на батчи
    batches = []
    batches_count = math.floor(len(all_md5s) / batch_size)

    for i in range(batches_count):
        from_idx = i * batch_size
        to_idx = (1+i) * batch_size

        batch_data = {
            'md5': shuffled_md5s[from_idx:to_idx],
            'shuffled_word_prompts': shuffled_word_prompts[from_idx:to_idx],
            'shuffled_idx_prompts': shuffled_idx_prompts[from_idx:to_idx]
        }
        batches.append(batch_data)

    print(f"Данные нарезаны на {batches_count} батчей")
    return batches

def learn_model(model, loss_function, optimizer, iterations_count, batch_size):
    # Обучение
    batches = generate_batches(batch_size)
    for i in range(iterations_count):
        

        optimizer.zero_grad()
        output = model(torch.stack(input_batch))

        current_loss = loss_function(output, torch.stack(learn_batch))
        current_loss.backward()
        optimizer.step()

        model.bake_memory()

        if (i + 1) % 1000 == 0:
            print(f"\nВвод: {now_control_inp}\nОжидаемое: {now_control_word}\nОтвет сети:",
                  model.alphabet_words[torch.argmax(output[0])])
            print(f'Iteration {i}, Loss: {current_loss.item():.4f}')

def use_model(model):
    # Использование
    while True:
        print("Введите предложение: ")
        words = input()
        words += " <unk>"
        if not words: break

        input_str, _ = model.parse_learn_couple(words)
        input_tensor = model.parse_words_idx(input_str)

        with torch.no_grad():
            output = model(input_tensor.unsqueeze(0))

        print("Ответ:", model.alphabet_words[torch.argmax(output).item()])

def main():
    LEARNING_RATE = 0.0001
    NUM_ITERATIONS = 5000

    music_model = MusicNN()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(music_model.parameters(), lr=LEARNING_RATE)

    start = datetime.datetime.now()
    learn_model(music_model, loss_function, optimizer, NUM_ITERATIONS)
    finish = datetime.datetime.now()
    print('Время работы: ' + str(finish - start))

    use_model(music_model)

if __name__ == "__main__":
    main()

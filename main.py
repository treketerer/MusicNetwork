import datetime
import json
import math
import random
from itertools import islice

import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor
from torch.utils.data import DataLoader

from music_nn import MusicNN
from dataset import MusicStreamingDataset

from miditok import REMI, TokenizerConfig


print("\nWORKER INITIALIZED")
torch.set_num_threads(8)

NEED_TO_LEARN = False
SAVED_MODEL_PATH = "./models/6841_music_model_1.pth"

def learn_model(model, dataset, loss_function, optimizer, epochs_count, batch_size, save_model_id):
    # Обучение
    print("ОБУЧЕНИЕ НАЧАЛОСЬ!")
    current_loss = None
    for epoch in range(epochs_count):
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=2,  # <--- Задействуем 8 ядер для чтения и парсинга
            persistent_workers=True,  # <--- Не убивать воркеры после эпохи (важно для Windows!)
            prefetch_factor=2,  # <--- Каждый воркер заранее готовит по 2 батча
            collate_fn=dataset.collate_fn,
            pin_memory=True  # <--- Ускоряет передачу данных (особенно если есть GPU)
        )

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            prompts = batch.get('idx_prompts')
            tokens = batch.get('tokens')

            inp = tokens[:, :-1]
            target = tokens[:, 1:]

            logits, _, _ = model(prompts, inp, None, None)

            current_loss = loss_function(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            current_loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                # print(f"\nВвод: {now_control_inp}\nОжидаемое: {now_control_word}\nОтвет сети:",
                #       model.alphabet_words[torch.argmax(output[0])])
                print(f'Epoch {epoch} Iteration {i+1}, Loss: {current_loss.item():.4f}')

        print(f"Эпоха {epoch} завершена!")

        path = f'./models/{save_model_id}_music_model_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': current_loss.item(),
        }, path)
        print(f"Модель сохранена в {path}")

def use_model(model, dataset):
    # Использование
    config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
    tokenizer = REMI(config)
    print("Токенайзер инициализирован!")

    while True:
        print("Введите предложение: ")
        words = input()
        if not words: break

        words_idx = dataset.words_to_idx(words)
        print(words_idx)
        input_tensor = tensor([words_idx], dtype=torch.long)

        outputs_tokens = []
        h, c = None, None

        current_token = torch.tensor([[0]], dtype=torch.long)

        with torch.no_grad():
            for _ in range(1000):
                logits, h, c = model(input_tensor, current_token, h, c)

                temperature = 1.4
                top_k = 10

                next_token_logits = logits[0, -1, :] / temperature
                threshold = torch.topk(next_token_logits, top_k).values[-1]
                next_token_logits[next_token_logits < threshold] = -float('Inf')
                probs = torch.softmax(next_token_logits, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)

                token_val = next_token.item()
                outputs_tokens.append(token_val)

                # Обновляем current_token для следующего шага
                current_token = next_token.unsqueeze(0)

        try:
            generated_sequence = tokenizer.decode(outputs_tokens)
            print(generated_sequence)
            output_path = f"./output/gen_{random.randint(0, 1000)}.mid"
            generated_sequence.dump_midi(output_path)
            print(f"Файл сохранен: {output_path}")
        except Exception as e:
            print(f"Ошибка сохранения MIDI: {e}")

def main():
    LEARNING_RATE = 0.0001
    EPOCHS_COUNT = 2
    BATCH_SIZE = 24

    dataset = MusicStreamingDataset(
        "./data/midi_words_prompts.jsonl",
        "./data/midi_idx_prompts.jsonl",
        "./data/words_alphabet.jsonl",
        "./data/parsed_midi.jsonl",
        buffer_size=256
    )

    print("Датасет инициализирован!")

    music_model = MusicNN(dataset.get_words_alphabet_len(), dataset.get_midi_alphabet_len())
    print("Модель инициализирована!")

    if NEED_TO_LEARN:
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(music_model.parameters(), lr=LEARNING_RATE)
        print("Оптимизаторы инициализированы!")

        start = datetime.datetime.now()
        print("ОБУЧЕНИЕ НАЧАЛОСЬ!")
        learn_model(music_model, dataset, loss_function, optimizer, EPOCHS_COUNT, BATCH_SIZE, random.randint(1111, 9999))
        finish = datetime.datetime.now()
        print('Обучение завершено!\nВремя работы: ' + str(finish - start))
    else:
        checkpoint = torch.load(SAVED_MODEL_PATH, weights_only=False)
        music_model.load_state_dict(checkpoint['model_state_dict'])
        music_model.eval()  # Обязательно для использования

    use_model(music_model, dataset)

if __name__ == "__main__":
    main()










# def parse_learn_couple(self, input_string, input_length):
#     split_string = input_string.split()
#
#     inp_part = split_string[:-1]
#     learn_word = split_string[-1]
#
#     inp_len = len(inp_part)
#
#     if inp_len > input_length:
#         return ' '.join(inp_part[-input_length:]), learn_word
#
#     for i in range(self.max_input_emp_length - inp_len):
#         if str.isdigit(split_string[0]):
#             inp_part.append(0)
#         else:
#             inp_part.append('<unk>')
#     return ' '.join(inp_part), learn_word
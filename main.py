import datetime
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim

from core.music_nn import MusicNN
from data_utils.dataset import MusicStreamingDataset
from gradio_ui import get_gradio_ui
from learning import learn_model
from inference import use_model

# Исправление ошибки в Gradle ConnectionResetError: [WinError 10054] Удаленный
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# CONFIGS
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS_COUNT = 5
BUFFER_SIZE = 1000
PRINT_COEF = 1000

paths = {
    "collab": "/content/data",
    "collab_output": "/content/project/models",
    "kaggle_dataset": "/kaggle/input/music-dataset",
    "kaggle_input_models": "/kaggle/input/models",
    "kaggle_output_models": "/kaggle/working",
    "local": "./data",
    "local_models": "./models"
}

data_path = paths.get("kaggle_dataset")
model_input_path = paths.get("kaggle_input_models")
model_output_path = paths.get("kaggle_output_models")

NEED_TO_LEARN = True
LOAD_LEARNED_MODEL = False
SAVED_MODEL_PATH = f"{model_input_path}/279558_music_model_0.pth"

SOUND_FONT_PATH = "./data/soundfonts/FluidR3_GM.sf2"

# LOGIC START

USE_MODEL = None
USE_DATASET = None

def main():
    global USE_MODEL, USE_DATASET

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Работа будет идти на {device}")
    if device == "cpu":
        torch.set_num_threads(8)

    dataset = MusicStreamingDataset(
        f"{data_path}/parsed_midi.jsonl",
        f"{data_path}/words_alphabet.jsonl",
        f"{data_path}/midi_alphabet.jsonl",
        buffer_size=BUFFER_SIZE
    )
    print("Датасет инициализирован!")

    music_model = MusicNN(dataset.get_words_alphabet_len(), dataset.get_midi_alphabet_len(), 129)
    print("Модель инициализирована!")

    optimizer = optim.Adam(music_model.parameters(), lr=LEARNING_RATE)

    if LOAD_LEARNED_MODEL:
        checkpoint = torch.load(SAVED_MODEL_PATH, weights_only=False, map_location=torch.device(device))
        music_model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Принудительно переносим состояние оптимизатора на GPU
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE

    if NEED_TO_LEARN:
        loss_function = nn.CrossEntropyLoss()
        print("Оптимизаторы инициализированы!")

        start = datetime.datetime.now()
        print("ОБУЧЕНИЕ НАЧАЛОСЬ!")
        learn_model(
            music_model,
            dataset,
            loss_function,
            optimizer,
            EPOCHS_COUNT,
            BATCH_SIZE,
            PRINT_COEF,
            model_output_path,
            random.randint(111111, 999999)
        )
        finish = datetime.datetime.now()
        print('Обучение завершено!\nВремя работы: ' + str(finish - start))
    else:
        USE_MODEL = music_model
        USE_DATASET = dataset

        gradio = get_gradio_ui(gradio_use)
        gradio.launch(share=False)

def gradio_use(prompt: str, temperature: float, top_k: int, duration: float, output_count: int):
    print("Генераци через Gradio")
    return use_model(USE_MODEL, USE_DATASET, prompt, temperature, top_k, duration, output_count, SOUND_FONT_PATH)

if __name__ == "__main__":
    print("\nWORKER INITIALIZED")
    main()


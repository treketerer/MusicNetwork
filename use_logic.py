import torch

from core.music_nn import MusicNN
from data_utils.dataset import MusicStreamingDataset

from miditok import REMI, TokenizerConfig
from midi2audio import FluidSynth
from pedalboard import Pedalboard, Reverb
import soundfile as sf

import uuid
import os
import subprocess

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
tokenizer = REMI(config)
print("Токенайзер инициализирован!")

# Использование
def use_model(model: MusicNN, dataset: MusicStreamingDataset, prompt: str, temperature: float, top_k: int, duration: float, output_count: int, sound_font: str):
    model.to(DEVICE)
    model.eval()

    print(f"Новый запрос:\nprompt: {prompt}\ntemperature: {temperature}\ntop_k: {top_k}\nduration: {duration}\noutput_count: {output_count}")

    if not prompt: return None
    words_idx = dataset.words_to_idx(prompt.lower())
    input_tensor = torch.tensor([words_idx], dtype=torch.long).to(DEVICE)

    outputs_tokens = []
    h, c = None, None

    current_token = torch.tensor([[1]], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        for _ in range(output_count):
            logits, h, c = model(input_tensor, current_token, h, c)

            next_token_logits = logits[0, -1, :] / temperature
            next_token_logits[110:130] /= (2-duration)

            next_token_logits = torch.nan_to_num(next_token_logits, nan=0.0, posinf=10.0, neginf=-10.0)
            # next_token_logits = torch.clamp(next_token_logits, -20.0, 20.0)  # жёсткие границы

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
        file_name = str(uuid.uuid4()).replace('-', '')[:15]
        midi_path = f"./output/midi/{file_name}.mid"
        mp3_path = f"./output/mp3/{file_name}.mp3"

        generated_sequence.dump_midi(midi_path) # создание midi
        save_midi_to_mp3(midi_path, mp3_path, sound_font) # сохранение mp3

        print(f"Файл сохранен:\nmidi - {midi_path}\nmp3 - {mp3_path}")

        # Накладывание эффектов
        audio, sr = sf.read(mp3_path)
        reverb = Reverb(room_size=0.05)
        board = Pedalboard([reverb])
        effected = board(audio, sr)
        sf.write(mp3_path, effected, sr)

        print("Эффекты наложены!")

        words_idx = ["<unk>" if x == 0 else x for x in words_idx]
        return_text = f"{prompt}\n{words_idx}"
        return return_text, midi_path, mp3_path
    except Exception as e:
        print(f"Ошибка сохранения MIDI: {e}")
        return None

def save_midi_to_mp3(midi_path: str, mp3_path: str, sound_font: str) -> str | None:
    try:
        sf2 = os.path.abspath(sound_font).replace(os.sep, '/')
        midi = os.path.abspath(midi_path).replace(os.sep, '/')
        mp3 = os.path.abspath(mp3_path).replace(os.sep, '/')

        if not os.path.isfile(sf2) or not os.path.isfile(midi):
            print(f"Файлы не переданы при преобразовании в mp3")
            return None

        cmd = [
            "fluidsynth",
            "-ni", "-F", mp3,
            "-r", "44100", sf2, midi
        ]

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=60
        )
        return mp3

    except Exception as e:
        print("Ошибка:", str(e))
        import traceback
        traceback.print_exc()
        return None

# top_p = 2.0
#
# next_token_logits = logits[0, -1, :].clone() / temperature
# sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
# cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
# sorted_indices_to_remove = cumulative_probs > top_p
# sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
# sorted_indices_to_remove[..., 0] = 0
#
# # Индексы, которые нужно убрать
# indices_to_remove = sorted_indices[sorted_indices_to_remove]
# next_token_logits[indices_to_remove] = -float('Inf')
#
# probs = torch.softmax(next_token_logits, dim=-1)
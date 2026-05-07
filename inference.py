import torch
from tqdm import tqdm

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
def use_model(model: MusicNN, dataset: MusicStreamingDataset, prompt: str, temperature: float, top_k: int, output_tacts_count: int, sound_font: str, backloop_song_vibe: torch.Tensor = None):
    model.to(DEVICE)
    model.eval()
    model.is_training = False

    print(f"Новый запрос:\nprompt: {prompt}\ntemperature: {temperature}\ntop_k: {top_k}\noutput_count: {output_tacts_count}")

    if not prompt: return None
    words_idx, instruments_idx = dataset.words_to_idx(prompt.lower())

    # words_idx = [1628, 1457, 1369, 1424, 1165, 946, 637, 1345, 1305, 1607]
    # instruments_idx = [128, 36, 88, 27, 29, 30]

    print(words_idx, instruments_idx)
    words_tensor = torch.tensor([words_idx], dtype=torch.long).to(DEVICE)

    if len(instruments_idx) == 0:
        instruments_idx = [0]

    instruments_tensor = torch.tensor([instruments_idx], dtype=torch.long).to(DEVICE)

    tacts = []

    with torch.no_grad():
        backloop_vec = backloop_song_vibe
        cond_h, cond_c = None, None
        for _ in tqdm(range(output_tacts_count)):
            tact_data, backloop_vec, cond_h, cond_c = model(words_tensor, instruments_tensor, backloop_vec=backloop_vec, temperature=temperature, short_notes_coef=1, top_k=top_k, conductor_h=cond_h, conductor_c=cond_c)
            tacts.append(tact_data)

    try:
        united_midi_data = []

        # 1. Посмотрим, какие токены вообще вылетают
        all_generated_tokens = []
        for tact in tacts:
            for inst in tact:
                all_generated_tokens.extend(tact[inst])

        # 2. Собираем MIDI
        for tact in tacts:
            united_midi_data.append(4)  # Bar

            # Для каждого инструмента в такте пишем: Program -> [его токены]
            # Токенайзер сам должен понять Position внутри этих токенов.
            for instrument_id, tokens in tact.items():
                inst_token = 282 + int(instrument_id)

                # Добавляем инструмент
                united_midi_data.append(inst_token)

                # Добавляем токены (фильтруем только системные)
                for t in tokens:
                    if t not in [0, 1, 2, 4]:
                        united_midi_data.append(t)

        print(f"Итоговая длина последовательности для декодера: {len(united_midi_data)}")

        # 3. Декодируем
        generated_sequence = tokenizer.decode(united_midi_data)

        total_notes = sum(len(track.notes) for track in generated_sequence.tracks)
        print(f"Декодер создал MIDI-объект: дорожек={len(generated_sequence.tracks)}, всего нот={total_notes}")

        if total_notes == 0:
            return "ОШИБКА: Модель сгенерировала пустые ноты. Попробуйте поднять Temperature до 1.2", None, None

        # 4. Сохранение
        file_name = str(uuid.uuid4()).replace('-', '')[:15]
        midi_path = f"./output/midi/{file_name}.mid"
        mp3_path = f"./output/mp3/{file_name}.mp3"
        os.makedirs("./output/midi", exist_ok=True)
        os.makedirs("./output/mp3", exist_ok=True)

        generated_sequence.dump_midi(midi_path)

        # Проверяем, создался ли файл
        if os.path.exists(midi_path):
            print(f"MIDI файл создан, размер: {os.path.getsize(midi_path)} байт")

        save_midi_to_mp3(midi_path, mp3_path, sound_font)

        # Проверяем MP3
        if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) < 1000:
            return "ОШИБКА: MP3 не создался или пустой. Проверьте SoundFont и FluidSynth", midi_path, None

        # Эффекты
        audio, sr = sf.read(mp3_path)
        board = Pedalboard([Reverb(room_size=0.03)])
        sf.write(mp3_path, board(audio, sr), sr)

        return f"Успех! Сгенерировано нот: {total_notes}", midi_path, mp3_path

    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return f"Ошибка: {str(e)}", None, None

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
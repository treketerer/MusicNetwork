import torch
from tqdm import tqdm

from core.music_nn import MusicNN
from data_utils.dataset import MusicStreamingDataset
import data_utils

from miditok import REMI, TokenizerConfig
from symusic import Score
from pedalboard import Pedalboard, Reverb
import soundfile as sf

import uuid
import os
import subprocess

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
tokenizer = REMI(config)
print("Токенайзер инициализирован!")

class Inference_Manager:
    def __init__(self, model: MusicNN, dataset: MusicStreamingDataset, max_token_in_tact: int, max_instruments: int, sound_font_path: str):
        self.model = model
        self.dataset = dataset
        self.sound_font_path = sound_font_path

        tokenizer_config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
        self.worker_tokenizer = REMI(tokenizer_config)
        self.midi_parser = data_utils.MidiParser()

        self.max_token_in_tact = max_token_in_tact
        self.max_instruments = max_instruments

    # Подготовка данных к обычной генерации
    def common_generation_prepare(self, prompt: str, temperature: float, top_k: int, output_count: int):
        if not prompt: return None
        words_idx, instruments_idx = self.dataset.words_to_idx(prompt.lower())

        return self.use_model(words_idx, instruments_idx, temperature, top_k, output_count)

    # Подготовка данных к генерации подражания
    def imitation_generation_prepare(self, prompt: str, midi_input: str, temperature: float, top_k: int, output_count: int):
        words_idx, _ = self.dataset.words_to_idx(prompt.lower())

        midi = Score(midi_input)
        tokens = self.worker_tokenizer(midi)

        if not hasattr(tokens, 'ids'):
            return "Ошибка", "Ошибка", "Ошибка"
        ids = tokens.ids

        start_programs_token = 282
        now_instrument_id = -2
        tacts_list, instruments_list = self.midi_parser.parse_tokens_in_ids_arr(ids, start_programs_token, self.worker_tokenizer)

        back_loops_vectors = []

        print(tacts_list[0])
        tact_data = torch.zeros((len(tacts_list), self.max_instruments, self.max_token_in_tact), dtype=torch.long)
        for tact_idx, tact in enumerate(tacts_list):
            for k_idx, key in enumerate(list(tact.keys())):
                if k_idx >= self.max_instruments: break
                notes_list = tact[key]
                actual_notes_to_fill = min(len(notes_list), self.max_token_in_tact)
                for n_key in range(actual_notes_to_fill):
                    tact_data[tact_idx, k_idx, n_key] = tact[key][n_key]

        with torch.no_grad():
            emb = self.model.instruments_lstm.midi_embeddings(tact_data.to(DEVICE)).unsqueeze(0)
            back_loops_vectors = self.model.backloop_encoder(emb)
        back_loops_vectors = back_loops_vectors.squeeze(0)
        mean_song_vecs = torch.mean(back_loops_vectors, dim=0)
        final_vec = torch.mean(torch.stack([mean_song_vecs, back_loops_vectors[-1]]), dim=0)
        final_vec = final_vec.unsqueeze(0).unsqueeze(0)

        return self.use_model(words_idx, instruments_list,
                         temperature, top_k, output_count,
                         backloop_song_vibe=final_vec)

    # Использование
    def use_model(self, words_idx: list, instruments_idx: list, temperature: float, top_k: int, output_tacts_count: int, backloop_song_vibe: torch.Tensor = None):
        self.model.to(DEVICE)
        self.model.eval()
        self.model.is_training = False

        print(f"Новый запрос:\nwords_idx: {words_idx}\ntemperature: {temperature}\ntop_k: {top_k}\noutput_count: {output_tacts_count}")

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
                tact_data, backloop_vec, cond_h, cond_c = self.model(words_tensor, instruments_tensor, backloop_vec=backloop_vec, temperature=temperature, short_notes_coef=1, top_k=top_k, conductor_h=cond_h, conductor_c=cond_c)
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

            self.save_midi_to_mp3(midi_path, mp3_path)

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

    def save_midi_to_mp3(self, midi_path: str, mp3_path: str) -> str | None:
        try:
            sf2 = os.path.abspath(self.sound_font_path).replace(os.sep, '/')
            midi = os.path.abspath(midi_path).replace(os.sep, '/')
            mp3 = os.path.abspath(mp3_path).replace(os.sep, '/')

            if not os.path.isfile(sf2) or not os.path.isfile(midi):
                print(f"Файлы не переданы при преобразовании в mp3")
                return None

            cmd = [
                "fluidsynth",
                "-ni",
                "-g", "1.0",           # Уровень громкости (чтобы не было тихо)
                "-F", mp3,             # Output файл
                "-T", "au",            # Важно для Windows! Формат рендера.
                "-O", "s16",           # Формат сэмплов
                "-r", "44100",
                sf2, midi
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
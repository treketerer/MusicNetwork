import torch
import os

# Путь к чекпоинту
INPUT_MODEL_PATH = "../models/372989_music_model_2_final.pth"
OUTPUT_MODEL_PATH = "../models/MusicNN_v1.9.pth"


def compress_model():
    print(f"Загрузка файла: {INPUT_MODEL_PATH}")
    # Загружаем в оперативную память
    checkpoint = torch.load(INPUT_MODEL_PATH, map_location='cpu')

    # Достаем ТОЛЬКО веса самой модели
    if 'model_state_dict' in checkpoint:
        model_weights = checkpoint['model_state_dict']
    else:
        # На случай, если это уже просто веса
        model_weights = checkpoint

    print("Очистка от оптимизатора Adam и сжатие весов (FP32 -> FP16)...")

    # Сжимаем тензоры для уменьшения размера в 2 раза
    compressed_weights = {}
    for key, value in model_weights.items():
        if value.is_floating_point():
            # Переводим float32 в float16 (идеально для инференса/хранения)
            compressed_weights[key] = value.half()
        else:
            # Оставляем int/long (индексы) как есть
            compressed_weights[key] = value

    # Формируем минималистичный словарь для сохранения
    minimal_checkpoint = {
        'model_state_dict': compressed_weights
    }

    # Сохраняем
    torch.save(minimal_checkpoint, OUTPUT_MODEL_PATH)

    # Сравниваем размеры
    old_size = os.path.getsize(INPUT_MODEL_PATH) / (1024 * 1024)
    new_size = os.path.getsize(OUTPUT_MODEL_PATH) / (1024 * 1024)

    print(f"Готово! Сохранено в: {OUTPUT_MODEL_PATH}")
    print(f"Старый размер: {old_size:.2f} MB")
    print(f"Новый размер:  {new_size:.2f} MB")
    print(f"Файл стал меньше в {old_size / new_size:.1f} раз(а).")


if __name__ == "__main__":
    compress_model()
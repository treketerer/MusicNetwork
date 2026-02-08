import torch
import torch.nn as nn
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def learn_model(model: nn.Module, dataset: torch.utils.data.IterableDataset, loss_function, optimizer, epochs_count: int, batch_size: int, print_coef: int, model_output_path: str, save_model_id: int):
    # Обучение
    model.train()
    model.to(DEVICE)
    print("ОБУЧЕНИЕ НАЧАЛОСЬ!")

    epoch = 0
    current_loss = None

    try:
        for epoch in range(epochs_count):
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=2,  # <--- Задействуем 8 ядер для чтения и парсинга
                persistent_workers=True,  # <--- Не убивать воркеры после эпохи (важно для Windows!)
                prefetch_factor=10,  # <--- Каждый воркер заранее готовит по 2 батча
                collate_fn=dataset.collate_fn,
                pin_memory=True  # <--- Ускоряет передачу данных (особенно если есть GPU)
            )

            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()

                prompts = batch.get('idx_prompts').to(DEVICE, non_blocking=True)
                tokens = batch.get('tokens').to(DEVICE, non_blocking=True)

                inp = tokens[:, :-1].to(DEVICE)
                target = tokens[:, 1:].to(DEVICE)

                logits, _, _ = model(prompts, inp, None, None)

                current_loss = loss_function(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                current_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if (i + 1) % print_coef == 0 or i == 0:
                    print(f'Epoch {epoch} Iteration {i + 1}, Loss: {current_loss.item():.4f}')

            print(f"Эпоха {epoch} завершена!")
            save_model(model_output_path, save_model_id, epoch, model, optimizer, current_loss)
    except Exception as e:
        print(f"Ошибка во время обучения: {e}")
        if current_loss is not None:
            save_model(model_output_path, save_model_id, epoch, model, optimizer, current_loss)

def save_model(model_output_path, save_model_id, epoch, model, optimizer, current_loss):
    path = f"{model_output_path}/{save_model_id}_music_model_{epoch}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': current_loss.item(),
    }, path)
    print(f"Модель сохранена в {path}")
import torch
import torch.nn as nn

from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from core import MusicNN
from data_utils import MusicStreamingDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def learn_model(model: MusicNN, dataset: MusicStreamingDataset, optimizer, scheduler,
                epochs_count: int, batch_size: int, accumulation_steps: int,
                model_output_path: str, save_model_id: int):
    optimizer.zero_grad()
    # Обучение

    if DEVICE == "cuda":
        torch.cuda.empty_cache()  # Очистить неиспользуемую память
        import gc
        gc.collect()

    model.train()
    model.to(DEVICE)
    print("ОБУЧЕНИЕ НАЧАЛОСЬ!")

    epoch = 0
    current_loss = None

    criterion_notes = nn.CrossEntropyLoss(ignore_index=0)
    criterion_insts = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0).to(DEVICE))

    try:
        global_loss_history = []
        for epoch in range(epochs_count):
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=2,  # <--- Задействуем 2 ядра для чтения и парсинга
                persistent_workers=True,  # <--- Не убивать воркеры после эпохи
                prefetch_factor=3,  # <--- Каждый воркер готовит 10 батчей
                collate_fn=dataset.collate_fn,
                pin_memory=True  # <--- Ускоряет передачу данных
            )

            loop = tqdm(train_loader, leave=False, mininterval=10.0)
            loop.set_description(f"Epoch {epoch}")

            local_loss_history = []

            save_coef = int(len(loop) / 2.7)
            print(save_coef)
            for i, batch in enumerate(loop):
                prompts = batch.get('idx_prompts').to(DEVICE, non_blocking=True)
                full_instruments = batch.get('instruments').to(DEVICE, non_blocking=True)
                tacts_data = batch.get('tacts_data').to(DEVICE, non_blocking=True)
                tacts_instruments = batch.get('tacts_instruments').to(DEVICE, non_blocking=True)

                inp_tdata = tacts_data.to(DEVICE)
                target_tdata = tacts_data.to(DEVICE)
                inp_inst = tacts_instruments.to(DEVICE)
                target_inst = tacts_instruments.to(DEVICE)

                tact_logits, instruments_logits = model(prompts, full_instruments, tacts_instr=inp_inst, tacts_data=inp_tdata)

                """
                tact_logits - (batch, songs, tacts, notes, note_emb)
                instruments_logits - (batch, songs, tacts, instruments_probability)
                """

                loss_notes = criterion_notes(
                    tact_logits[:, :, :, :-1, :].reshape(-1, dataset.midi_alphabet_len),
                    target_tdata[:, :, :, 1:].reshape(-1)
                )

                target_inst_multihot = torch.zeros(target_inst.shape[0], target_inst.shape[1], 130, device=DEVICE)
                target_inst_multihot.scatter_(2, target_inst, 1)

                target_real = target_inst_multihot[:, :, :129]

                loss_inst = criterion_insts(
                    instruments_logits.reshape(-1, 129),
                    target_real.reshape(-1, 129).float()
                )

                current_loss = loss_notes * 1.0 + loss_inst * 1.4
                loss_normalized = current_loss / accumulation_steps
                loss_normalized.backward()

                if (i + 1) % accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    current_lr = optimizer.param_groups[0]['lr']

                    loop.set_postfix(loss=current_loss.item(), loss_inst=loss_inst.item(), loss_notes=loss_notes.item(), lr=current_lr)
                    local_loss_history.append(current_loss.item())

                if i % save_coef == 0 and i > 0:
                    save_model(model_output_path, save_model_id, f"{epoch}_{i}", model, optimizer, current_loss)

                    avg_epoch_loss = sum(local_loss_history) / len(local_loss_history)
                    scheduler.step(avg_epoch_loss)
                    global_loss_history += [*local_loss_history]
                    local_loss_history = []
                    print(f"Средний лосс за срез: {avg_epoch_loss:.4f}")

            print(f"Эпоха {epoch} завершена!")
            try:
                global_loss_history += local_loss_history

                avg_epoch_loss = sum(local_loss_history) / len(local_loss_history)
                scheduler.step(avg_epoch_loss)
                print(f"Средний лосс за срез: {avg_epoch_loss:.4f}")

                save_model(model_output_path, save_model_id, f"{epoch}_final", model, optimizer, current_loss)
                save_loss_image(local_loss_history, epoch, model_output_path, save_model_id)
            except Exception as ex:
                save_model(model_output_path, save_model_id, f"{epoch}", model, optimizer, current_loss)

        save_loss_image(global_loss_history, epoch, model_output_path, save_model_id)
        # save_model(model_output_path, save_model_id, f"{epoch}", model, optimizer, current_loss)

    except Exception as e:
        print(f"\nОшибка во время обучения: {e}")
        if current_loss is not None:
            save_model(model_output_path, save_model_id, epoch, model, optimizer, current_loss)

def save_loss_image(loss_history: list, epoch_description: str | int, model_output_path: str, save_model_id: int):
    plt.clf()
    plt.plot(loss_history)
    plt.title(f"Loss Epoch {epoch_description} {save_model_id}")
    plt.savefig(f"{model_output_path}/{save_model_id}_loss_epoch_{epoch_description}.png")  # Сохраняем картинку
    plt.close()

def save_model(model_output_path, save_model_id, epoch, model, optimizer, current_loss):
    path = f"{model_output_path}/{save_model_id}_music_model_{epoch}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': current_loss.item(),
    }, path)
    print(f"\nМодель сохранена в {path}")
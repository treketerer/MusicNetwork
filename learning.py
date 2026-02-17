import torch
import torch.nn as nn

from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from core import MusicNN
from data_utils import MusicStreamingDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def learn_model(model: MusicNN, dataset: MusicStreamingDataset, loss_function, optimizer, epochs_count: int, batch_size: int, print_coef: int, model_output_path: str, save_model_id: int):
    # Обучение
    model.train()
    model.to(DEVICE)
    print("ОБУЧЕНИЕ НАЧАЛОСЬ!")

    epoch = 0
    current_loss = None

    criterion_notes = nn.CrossEntropyLoss(ignore_index=0)
    criterion_insts = nn.BCEWithLogitsLoss()

    try:
        for epoch in range(epochs_count):
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=1,  # <--- Задействуем 2 ядра для чтения и парсинга
                persistent_workers=True,  # <--- Не убивать воркеры после эпохи
                prefetch_factor=1,  # <--- Каждый воркер готовит 10 батчей
                collate_fn=dataset.collate_fn,
                pin_memory=False  # <--- Ускоряет передачу данных
            )

            loop = tqdm(train_loader, leave=False)
            loss_history = []

            for i, batch in enumerate(loop):
                optimizer.zero_grad()

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

                target_inst_multihot = torch.zeros(target_inst.shape[0], target_inst.shape[1], 129, device=DEVICE)
                target_inst_multihot.scatter_(2, target_inst, 1)

                loss_inst = criterion_insts(
                    instruments_logits.reshape(-1, 129),
                    target_inst_multihot.reshape(-1, 129).float()
                )

                instrum_loss_coef = 0.5
                current_loss = loss_notes + (instrum_loss_coef * loss_inst)

                current_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if (i + 1) % print_coef == 0:
                    loop.set_description(f"Epoch {epoch}")
                    loop.set_postfix(loss=current_loss.item())
                    loss_history.append(current_loss.item())
                if (i+1) % 5 == 0:
                    save_model(model_output_path, save_model_id, epoch, model, optimizer, current_loss)
                    plt.plot(loss_history)
                    plt.title(f"Loss Epoch {epoch} {save_model_id}")
                    plt.savefig(f"{model_output_path}/{save_model_id}_loss_epoch_{epoch}.png")  # Сохраняем картинку
                    plt.close()

            print(f"Эпоха {epoch} завершена!")
            save_model(model_output_path, save_model_id, epoch, model, optimizer, current_loss)

    except Exception as e:
        print(f"\nОшибка во время обучения: {e}")
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
    print(f"\nМодель сохранена в {path}")
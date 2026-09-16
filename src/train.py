"""Training loop with warmup + cosine schedule (per-epoch) and best-val checkpoint."""

import copy
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from tqdm.auto import tqdm


def _build_scheduler(optimizer, epochs: int, warmup_epochs: int):
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, eta_min=0.0, T_max=epochs)
    warmup = LinearLR(optimizer, start_factor=1.0 / warmup_epochs,
                      end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, eta_min=0.0,
                               T_max=max(epochs - warmup_epochs, 1))
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs])


def _is_ivon(optimizer) -> bool:
    return type(optimizer).__name__ == "IVON"


def train_model(
    model: nn.Module,
    optimizer,
    train_loader,
    val_loader,
    epochs: int = 100,
    warmup_epochs: int = 5,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
    patience: int = 9999) -> tuple:

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    scheduler = _build_scheduler(optimizer, epochs, warmup_epochs)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    ivon = _is_ivon(optimizer)
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            if ivon:
                with optimizer.sampled_params(train=True):
                    optimizer.zero_grad()
                    loss = criterion(model(x), y)
                    loss.backward()
            else:
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()

            optimizer.step()
            train_loss += loss.item() * len(y)
            train_n += len(y)
            pbar.set_postfix(loss=f"{train_loss/train_n:.4f}")

        scheduler.step()

        val_loss = _eval_loss(model, val_loader, criterion, device, ivon)
        history.append({"epoch": epoch, "train_loss": train_loss / train_n, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"early stop epoch {epoch}, best={best_val_loss:.4f}")
                break

        if verbose:
            print(f"epoch {epoch}/{epochs} val={val_loss:.4f} best={best_val_loss:.4f}")

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def _eval_loss(model, loader, criterion, device, ivon: bool) -> float:
    model.eval()
    total_loss, total_n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        total_n += len(y)
    return total_loss / total_n

"""In-domain (Acc, NLL, ECE) and OOD (FPR@95, AUROC) metrics."""

import contextlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve


@contextlib.contextmanager
def _posterior_sample(model, optimizer):
    """Perturb model weights by one posterior sample (IVON only)."""
    with optimizer.sampled_params(train=False):
        yield


def _uses_posterior(optimizer) -> bool:
    return optimizer is not None and type(optimizer).__name__ == "IVON"


@torch.no_grad()
def collect_predictions(model, loader, device=torch.device("cpu"),
                        optimizer=None, n_samples: int = 64):
    model.eval()
    mc = _uses_posterior(optimizer)
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        if mc:
            batch_probs = []
            for _ in range(n_samples):
                with _posterior_sample(model, optimizer):
                    p = F.softmax(model(x), dim=-1).cpu().numpy()
                batch_probs.append(p)
            probs = np.mean(batch_probs, axis=0)
        else:
            probs = F.softmax(model(x), dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == labels).mean())


def nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    p_correct = probs[np.arange(len(labels)), labels].clip(eps, 1.0)
    return float(-np.log(p_correct).mean())


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE with equal-width bins."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece_val += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece_val)


def indomain_metrics(model, loader, device=torch.device("cpu"), n_bins: int = 15,
                     optimizer=None, n_samples: int = 64):
    probs, labels = collect_predictions(model, loader, device, optimizer, n_samples)
    return {
        "Acc": accuracy(probs, labels),
        "NLL": nll(probs, labels),
        "ECE": ece(probs, labels, n_bins)}


@torch.no_grad()
def collect_confidence(model, loader, device=torch.device("cpu"),
                       optimizer=None, n_samples: int = 64):
    """Max-softmax confidence per sample.

    - IVON: averaged over n_samples posterior draws.
    - All others (AdamW, uCBOpt): single forward pass max-softmax.
    """
    model.eval()
    mc = _uses_posterior(optimizer)
    scores = []

    for x, *_ in loader:
        x = x.to(device)
        if mc:
            batch_probs = []
            for _ in range(n_samples):
                with _posterior_sample(model, optimizer):
                    p = F.softmax(model(x), dim=-1).cpu().numpy()
                batch_probs.append(p)
            avg_probs = np.mean(batch_probs, axis=0)
            scores.append(avg_probs.max(axis=1))
        else:
            probs = F.softmax(model(x), dim=-1)
            scores.append(probs.max(dim=1).values.cpu().numpy())

    return np.concatenate(scores)


def fpr_at_tpr(in_scores, ood_scores, tpr_target: float = 0.95, eps: float = 0.0005) -> float:
    """FPR@TPR95 matching the paper's roc_curve-based implementation."""
    labels = np.concatenate([np.ones(len(in_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([in_scores, ood_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = (tpr <= (tpr_target + eps)) >= (tpr_target - eps)
    return float(fpr[idx].mean())


def auroc(in_scores, ood_scores) -> float:
    labels = np.concatenate([np.ones(len(in_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([in_scores, ood_scores])
    return float(roc_auc_score(labels, scores))


def ood_metrics(model, id_loader, ood_loader, device=torch.device("cpu"),
                optimizer=None, n_samples: int = 64):
    in_scores = collect_confidence(model, id_loader, device, optimizer, n_samples)
    ood_scores = collect_confidence(model, ood_loader, device, optimizer, n_samples)
    return {
        "FPR@95": fpr_at_tpr(in_scores, ood_scores),
        "AUROC": auroc(in_scores, ood_scores)}

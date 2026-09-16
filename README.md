# Reproducing Singh et al. (2026)

Light variant reproduction of the reference results from
*Maxitive Donsker–Varadhan Formulation for Possibilistic Variational Inference* (Singh et al., 2026). Compares AdamW, IVON, and uCBOpt (3 seeds each: 0, 1, 2) on a LeNet-5 variant (44,426 parameters) trained on Fashion-MNIST (100 epochs, linear warmup over 5 epochs + cosine decay), evaluated in-domain (Acc, NLL, ECE) and on EMNIST-Letters as OOD (FPR@95, AUROC).


## Structure

```
src/
  data.py        -- Fashion-MNIST (54k train / 6k val / 10k test) and EMNIST-Letters (14.8k test) loaders
  model.py       -- LeNet-5 variant architecture (44,426 parameters, dropout disabled)
  optimizers.py  -- AdamW, IVON, and uCBOpt implementations
  train.py       -- training loop (batch size 128, linear warmup + cosine decay, no early stopping)
  evaluate.py    -- Acc, NLL, ECE (15 bins), FPR@95, AUROC metrics (lowest val-loss checkpoint)
checkpoints/     -- saved model weights and loss histories (3 seeds x 3 optimizers)
results/         -- CSV metric tables
xp_reprod.ipynb  -- full experiment (training + evaluation)
requirements.txt
```


## Setup

The experiment notebook is configured for Google Colab (T4 GPU runtime).
To run on a local machine with a CUDA-enabled GPU instead:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Usage

Open `xp_reprod.ipynb` and execute cells in order:

- Google Colab: Open the notebook directly on Colab and run all cells sequentially.
- Local environment: Activate the virtual environment, then skip Cell 1 (Google Drive mount) before executing the rest of the notebook.

The notebook is split into two sections: training (cells 1-7) and evaluation (cells 8-13).
Checkpoints are saved to `checkpoints/` after training (saving the checkpoint with the lowest validation loss). Re-running the evaluation section loads them without retraining.
Results are written to `results/`.


## Hyperparameters

All optimizers use momentum beta_1 = 0.9:
- AdamW: rho = 1e-3, delta = 1e-2
- IVON: rho = 0.2, delta = 2e-3, beta_2 = 0.99999, h_0 = 0.5, effective sample size 5e4, 1 MC sample/step
- uCBOpt: rho = 1e-2, delta = 2e-3, beta_2 = 0.99999, h_0 = 0.05, fixed vartheta_0 = 8e-6


## Results (mean +/- std, 3 seeds)

Evaluated on the checkpoint with the lowest validation loss.
Single deterministic forward pass for AdamW and uCBOpt at inference time, 64-sample Monte-Carlo ensemble for IVON.

| Optimizer | Acc (%) ^ | NLL v | ECE v | FPR@95 v | AUROC ^ |
|---|---|---|---|---|---|
| uCBOpt | 90.9 +/- 0.2 | 0.272 +/- 0.004 | 0.025 +/- 0.001 | 0.263 +/- 0.005 | 0.795 +/- 0.005 |
| IVON | 91.0 +/- 0.2 | 0.258 +/- 0.003 | 0.019 +/- 0.001 | 0.241 +/- 0.009 | 0.812 +/- 0.012 |
| AdamW | 89.6 +/- 0.3 | 0.296 +/- 0.005 | 0.022 +/- 0.006 | 0.335 +/- 0.005 | 0.722 +/- 0.016 |
| AdamW (paper) | 90.0 +/- 0.5 | 0.294 +/- 0.007 | 0.026 +/- 0.004 | 0.321 +/- 0.008 | 0.745 +/- 0.009 |
| IVON (paper) | 91.1 +/- 0.3 | 0.258 +/- 0.001 | 0.013 +/- 0.004 | 0.249 +/- 0.024 | 0.804 +/- 0.030 |
| uCBOpt (paper) | 90.7 +/- 0.2 | 0.270 +/- 0.002 | 0.023 +/- 0.000 | 0.245 +/- 0.010 | 0.810 +/- 0.012 |
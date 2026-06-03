"""Étape 5 — entraîne le classifieur sur features positives vs négatives.

Boucle d'entraînement légère (torch pur). Les négatifs viennent des features
pré-calculées d'openWakeWord (~2000 h de parole/bruit/musique), ce qui donne
au modèle un faible taux de fausses activations sans qu'on ait à télécharger
des téraoctets de données.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import WakeWordModel, export_onnx, N_FRAMES, EMB_DIM


def _load_negative_features(path: str | Path, max_windows: int) -> np.ndarray:
    """Charge les features négatives pré-calculées (.npy d'openWakeWord).

    Gère plusieurs formats possibles :
      - (n, 16, 96)        → fenêtres prêtes, utilisées directement
      - (n, 1536)          → aplati → reshape en (n, 16, 96)
      - (total_frames, 96) → flux de frames → découpé en fenêtres de 16
    On échantillonne `max_windows` fenêtres au hasard.
    """
    neg = np.load(path, mmap_mode="r")

    if neg.ndim == 3 and neg.shape[1:] == (N_FRAMES, EMB_DIM):
        windows = neg
    elif neg.ndim == 2 and neg.shape[1] == N_FRAMES * EMB_DIM:
        windows = neg.reshape(-1, N_FRAMES, EMB_DIM)
    elif neg.ndim == 2 and neg.shape[1] == EMB_DIM:
        # flux (total_frames, 96) → fenêtres non chevauchantes de 16 frames
        n_full = neg.shape[0] // N_FRAMES
        windows = np.asarray(neg[:n_full * N_FRAMES]).reshape(-1, N_FRAMES, EMB_DIM)
    else:
        raise ValueError(
            f"Format de features négatives inattendu : {neg.shape}. "
            f"Attendu (n,16,96), (n,1536) ou (frames,96)."
        )

    n = min(max_windows, len(windows))
    idx = np.random.choice(len(windows), n, replace=False)
    return np.asarray(windows[idx], dtype=np.float32)


def train(
    positive_features: np.ndarray,      # (n_pos, 16, 96)
    negative_features_path: str | Path,
    out_onnx: str | Path,
    neg_ratio: int = 10,                # nb de négatifs par positif
    layer_dim: int = 128,
    n_blocks: int = 1,
    epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 1e-3,
    false_activation_penalty: float = 1.0,  # poids des négatifs dans la loss
    device: str | None = None,
) -> dict:
    """Entraîne et exporte le modèle ONNX. Retourne des métriques simples."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")

    n_pos = len(positive_features)
    n_neg = min(n_pos * neg_ratio, 500_000)
    neg = _load_negative_features(negative_features_path, n_neg)
    print(f"[train] positifs={n_pos}  négatifs={len(neg)}")

    X = np.concatenate([positive_features, neg]).astype(np.float32)
    y = np.concatenate([np.ones(n_pos), np.zeros(len(neg))]).astype(np.float32)

    # split train / val
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    n_val = max(1, int(0.1 * len(X)))
    Xtr, ytr = X[n_val:], y[n_val:]
    Xval, yval = X[:n_val], y[:n_val]

    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = WakeWordModel((N_FRAMES, EMB_DIM), layer_dim, n_blocks).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # pos_weight < 1 pénalise davantage les faux positifs (négatifs)
    pos_weight = torch.tensor([1.0 / false_activation_penalty], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        # validation
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(Xval).to(device)
            scores = torch.sigmoid(model(xb)).cpu().numpy().ravel()
        pred = (scores > 0.5).astype(np.float32)
        acc = float((pred == yval).mean())
        # recall sur les positifs, FP rate sur les négatifs
        pos_mask = yval == 1
        recall = float((pred[pos_mask] == 1).mean()) if pos_mask.any() else 0.0
        fp = float((pred[~pos_mask] == 1).mean()) if (~pos_mask).any() else 0.0
        print(f"  epoch {ep+1:2d}/{epochs}  loss={tot/len(Xtr):.4f}  "
              f"val_acc={acc:.3f}  recall={recall:.3f}  fp={fp:.3f}")

    Path(out_onnx).parent.mkdir(parents=True, exist_ok=True)
    # Filet : sauve les poids AVANT l'export ONNX. Si l'export échoue, on peut
    # ré-exporter sans relancer génération + entraînement.
    ckpt = Path(out_onnx).with_suffix(".pt")
    torch.save(model.state_dict(), ckpt)
    print(f"[train] poids sauvés → {ckpt}")

    export_onnx(model, str(out_onnx))
    return {"val_acc": acc, "recall": recall, "fp_rate": fp}

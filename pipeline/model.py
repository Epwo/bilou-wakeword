"""Étape 4 — le classifieur openWakeWord.

Petit réseau entièrement connecté qui prend un embedding (16, 96) aplati et
sort un score [0, 1]. Architecture reprise du train.py officiel openWakeWord
(blocs Linear + LayerNorm + ReLU). N'utilise QUE torch.nn — aucune dépendance
fragile.
"""

from __future__ import annotations

import torch
import torch.nn as nn


N_FRAMES = 16
EMB_DIM = 96


class FCNBlock(nn.Module):
    def __init__(self, layer_dim: int):
        super().__init__()
        self.fcn_layer = nn.Linear(layer_dim, layer_dim)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(layer_dim)

    def forward(self, x):
        return self.relu(self.layer_norm(self.fcn_layer(x)))


class WakeWordModel(nn.Module):
    """Classifieur binaire sur embeddings (16, 96).

    Sortie : logit (pas de sigmoid interne — on utilise BCEWithLogitsLoss à
    l'entraînement, et on applique sigmoid à l'export ONNX pour matcher le
    format attendu par openWakeWord à l'inférence)."""

    def __init__(self, input_shape=(N_FRAMES, EMB_DIM), layer_dim: int = 128,
                 n_blocks: int = 1):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(input_shape[0] * input_shape[1], layer_dim)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.LayerNorm(layer_dim)
        self.blocks = nn.ModuleList([FCNBlock(layer_dim) for _ in range(n_blocks)])
        self.head = nn.Linear(layer_dim, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.norm1(self.relu1(self.layer1(x)))
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)  # logit


class _ExportWrapper(nn.Module):
    """Ajoute la sigmoid pour l'export ONNX (openWakeWord attend un score
    [0,1] en sortie du modèle à l'inférence)."""

    def __init__(self, model: WakeWordModel):
        super().__init__()
        self.model = model
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))


def export_onnx(model: WakeWordModel, path: str,
                input_shape=(N_FRAMES, EMB_DIM)) -> None:
    """Exporte le modèle (avec sigmoid) en ONNX, format openWakeWord.

    On force l'ANCIEN exporter TorchScript (`dynamo=False`) : il est stable
    pour ce petit réseau FC, et n'a pas besoin d'onnxscript. Le nouveau
    backend dynamo (défaut des torch récents) plante sur ce modèle sur Colab.
    """
    # Exporter sur CPU : le modèle peut être sur GPU (entraîné sur cuda),
    # mais le dummy input est sur CPU → on aligne tout sur CPU.
    model = model.eval().cpu()
    wrapper = _ExportWrapper(model).eval().cpu()
    dummy = torch.randn(1, *input_shape)   # CPU
    kwargs = dict(
        input_names=["onnx____Flatten_0"],      # nom attendu par openWakeWord
        output_names=["output"],
        dynamic_axes={"onnx____Flatten_0": {0: "batch"}, "output": {0: "batch"}},
        opset_version=14,
    )
    try:
        torch.onnx.export(wrapper, dummy, path, dynamo=False, **kwargs)
    except TypeError:
        # Version de torch sans le paramètre `dynamo` → l'ancien exporter est
        # déjà le défaut.
        torch.onnx.export(wrapper, dummy, path, **kwargs)
    print(f"[export] modèle ONNX écrit → {path}")

# bilou-wakeword

Pipeline d'entraînement de mot de réveil ("wake word") pour
[openWakeWord](https://github.com/dscripka/openWakeWord), reconstruite from
scratch pour être **robuste** là où le notebook officiel casse.

Produit un fichier `bilou.onnx` utilisable directement par openWakeWord
(et donc par le robot Reachy).

## Pourquoi ce repo

Le notebook officiel openWakeWord épingle des versions de ~2023 qui sont
incompatibles avec le Colab actuel (Python 3.12, numpy 2, torch récent). Il
casse en cascade : `generate_samples` supprimé, conflit ABI torch/torchaudio,
conflit numpy 1.x/2.x, tensorflow obsolète…

Cette pipeline **évite tout ça** en ne gardant que les dépendances saines :

| Étape | Outil | Pourquoi c'est robuste |
|---|---|---|
| Générer les samples du mot | `piper-sample-generator` (API CLI moderne) | plus l'ancien `import generate_samples` supprimé |
| Audio → embeddings | modèles **ONNX** d'openWakeWord | pas de torchaudio |
| Augmentation bruit/réverb | **numpy** maison | pas de torch-audiomentations |
| Classifieur | **torch.nn** pur | pas de torchmetrics / speechbrain |
| Export | `torch.onnx` | pas de tensorflow / onnx_tf |

## Principe (comment ça marche)

openWakeWord = un petit classifieur sur des **embeddings audio**. On n'entraîne
pas un gros modèle :

1. Un TTS (piper, voix française) **fabrique** des milliers d'exemples du mot
   "bilou" (le mot n'existe dans aucun dataset — on le synthétise).
2. Deux modèles ONNX figés (melspectrogram + embedding) transforment chaque
   clip en vecteurs `(16, 96)`.
3. Les **négatifs** (≈2000 h de parole/bruit/musique) sont des features
   pré-calculées fournies par openWakeWord — pas besoin de télécharger des To.
4. On entraîne un petit réseau FC (quelques couches) à séparer positifs /
   négatifs. ~10 min.
5. Export `bilou.onnx`.

## Utilisation — Colab (recommandé, GPU gratuit)

Ouvre `train_colab.ipynb` dans Google Colab, mets le runtime sur GPU, et
exécute les cellules. À la fin, `bilou.onnx` se télécharge.

## Utilisation — local (si tu as un GPU, sinon lent)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_data.py                       # voix FR + features négatives
python run_all.py --word bilou --n-samples 2000 --epochs 20
# → models/bilou.onnx
```

Options utiles :

```bash
# Plus de samples = meilleur modèle (mais plus lent)
python run_all.py --word bilou --n-samples 20000

# Augmentation avec tes propres clips de bruit ambiant
python run_all.py --word bilou --noise-dir path/to/noise_wavs/

# Un autre mot
python run_all.py --word reachy
```

## Structure

```
bilou-wakeword/
├── README.md
├── requirements.txt
├── download_data.py        télécharge voix FR + features négatives
├── run_all.py              orchestrateur (génère → features → entraîne → export)
├── train_colab.ipynb       notebook Colab clé-en-main
└── pipeline/
    ├── generate.py         1. samples du mot via piper (nouvelle API)
    ├── features.py         2. audio → embeddings (ONNX, pas torchaudio)
    ├── augment.py          2b. augmentation bruit/réverb (numpy)
    ├── model.py            4. classifieur FC + export ONNX
    └── train.py            5. boucle d'entraînement
```

## Brancher le modèle sur Reachy

Une fois `bilou.onnx` obtenu :

```bash
cp bilou.onnx ~/Documents/code/reachy/voice_agent/wake/models/bilou.onnx
cd ~/Documents/code/reachy/voice_agent
source .venv_wake/bin/activate
python wake/wake_word.py --model wake/models/bilou.onnx --meter
```

## Statut / avertissement

Cette pipeline a été conçue pour être robuste, mais **n'a pas encore été
testée de bout en bout** (l'entraînement nécessite Colab/GPU). Les points
les plus susceptibles de demander un ajustement :

- L'API exacte de `piper-sample-generator` (flag `--length-scale` au pluriel
  ou singulier selon la version) — voir `pipeline/generate.py`.
- L'API `AudioFeatures.embed_clips` d'openWakeWord (nom de méthode / forme de
  sortie) — voir `pipeline/features.py`.

Si une étape casse, l'erreur sera localisée (un seul module), pas une cascade
de conflits de dépendances comme dans le notebook original.

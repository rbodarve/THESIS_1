<div align="center">

# 🛡️ THESIS_1 — Bilingual NSFW Content Detection

**Two lightweight, edge-deployable models for moderating unsafe on-screen content**
— images *and* Tagalog/Taglish text.

<!-- Badges -->
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![YOLOv10](https://img.shields.io/badge/Detector-YOLOv10n-00FFFF?logo=yolo&logoColor=black)
![TensorFlow](https://img.shields.io/badge/Classifier-BiLSTM-FF6F00?logo=tensorflow&logoColor=white)
![TFLite](https://img.shields.io/badge/Deploy-TensorFlow_Lite-FF6F00?logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/status-research-blue)

</div>

---

## 📑 Table of Contents

- [About](#-about)
- [Models at a Glance](#-models-at-a-glance)
- [Visual Detector — YOLOv10n](#-visual-detector--yolov10n)
- [Text Classifier — BiLSTM](#-text-classifier--bilstm)
- [Results](#-results)
- [Repository Layout](#-repository-layout)
- [Data & Weights](#-data--weights)
- [Getting Started](#-getting-started)
- [Known Issues](#-known-issues)
- [License](#-license)

---

## 📖 About

**THESIS_1** pairs two independent models to flag unsafe content shown on a
screen. Each is trained separately and exported to **TensorFlow Lite** for
on-device / mobile inference, so moderation can run without a network round-trip.

- 🖼️ **Vision** — detects unsafe imagery (`nudity`, `gore`) vs `safe`.
- 💬 **Language** — detects unsafe **Tagalog/Taglish** text (`NSFW` vs `Safe`).

---

## 🧩 Models at a Glance

| Module | Modality | Architecture | Task | Labels | Export |
|--------|----------|--------------|------|--------|--------|
| [`yolov10nver6/`](yolov10nver6/) | 🖼️ Image | YOLOv10-nano | Object detection | `safe`, `nudity`, `gore` | ONNX · SavedModel · TFLite |
| [`lstm_nsfw/`](lstm_nsfw/) | 💬 Text | Bidirectional LSTM | Binary classification | `Safe`, `NSFW` | Keras · TFLite |

---

## 🖼️ Visual Detector — YOLOv10n

A YOLOv10-nano object detector that locates and flags unsafe visual content.

- **Notebook:** [`yolov10nver5(1).ipynb`](yolov10nver6/yolov10nver5%281%29.ipynb)
  — houses the `ScreenContentDetector` training pipeline and a `TFLiteInference`
  wrapper (developed on Google Colab).
- **Classes:** `safe`, `nudity`, `gore` — see [`labelmap.txt`](yolov10nver6/output/labelmap.txt).
- **Training:** 50 epochs · 640px input · batch 16 · GPU
  ([`args.yaml`](yolov10nver6/runs/train/screen_content_detector/args.yaml)).
- **Exports:** ONNX, TF SavedModel, TFLite (float16 / float32).

---

## 💬 Text Classifier — BiLSTM

A Bidirectional LSTM that classifies Tagalog/Taglish text as safe or unsafe.

- **Dataset:** ~11,110 labeled samples (`text,label`) — *not committed* (see [Data & Weights](#-data--weights)).
- **Architecture** ([`config.json`](lstm_nsfw/models/lstm_tagalog_nsfw/config.json)):

  | Hyperparameter | Value |
  |----------------|-------|
  | Vocabulary size | 10,000 |
  | Embedding dim | 128 |
  | LSTM units | 128 (bidirectional) |
  | Dense units | 64 |
  | Max sequence length | 128 |

- **Artifacts:** `best_model.keras`, `final_model.keras`, `tokenizer.pickle`, `model.tflite` (*not committed*).

---

## 📊 Results

### YOLOv10n — Visual Detector

| Metric | mAP50 | mAP50-95 |
|--------|:-----:|:--------:|
| **Validation** | ~0.21 | ~0.13 |

<sub>Full per-epoch log: [`results.csv`](yolov10nver6/runs/train/screen_content_detector/results.csv)</sub>

<div align="center">
<img src="yolov10nver6/runs/train/screen_content_detector/results.png" width="80%" alt="YOLO training results"/>
<br/>
<img src="yolov10nver6/runs/train/screen_content_detector/confusion_matrix.png" width="48%" alt="YOLO confusion matrix"/>
<img src="yolov10nver6/runs/train/screen_content_detector/BoxPR_curve.png" width="48%" alt="YOLO precision-recall curve"/>
</div>

### BiLSTM — Text Classifier

| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|:-------:|
| Safe  | 0.852 | 0.864 | 0.858 | 1283 |
| NSFW  | 0.811 | 0.795 | 0.803 | 939 |
| **Accuracy** | | | **0.835** | 2222 |

<sub>Full report: [`lstm_metrics.txt`](lstm_nsfw/metrics/lstm_metrics.txt)</sub>

<div align="center">
<img src="lstm_nsfw/models/lstm_tagalog_nsfw/confusion_matrix.png" width="48%" alt="LSTM confusion matrix"/>
<img src="lstm_nsfw/models/lstm_tagalog_nsfw/training_history.png" width="48%" alt="LSTM training history"/>
</div>

---

## 🗂️ Repository Layout

```
THESIS_1/
├── yolov10nver6/                 # 🖼️ visual NSFW detector (YOLOv10n)
│   ├── yolov10nver5(1).ipynb     #   training + TFLite inference notebook
│   ├── output/                   #   labelmap, eval metrics, inference results
│   └── runs/                     #   training/validation logs, plots, weights
└── lstm_nsfw/                    # 💬 text NSFW classifier (BiLSTM)
    ├── dataset.csv               #   (ignored) Tagalog text dataset
    ├── metrics/                  #   evaluation report
    └── models/lstm_tagalog_nsfw/ #   config + (ignored) weights
```

---

## 🔒 Data & Weights

Large binaries are **excluded from version control** via [`.gitignore`](.gitignore):

| Excluded | Patterns |
|----------|----------|
| 📦 Datasets | `lstm_nsfw/dataset.csv`, `*.npy` |
| 🧠 Weights / exports | `*.pt`, `*.keras`, `*.tflite`, `*.onnx`, `*.pb`, `*.pickle`, SavedModel `variables/` |
| 🖼️ Raw image dumps | `train_batch*` / `val_batch*` previews, per-image inference outputs |

> ⚠️ Image dumps can contain dataset imagery (incl. NSFW content) — keep them out of the repo.
> Metrics, configs, plots, and the notebook **are** tracked so results stay auditable.
> Share datasets and trained weights out-of-band (e.g. Google Drive); they regenerate by re-running the notebooks.

---

## 🚀 Getting Started

```bash
pip install ultralytics tensorflow scikit-learn pandas
```

**🖼️ Visual model**
Open the YOLO notebook and run the cells (designed for Colab; adjust the `data:`
path in `args.yaml` for local runs).

**💬 Text model**
```python
# Load tokenizer + model, then predict
import pickle, tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

tok = pickle.load(open("lstm_nsfw/models/lstm_tagalog_nsfw/tokenizer.pickle", "rb"))
model = tf.keras.models.load_model("lstm_nsfw/models/lstm_tagalog_nsfw/best_model.keras")

seq = pad_sequences(tok.texts_to_sequences(["ulol manigas ka"]), maxlen=128)
print("NSFW" if model.predict(seq)[0][0] > 0.5 else "Safe")
```

---

## 🐞 Known Issues

- **YOLO TFLite eval returns zeros.** The evaluation under
  [`output/metrics_results/`](yolov10nver6/output/metrics_results/) reports
  `precision/recall/mAP = 0.0`, while training metrics are non-zero — pointing to
  an output-parsing mismatch in `TFLiteInference` post-processing, not a failed model.
- **Low detection mAP** (~0.21 mAP50) — the visual detector needs more data /
  tuning before it's production-ready.
- **Mild class imbalance** in the text test split (1283 safe vs 939 NSFW).

---

## 📄 License

Academic thesis project — released for educational and research purposes.

<div align="center">
<sub>Built for content-moderation research · YOLOv10n + BiLSTM · TensorFlow Lite</sub>
</div>

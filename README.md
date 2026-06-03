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
│   ├── config.yaml               #   pipeline settings (paths, hyperparameters)
│   ├── requirements.txt          #   Colab-pinned dependencies
│   ├── output/                   #   labelmap, eval metrics, inference results
│   └── runs/                     #   training/validation logs, plots, weights
└── lstm_nsfw/                    # 💬 text NSFW classifier (BiLSTM)
    ├── config.yaml               #   model + data settings
    ├── requirements.txt          #   Colab-pinned dependencies
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

Each model ships its own pinned `requirements.txt` (Colab-friendly, Python 3.12):

```bash
pip install -r yolov10nver6/requirements.txt   # visual detector
pip install -r lstm_nsfw/requirements.txt      # text classifier
```

> On Google Colab most deps are preinstalled — for the YOLO notebook, the in-notebook
> `pip install ultralytics` cell is usually enough.

**🖼️ Visual model**
Open the YOLO notebook and run the cells (designed for Colab). Settings (dataset path,
classes, epochs, weights) live in [`config.yaml`](yolov10nver6/config.yaml) — upload it
next to the notebook on Colab to override the built-in defaults, or just run as-is.

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

**Fixed in the latest notebook** (the committed plots/CSVs under
[`runs/`](yolov10nver6/runs/) and [`output/`](yolov10nver6/output/) still reflect
the *old* run — rerun the notebook to refresh them):

- **Zeroed-out eval metrics.** The saved
  [`metrics_results/`](yolov10nver6/output/metrics_results/) files reported
  `precision/recall/mAP = 0.0` even though the validation log showed real numbers
  (e.g. mAP50 ≈ 0.20). Root cause was **not** TFLite post-processing — it was
  `ModelEvaluator.evaluate()` reading a non-existent `results.metrics` attribute and
  falling back to zeros. The evaluation runs on the PyTorch `best.pt`, not the TFLite
  model. Now reads the Ultralytics `results.box` (`mp`, `mr`, `map50`, `map`).
- **Training from scratch.** Training initialised from `yolov10n.yaml` (random weights),
  a major cause of the low mAP. Now starts from pretrained `yolov10n.pt` (transfer
  learning), configurable via [`config.yaml`](yolov10nver6/config.yaml).
- **TFLite metadata never attached.** `create_label_file()` ran *after* the TFLite
  export, so `labelmap.txt` didn't exist when metadata population was attempted. The
  label map is now created **before** export.

**Still open:**

- **Background-heavy dataset.** Most images carry no labels (train: ~7,477 of 10,958
  are backgrounds; val: ~1,239 of 1,547). In validation, `safe` and `gore` score
  P/R/mAP ≈ 0 while only `nudity` is learned (mAP50 ≈ 0.61) — the real driver of the
  low overall mAP. Modelling `safe` as a *boxed detection class* is also questionable.
- **`TFLiteInference` is a loader stub.** `infer()` returns the raw output tensor with
  no decoding/NMS/coordinate scaling, so there is no working end-to-end TFLite
  detection path yet — it only loads the model and reports the input shape.
- **Mild class imbalance** in the text test split (1283 safe vs 939 NSFW).

---

## 📄 License

Academic thesis project — released for educational and research purposes.

<div align="center">
<sub>Built for content-moderation research · YOLOv10n + BiLSTM · TensorFlow Lite</sub>
<br/>
<sub>README layout adapted from <a href="https://github.com/othneildrew/Best-README-Template">Best-README-Template</a> by Othneil Drew · badges by <a href="https://shields.io">Shields.io</a></sub>
</div>

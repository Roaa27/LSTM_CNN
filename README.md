# 🎬 Human Activity Recognition — CNN + BiLSTM

> **Deep Learning pipeline for video-based action classification on UCF101**  
> Combines spatial feature extraction via CNN with temporal sequence modeling via Bidirectional LSTM.

---

## 📌 Overview

This project presents an end-to-end **Human Activity Recognition (HAR)** system trained on the [UCF101](https://www.crcv.ucf.edu/data/UCF101/) benchmark dataset. The architecture couples a **convolutional feature extractor** applied per-frame with a **Bidirectional LSTM** that captures temporal dynamics across the sequence — enabling accurate classification of 15 distinct human activities.

| Property        | Value                          |
|-----------------|-------------------------------|
| Dataset         | UCF101 (15 selected classes)  |
| Frames/Video    | 20                            |
| Frame Resolution| 64 × 64 pixels                |
| Model           | CNN + BiLSTM                  |
| Classes         | 15                            |
| Framework       | TensorFlow / Keras            |

---

## 🎯 Supported Activity Classes

| # | Class | Description |
|---|-------|-------------|
| 0 | WalkingWithDog | Person walking with a dog |
| 1 | JumpingJack | Person jumping with arms and legs spread |
| 2 | PushUps | Person doing push-ups |
| 3 | Basketball | Person playing basketball |
| 4 | SoccerJuggling | Person juggling a soccer ball |
| 5 | VolleyballSpiking | Person spiking a volleyball |
| 6 | TennisSwing | Person swinging a tennis racket |
| 7 | Punch | Person punching |
| 8 | PlayingGuitar | Person playing guitar |
| 9 | PlayingPiano | Person playing piano |
| 10 | GolfSwing | Person playing golf |
| 11 | HorseRiding | Person riding a horse |
| 12 | SkateBoarding | Person skateboarding |
| 13 | Surfing | Person surfing on waves |
| 14 | Bowling | Person bowling |

---

## 🗂️ Project Structure

```
HAR-CNN-BiLSTM/
│
├── notebook.ipynb          # Main Colab notebook (training + evaluation)
├── README.md               # Project documentation
│
├── data/
│   └── UCF-101/            # Extracted dataset (downloaded at runtime)
│
└── outputs/
    ├── training_curves.png # Accuracy & loss plots
    └── confusion_matrix.png
```

---

## ⚙️ Data Preprocessing Pipeline

```
Raw Video (.avi)
      │
      ▼
Extract N_FRAMES (20) via uniform temporal sampling
      │
      ▼
Resize each frame → 64 × 64 px
      │
      ▼
Normalize pixel values to [0.0, 1.0]
      │
      ▼
Stack into tensor: (20, 64, 64, 3)
      │
      ▼
Feed into TimeDistributed(CNN) → BiLSTM
```

Up to **100 videos per class** are loaded, shuffled, and split 80/20 (train/test) with **stratification** to ensure balanced class representation.

---

## 🧠 Model Architecture

### 🔷 CNN Feature Extractor (applied per frame via `TimeDistributed`)

| Layer | Filters | Kernel | Activation | Notes |
|-------|---------|--------|------------|-------|
| Conv2D | 32 | 3×3 | ReLU | + BatchNorm + MaxPool |
| Conv2D | 64 | 3×3 | ReLU | + BatchNorm + MaxPool |
| Conv2D | 128 | 3×3 | ReLU | + BatchNorm + MaxPool |
| Conv2D | 256 | 3×3 | ReLU | + BatchNorm + MaxPool |
| Dropout | — | — | — | rate = 0.4 |
| Flatten | — | — | — | Output feature vector |

### 🔷 Temporal Model

| Layer | Units | Notes |
|-------|-------|-------|
| Bidirectional LSTM | 256 (×2) | `return_sequences=False` |
| Dropout | — | rate = 0.5 |

### 🔷 Classifier Head

| Layer | Units | Activation |
|-------|-------|------------|
| Dense | 64 | ReLU |
| BatchNormalization | — | — |
| Dropout | — | rate = 0.3 |
| Dense (output) | 15 | Softmax |

---

## 🛠️ Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Batch Size | 8 |
| Max Epochs | 60 |
| Validation Split | 20% |
| Random Seed | 42 |

### Callbacks

| Callback | Monitor | Configuration |
|----------|---------|---------------|
| `EarlyStopping` | `val_accuracy` | patience=12, restore best weights |
| `ReduceLROnPlateau` | `val_loss` | factor=0.5, patience=3, min_lr=1e-6 |

---

## 📊 Evaluation

The model is evaluated using:

- ✅ **Test Accuracy** and **Test Loss**
- 📊 **Confusion Matrix** (seaborn heatmap)
- 📋 **Classification Report** (precision, recall, F1-score per class)

---

## 🧪 Inference — Single Video Prediction

```python
test_video("/content/UCF-101/JumpingJack/v_JumpingJack_g01_c01.avi")
```

**Output:**
```
✅ Prediction  : Jumping Jack
🧠 Description : Person jumping with arms and legs spread
📊 Confidence  : 99.95%
```

The function also displays the first frame of the video with the prediction and confidence overlaid.

---

## 🚀 How to Run

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `notebook.ipynb`
3. Run cells sequentially — the notebook will:
   - Install dependencies (`opencv-python`, `unrar`)
   - Download & extract the UCF101 dataset (~7 GB)
   - Preprocess videos and build the dataset
   - Train the CNN-BiLSTM model
   - Evaluate and visualize results
4. Test on any video:
   ```python
   test_video("/content/UCF-101/<ClassName>/<video_file>.avi")
   ```


---

## 📦 Dependencies

```txt
tensorflow >= 2.x
opencv-python
numpy
matplotlib
seaborn
scikit-learn
```

Install via:
```bash
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn
```

---

## 📈 Results Summary

| Metric | Value |
|--------|-------|
| Test Accuracy || Test Accuracy | 81.98% |
| Test Loss     | 0.6909 |

> Replace placeholders with your actual results after running the notebook.

---



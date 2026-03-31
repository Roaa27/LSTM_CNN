# 🎥 Human Activity Recognition using CNN-BiLSTM (UCF101)

## 📌 Project Overview

This project implements a Deep Learning model for **Human Activity Recognition (HAR)** using a hybrid architecture of **CNN + Bidirectional LSTM (BiLSTM)**.

The model processes videos by extracting frames, learning spatial features using CNN, and capturing temporal relationships using BiLSTM.

---

## 🎯 Objective

To classify human activities from videos by combining:

* 🧠 **CNN** → Extract spatial features from each frame
* 🔁 **BiLSTM** → Learn temporal patterns across frames

---

## 📂 Dataset

We used the **UCF101 dataset**, a popular benchmark for action recognition.

🔗 Dataset link:
https://www.crcv.ucf.edu/data/UCF101/

---

## 🏷️ Selected Classes (15 Classes)

* WalkingWithDog
* JumpingJack
* PushUps
* Basketball
* SoccerJuggling
* VolleyballSpiking
* TennisSwing
* Punch
* PlayingGuitar
* PlayingPiano
* GolfSwing
* HorseRiding
* SkateBoarding
* Surfing
* Bowling

---

## ⚙️ Data Preprocessing

* Extracted frames from each video
* Selected **20 frames per video**
* Resized frames to **64×64 pixels**
* Normalized pixel values to **[0, 1]**
* Ensured fixed sequence length for LSTM input

---

## 🧠 Model Architecture

### 🔹 CNN Feature Extractor (from scratch)

* Conv2D (32 filters) + BatchNormalization + MaxPooling
* Conv2D (64 filters) + BatchNormalization + MaxPooling
* Conv2D (128 filters) + BatchNormalization + MaxPooling
* Dropout (0.3)
* Flatten

---

### 🔹 Temporal Model

* **Bidirectional LSTM (128 units)**
* Dropout (0.5)

---

### 🔹 Fully Connected Layers

* Dense (64 neurons, ReLU)
* BatchNormalization
* Dropout (0.3)
* Output Layer (15 classes, Softmax)

---

## 🛠️ Training Configuration

* Optimizer: Adam
* Loss Function: Categorical Crossentropy
* Batch Size: 8
* Epochs: 30
* Validation Split: 20%
* Callbacks:

  * EarlyStopping (monitor: val_accuracy, patience: 7)
  * ReduceLROnPlateau (reduce LR on plateau)

---

## 📊 Evaluation Metrics

The model is evaluated using:

* ✅ Accuracy
* 📉 Loss
* 📊 Confusion Matrix
* 📋 Classification Report

---


## 🧪 Prediction System

The model predicts the activity from a video and provides:

* 🎯 Predicted Class
* 🧠 Human-readable Description
* 📊 Confidence Score

---

### 🔍 Example Output

```
Prediction: Jumping Jack
Description: Person jumping with arms and legs spread
Confidence: 99.95%
```

---

## 🧠 Label Mapping

The model maps numeric predictions to readable labels using:

* `LABELS_MAP` → class names
* `DESCRIPTIONS` → activity descriptions

---

## 🚀 How to Run

1. Open **Google Colab**
2. Upload the notebook
3. Run all cells sequentially
4. Test using:

```python
test_video("/content/UCF-101/JumpingJack/v_JumpingJack_g01_c01.avi")
```



Developed as part of a Deep Learning project for Human Activity Recognition using video data.

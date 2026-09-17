# CodeAlpha Speech Emotion Recognition Internship

## Project Overview

This project implements a Speech Emotion Recognition system using audio features and a Convolutional Neural Network (CNN).

The system processes speech recordings, extracts Mel-Frequency Cepstral Coefficients (MFCCs), and classifies speech into eight emotional categories.

## Dataset

The project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) speech dataset.

- Audio files: 1,440
- Emotions: 8
- Audio format: WAV
- Sample rate: 48 kHz
- Features: MFCCs

### Emotion Classes

- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised

## Methodology

### 1. Data Preparation

The audio files were loaded and their emotion labels were extracted from the RAVDESS filename structure.

### 2. Feature Extraction

MFCC features were extracted using Librosa.

Each audio file was converted into a fixed-size feature representation:

`40 MFCC coefficients × 174 time frames`

Final feature array:

`X shape: (1440, 40, 174)`

### 3. Label Encoding

Emotion labels were encoded into numerical classes using Scikit-learn LabelEncoder.

### 4. Train/Test Split

The dataset was split using stratified sampling:

- Training samples: 1,152
- Test samples: 288
- Test size: 20%
- Random state: 42

### 5. CNN Model

A Convolutional Neural Network was developed using TensorFlow/Keras.

The architecture includes:

- Conv2D layers
- MaxPooling2D layers
- Flatten layer
- Dense layer with 128 neurons
- Dropout layer
- Softmax output layer with 8 classes

## Results

### Test Performance

- Test Accuracy: 45.49%
- Test Loss: 2.2853

### Classification Performance

| Emotion | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Angry | 0.45 | 0.58 | 0.51 |
| Calm | 0.57 | 0.71 | 0.64 |
| Disgust | 0.49 | 0.45 | 0.47 |
| Fearful | 0.43 | 0.46 | 0.44 |
| Happy | 0.38 | 0.26 | 0.31 |
| Neutral | 0.34 | 0.58 | 0.43 |
| Sad | 0.17 | 0.08 | 0.11 |
| Surprised | 0.59 | 0.59 | 0.59 |

## Model Analysis

The training curves indicate overfitting. Training accuracy increased to approximately 99%, while validation accuracy remained around 54%.

The confusion matrix shows that some emotions, particularly calm and surprised, were recognized more consistently, while sad was frequently confused with other emotions.

## Technologies

- Python
- TensorFlow
- Keras
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Project Structure

```text
CodeAlpha_SpeechEmotionRecognition_Internship/
│
├── data/
├── notebooks/
│   └── 01_speech_emotion_eda.ipynb
├── models/
│   └── speech_emotion_cnn.keras
├── results/
├── src/
├── .gitignore
└── README.md
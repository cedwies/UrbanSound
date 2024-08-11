# src/visualization/visualize.py

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_waveform(y, sr, title='Waveform'):
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

def plot_mel_spectrogram(mel_spec, sr, title='Mel Spectrogram'):
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(mel_spec, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

def plot_class_distribution(y, class_labels):
    plt.figure(figsize=(12, 6))
    sns.countplot(y)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(range(len(class_labels)), class_labels, rotation=45)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == '__main__':
    # Load some sample data
    y, sr = librosa.load(librosa.ex('choice'))
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Plot waveform and mel spectrogram
    plot_waveform(y, sr)
    plot_mel_spectrogram(log_mel_spec, sr)

    # Note: You would typically call other functions like plot_class_distribution,
    # plot_confusion_matrix, and plot_training_history from your main training script
    # or a separate analysis script after training the model.

# src/data/make_dataset.py

import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split

def load_audio_file(file_path):
    """Load an audio file and return its waveform and sample rate."""
    return librosa.load(file_path, sr=None)

def extract_features(y, sr):
    """Extract mel-spectrogram features from an audio signal."""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

def process_data(data_dir, metadata_file):
    """Process all audio files and create a dataset of mel-spectrograms."""
    metadata = pd.read_csv(metadata_file)
    features = []
    labels = []

    for index, row in metadata.iterrows():
        file_path = os.path.join(data_dir, f'fold{row.fold}', row.slice_file_name)
        y, sr = load_audio_file(file_path)
        mel_spec = extract_features(y, sr)
        features.append(mel_spec)
        labels.append(row.classID)

    return np.array(features), np.array(labels)

def main():
    data_dir = '../../data/raw/UrbanSound8K/audio'
    metadata_file = '../../data/raw/UrbanSound8K/metadata/UrbanSound8K.csv'
    output_dir = '../../data/processed'

    features, labels = process_data(data_dir, metadata_file)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Save the processed data
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

if __name__ == '__main__':
    main()

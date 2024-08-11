# src/features/build_features.py

import numpy as np
import pandas as pd
from src.data.preprocess import load_audio, pad_or_truncate, augment_audio, extract_features

def build_features(audio_paths, labels, sr=22050, duration=4, augment=False):
    """Build features for a list of audio files."""
    features = []
    processed_labels = []
    
    for audio_path, label in zip(audio_paths, labels):
        # Load and preprocess audio
        y = load_audio(audio_path, sr=sr, duration=duration)
        y = pad_or_truncate(y, sr * duration)
        
        if augment:
            y = augment_audio(y, sr)
        
        # Extract features
        mel_spec = extract_features(y, sr)
        features.append(mel_spec)
        processed_labels.append(label)
        
        if augment:
            # Add augmented version
            y_aug = augment_audio(y, sr)
            mel_spec_aug = extract_features(y_aug, sr)
            features.append(mel_spec_aug)
            processed_labels.append(label)
    
    return np.array(features), np.array(processed_labels)

def save_features(features, labels, output_path):
    """Save features and labels to a numpy file."""
    np.savez(output_path, features=features, labels=labels)

def load_features(input_path):
    """Load features and labels from a numpy file."""
    data = np.load(input_path)
    return data['features'], data['labels']

if __name__ == "__main__":
    # Test the functions
    test_audio_paths = ["path/to/audio1.wav", "path/to/audio2.wav"]
    test_labels = [0, 1]
    features, labels = build_features(test_audio_paths, test_labels, augment=True)
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Save and load features
    save_features(features, labels, "test_features.npz")
    loaded_features, loaded_labels = load_features("test_features.npz")
    print(f"Loaded features shape: {loaded_features.shape}")
    print(f"Loaded labels shape: {loaded_labels.shape}")

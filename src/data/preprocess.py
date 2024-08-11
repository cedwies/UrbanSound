# src/data/preprocess.py

import numpy as np
import librosa

def load_audio(file_path, sr=22050, duration=4):
    """Load an audio file and return its waveform."""
    y, _ = librosa.load(file_path, sr=sr, duration=duration)
    return y

def pad_or_truncate(y, target_length):
    """Pad or truncate an audio sample to the target length."""
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    return y

def augment_audio(y, sr, noise_factor=0.005, pitch_shift=1):
    """Apply data augmentation to the audio sample."""
    # Add noise
    noise = np.random.randn(len(y))
    augmented = y + noise_factor * noise
    
    # Pitch shift
    augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=pitch_shift)
    
    return augmented

def extract_features(y, sr):
    """Extract mel-spectrogram features from an audio signal."""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

if __name__ == "__main__":
    # Test the functions
    test_audio = "path/to/test/audio.wav"
    y = load_audio(test_audio)
    y_padded = pad_or_truncate(y, 22050 * 4)  # 4 seconds at 22050 Hz
    y_augmented = augment_audio(y_padded, 22050)
    features = extract_features(y_augmented, 22050)
    print(f"Feature shape: {features.shape}")

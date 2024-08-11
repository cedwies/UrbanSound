# src/models/predict_model.py

import numpy as np
from tensorflow.keras.models import load_model
from src.features.build_features import load_features, build_features

def load_trained_model(model_path):
    """Load a trained model from a file."""
    return load_model(model_path)

def predict_single_audio(model, audio_path, sr=22050, duration=4):
    """Predict the class of a single audio file."""
    features, _ = build_features([audio_path], [0], sr=sr, duration=duration)
    prediction = model.predict(features)
    return prediction[0]

def predict_batch(model, audio_paths, sr=22050, duration=4):
    """Predict classes for a batch of audio files."""
    features, _ = build_features(audio_paths, [0]*len(audio_paths), sr=sr, duration=duration)
    predictions = model.predict(features)
    return predictions

if __name__ == "__main__":
    # Test the functions
    model_path = "path/to/trained_model.h5"
    test_audio = "path/to/test_audio.wav"
    
    model = load_trained_model(model_path)
    
    # Single prediction
    single_prediction = predict_single_audio(model, test_audio)
    print(f"Single prediction: {single_prediction}")
    
    # Batch prediction
    test_audios = ["path/to/audio1.wav", "path/to/audio2.wav", "path/to/audio3.wav"]
    batch_predictions = predict_batch(model, test_audios)
    print(f"Batch predictions shape: {batch_predictions.shape}")

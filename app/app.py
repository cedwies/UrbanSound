# app/app.py

from flask import Flask, request, render_template, jsonify
import numpy as np
import librosa
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('../models/trained_model.h5')

# Define class labels
class_labels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
                'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

def extract_features(audio_file):
    """Extract mel-spectrogram features from an audio file."""
    y, sr = librosa.load(audio_file, duration=4)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            # Extract features from the uploaded audio file
            features = extract_features(file)
            
            # Reshape features for model input
            features = features.reshape(1, *features.shape, 1)
            
            # Make prediction
            prediction = model.predict(features)
            predicted_class = class_labels[np.argmax(prediction)]
            confidence = float(np.max(prediction))
            
            return jsonify({
                'predicted_class': predicted_class,
                'confidence': confidence
            })
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

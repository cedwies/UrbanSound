# tests/test_data.py

import unittest
import numpy as np
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data.make_dataset import load_audio_file, extract_features

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        # Create a dummy audio file for testing
        self.dummy_audio = np.sin(np.linspace(0, 1000, 22050)).astype(np.float32)
        np.save('dummy_audio.npy', self.dummy_audio)

    def tearDown(self):
        # Remove the dummy audio file
        os.remove('dummy_audio.npy')

    def test_load_audio_file(self):
        y, sr = load_audio_file('dummy_audio.npy')
        self.assertEqual(sr, 22050)  # Default sample rate
        self.assertEqual(len(y), 22050)  # 1 second of audio at 22050 Hz

    def test_extract_features(self):
        mel_spec = extract_features(self.dummy_audio, 22050)
        self.assertEqual(mel_spec.shape[0], 128)  # 128 mel bands
        self.assertTrue(mel_spec.shape[1] > 0)  # At least one time frame

if __name__ == '__main__':
    unittest.main()

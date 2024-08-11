# tests/test_features.py

import unittest
import numpy as np
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from features.build_features import build_features, save_features, load_features

class TestFeatures(unittest.TestCase):
    def setUp(self):
        # Create dummy audio data
        self.dummy_audio = np.sin(np.linspace(0, 1000, 22050)).astype(np.float32)
        np.save('dummy_audio.npy', self.dummy_audio)
        self.audio_paths = ['dummy_audio.npy']
        self.labels = [0]

    def tearDown(self):
        # Remove dummy files
        os.remove('dummy_audio.npy')
        if os.path.exists('test_features.npz'):
            os.remove('test_features.npz')

    def test_build_features(self):
        features, labels = build_features(self.audio_paths, self.labels)
        self.assertEqual(features.shape[0], 1)  # One audio file
        self.assertEqual(features.shape[1], 128)  # 128 mel bands
        self.assertTrue(features.shape[2] > 0)  # At least one time frame
        self.assertEqual(labels.shape[0], 1)  # One label

    def test_build_features_with_augmentation(self):
        features, labels = build_features(self.audio_paths, self.labels, augment=True)
        self.assertEqual(features.shape[0], 2)  # Original + augmented
        self.assertEqual(labels.shape[0], 2)  # Original + augmented

    def test_save_and_load_features(self):
        features, labels = build_features(self.audio_paths, self.labels)
        save_features(features, labels, 'test_features.npz')
        
        loaded_features, loaded_labels = load_features('test_features.npz')
        self.assertTrue(np.array_equal(features, loaded_features))
        self.assertTrue(np.array_equal(labels, loaded_labels))

if __name__ == '__main__':
    unittest.main()

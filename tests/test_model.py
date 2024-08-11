# tests/test_model.py

import unittest
import numpy as np
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.train_model import build_model

class TestModel(unittest.TestCase):
    def test_build_model(self):
        input_shape = (128, 64, 1)  # Example input shape
        num_classes = 10
        model = build_model(input_shape, num_classes)

        # Check model structure
        self.assertEqual(len(model.layers), 9)
        self.assertEqual(model.input_shape, (None, 128, 64, 1))
        self.assertEqual(model.output_shape, (None, 10))

    def test_model_predict(self):
        input_shape = (128, 64, 1)
        num_classes = 10
        model = build_model(input_shape, num_classes)

        # Generate a random input
        x = np.random.random((1, 128, 64, 1))
        
        # Get prediction
        prediction = model.predict(x)

        # Check prediction shape and values
        self.assertEqual(prediction.shape, (1, 10))
        self.assertTrue(np.allclose(np.sum(prediction), 1.0))  # Sum of probabilities should be close to 1
        self.assertTrue(np.all(prediction >= 0) and np.all(prediction <= 1))  # All probabilities should be between 0 and 1

if __name__ == '__main__':
    unittest.main()

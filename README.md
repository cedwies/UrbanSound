# Urban Sound Classification using Deep Learning

## Approach

This project uses a Convolutional Neural Network (CNN) architecture to classify urban sounds. The approach involves the following steps:

1. **Data Preparation**: Utilize the UrbanSound8K dataset, which contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes.
2. **Feature Extraction**: Convert audio files into mel-spectrograms, which serve as the input to our CNN model.
3. **Model Architecture**: Implement a CNN with multiple convolutional layers, max pooling, and dense layers.
4. **Training**: Use k-fold cross-validation to ensure robust model performance.
5. **Evaluation**: Assess the model using accuracy, precision, recall, and F1-score metrics.
6. **Deployment**: Develop a simple web interface for real-time sound classification.

## Project Structure

```
urban-sound-classification/
│
├── data/
│   ├── raw/                 # Raw audio files (not included in repo)
│   └── processed/           # Processed mel-spectrograms
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   └── 03_model_training_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── make_dataset.py
│   │   └── preprocess.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── train_model.py
│   │   └── predict_model.py
│   │
│   └── visualization/
│       └── visualize.py
│
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_model.py
│
├── app/
│   ├── static/
│   ├── templates/
│   └── app.py
│
├── models/
│   └── trained_model.h5
│
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```





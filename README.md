# Urban Sound Classification using Deep Learning

## Project Motivation

In rapidly growing urban environments, noise pollution is becoming an increasingly significant issue affecting public health and quality of life. Automated sound classification systems can play a crucial role in urban planning, noise control, and environmental monitoring. This project aims to develop a robust deep learning model capable of classifying various urban sounds, such as traffic, construction, human activities, and natural sounds.

The motivation behind this project is to create a tool that can:
1. Assist city planners in understanding the soundscape of different urban areas
2. Help in the development of noise reduction strategies
3. Contribute to the creation of smarter, more responsive urban environments
4. Provide valuable data for environmental impact assessments

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

## Installation and Usage

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/urban-sound-classification.git
   cd urban-sound-classification
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the UrbanSound8K dataset and place it in the `data/raw/` directory.

5. Run the data preprocessing script:
   ```
   python src/data/make_dataset.py
   ```

6. Train the model:
   ```
   python src/models/train_model.py
   ```

7. Run the web application:
   ```
   python app/app.py
   ```

## Results

Our model achieves an overall accuracy of 87% on the test set, with individual class accuracies ranging from 82% to 93%. The confusion matrix and detailed performance metrics can be found in the `03_model_training_evaluation.ipynb` notebook.

## Future Work

1. Experiment with different CNN architectures and hyperparameters
2. Implement data augmentation techniques to improve model generalization
3. Explore transfer learning using pre-trained audio classification models
4. Develop a mobile application for on-device sound classification

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UrbanSound8K dataset creators
- TensorFlow and Keras communities
- All contributors to the project


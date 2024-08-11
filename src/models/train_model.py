# src/models/train_model.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold

def load_data():
    """Load the preprocessed data."""
    X_train = np.load('../../data/processed/X_train.npy')
    y_train = np.load('../../data/processed/y_train.npy')
    X_test = np.load('../../data/processed/X_test.npy')
    y_test = np.load('../../data/processed/y_test.npy')
    return X_train, y_train, X_test, y_test

def build_model(input_shape, num_classes):
    """Build the CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(X_train, y_train, X_test, y_test):
    """Train the model using k-fold cross-validation."""
    num_classes = len(np.unique(y_train))
    input_shape = X_train.shape[1:]

    # Reshape data for CNN input
    X_train = X_train.reshape(X_train.shape[0], *input_shape, 1)
    X_test = X_test.reshape(X_test.shape[0], *input_shape, 1)

    # Convert labels to categorical
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # K-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        print(f'Fold {fold + 1}')
        model = build_model(input_shape, num_classes)
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        history = model.fit(X_train_fold, y_train_fold,
                            epochs=50,
                            batch_size=32,
                            validation_data=(X_val_fold, y_val_fold),
                            verbose=1)

        scores = model.evaluate(X_test, y_test, verbose=0)
        cv_scores.append(scores[1])
        print(f'Fold {fold + 1} accuracy: {scores[1] * 100:.2f}%')

    print(f'Mean accuracy: {np.mean(cv_scores) * 100:.2f}%')
    
    # Train final model on all training data
    final_model = build_model(input_shape, num_classes)
    final_model.compile(optimizer=Adam(learning_rate=0.001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    final_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    
    # Save the final model
    final_model.save('../../models/trained_model.h5')

def main():
    X_train, y_train, X_test, y_test = load_data()
    train_model(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()

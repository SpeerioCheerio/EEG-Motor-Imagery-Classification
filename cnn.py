import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def load_preprocessed_data():
    PROJECT_DIR = r"C:\Users\Alexander Speer\Desktop\Columbia Spring 2025\AML\Project1"
    data_file = os.path.join(PROJECT_DIR, "X_segmented_80Hz.npz")
   
    data = np.load(data_file)
    X = data["X_segmented"]
    y = data["y_segmented"]


    print("Preprocessed data loaded.")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y


#building the 1 diminsional CNN
def build_1d_cnn(input_shape):
    model = keras.Sequential([
        layers.Conv1D(filters=16, kernel_size=5, activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
       
        layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
       
        layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
       
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])
   
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
   
    return model


def main():
    # loading and preparing the data for use
    X, y = load_preprocessed_data()
   
    #reshaping everything for transport to the 1d
    X = np.transpose(X, (0, 2, 1))  # -----> should look like this -> (n_samples, 160, 64)
    print(f"Reshaped X for CNN: {X.shape}")


    # training and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
   
    # building the CNN
    input_shape = (X.shape[1], X.shape[2])  # (time_points, channels)
    model = build_1d_cnn(input_shape)
    model.summary()
   
    # Training the model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        verbose=1
    )
   
    # 5)evaluating the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")


if __name__ == '__main__':
    main()

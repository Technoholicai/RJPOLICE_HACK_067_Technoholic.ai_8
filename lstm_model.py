from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from data_processing import load_data
import numpy as np

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(256, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    # Pad sequences to a fixed length
    max_sequence_length = 30  # Assuming 30 frames per video
    X_train_padded = pad_sequences(X_train, maxlen=max_sequence_length, padding='post', dtype='float32')
    X_test_padded = pad_sequences(X_test, maxlen=max_sequence_length, padding='post', dtype='float32')

    # Compile the model
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_padded, y_train, epochs=10, batch_size=80, validation_data=(X_test_padded, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_padded, y_test)
    print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    num_videos = 2200
    X_train, X_test, y_train, y_test = load_data(num_videos=num_videos)

    # Modify the input shape passed to create_lstm_model
    model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    train_and_evaluate_model(model, X_train, y_train, X_test, y_test)

    # Evaluate additional metrics
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_binary))

    # Save the model
    model.save("lstm_model.h5")

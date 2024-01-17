import tensorflow as tf
from tensorflow.keras import layers, models
from data_processor import load_data, preprocess_data, split_data

# Set your data directory
data_directory = 'C:/Users/vansh/Desktop/DEEP/Test/Test'

# Load data
X, y = load_data(data_directory)

# Preprocess data
X, y = preprocess_data(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = split_data(X, y)

# Display some information about the dataset
print("Number of training samples:", len(X_train))
print("Number of testing samples:", len(X_test))
print("Shape of each image:", X_train[0].shape)

# Define the CNN model
def create_cnn_model(input_shape):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten layer
    model.add(layers.Flatten())

    # Dense (fully connected) layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))  # Two classes: real and fake

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Get input shape based on the first image in the training set
input_shape = X_train[0].shape

# Create the CNN model
model = create_cnn_model(input_shape)

# Display the model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save('cnn_model.h5')

# Load the saved model (Optional)
loaded_model = models.load_model('cnn_model.h5')


# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")


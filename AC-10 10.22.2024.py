#selam Kelil
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.matlib import repmat
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Dataset parameters
COPIES = 200           # Number of times to copy the song
WINDOW_LENGTH = 12      # Number of notes per input sequence

# Neural network parameters
NN_NODES = 15           # Number of nodes in the RNN
EPOCHS = 100            # Number of training epochs
BATCH_SIZE = None       # Default batch size (auto-handled by TensorFlow)

def dataset_from_song(song, copies, window_length):
    """Generate training data from repeated copies of the song."""
    # Repeat the song `copies` times to create more data points
    repeated_song = repmat(song, 1, copies)[0]
    num_windows = len(repeated_song) - window_length

    # Create input (X) and target (Y) datasets
    x_train, y_train = [], []
    for i in range(num_windows):
        x_train.append(repeated_song[i:i + window_length])
        y_train.append(repeated_song[i + window_length])

    # Convert to NumPy arrays and reshape for RNN input
    x_train = np.expand_dims(np.array(x_train, dtype='float32'), axis=-1)
    y_train = np.expand_dims(np.array(y_train, dtype='float32'), axis=-1)

    return x_train, y_train

# Define the song and generate the training data
song = np.array([72, 74, 76, 77, 79, 81, 83, 84])
x_train, y_train = dataset_from_song(song, COPIES, WINDOW_LENGTH)

# Build the RNN model
model = Sequential([
    SimpleRNN(NN_NODES, activation='relu', input_shape=(WINDOW_LENGTH, 1)),
    Dense(1)  # Single output for predicting the next note
])

# Compile the model
model.compile(
    loss='mean_squared_error',
    optimizer='adam'
)

# Early stopping callback to prevent overfitting
callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=100, restore_best_weights=True
)

# Train the model
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[callback]
)

# Evaluate the model on the training set
print("Finished training:")
model.evaluate(x_train, y_train)

# Make predictions on the training set
predictions = model.predict(x_train).round()
accuracy = 100 * np.mean(predictions == y_train)
print(f"Train set accuracy: {accuracy:.2f}%")

# Plot training predictions vs. actual values
def plot_predictions(predictions, y_true, title, filename):
    """Plot predictions vs. true values and save the figure."""
    plt.figure()
    index = np.arange(len(predictions))
    plt.plot(index, predictions, 'b', label='Predictions')
    plt.plot(index, y_true, 'r', label='True')
    plt.scatter(index, predictions, color='b')
    plt.scatter(index, y_true, color='r')
    plt.xlabel("Datapoint")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

plot_predictions(predictions[:50], y_train[:50], 
                 "Training Set Predictions", "Training.png")

# Plot training loss per epoch
plt.figure()
plt.plot(history.history['loss'], marker='o', linestyle='-', color='b')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.grid(True)
plt.savefig("Training_error.png")
plt.close()

# Test the model with transposed versions of the song
accuracies = []
for scale_steps in range(1, 32, 4):
    test_song = song + scale_steps
    x_test, y_test = dataset_from_song(test_song, copies=3, window_length=WINDOW_LENGTH)

    predictions = model.predict(x_test).round()
    acc = 100 * np.mean(predictions == y_test)
    accuracies.append(acc)

    print(f"Test set accuracy for scale step {scale_steps}: {acc:.2f}%")
    plot_predictions(predictions, y_test, 
                     f"Test Set (Scale {scale_steps})", f"Testing_{scale_steps}.png")

# Plot test accuracies for different scale steps
plt.figure(figsize=(10, 5))
plt.bar(range(len(accuracies)), accuracies, color='skyblue', width=0.5)
plt.xlabel('Test Case')
plt.ylabel('Accuracy (%)')
plt.title('Test Set Accuracy for Different Scale Steps')
plt.grid(axis='y')
plt.savefig("Testing_accuracies.png")
plt.close()

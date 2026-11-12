# ARCHIVAL SOURCE (Autumn 2024): file-system paths anonymized for sharing.
# Set REPLACE_WITH_PROJECT_ROOT to your local dataset path before running.
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

# Start timer
start_time = time.time()

# Set up data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from the combined directories
train_generator = train_datagen.flow_from_directory(
    r"REPLACE_WITH_PROJECT_ROOT\Combined\train",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)
validation_generator = val_datagen.flow_from_directory(
    r"REPLACE_WITH_PROJECT_ROOT\Combined\valid",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)
test_generator = test_datagen.flow_from_directory(
    r"REPLACE_WITH_PROJECT_ROOT\Combined\test",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Extract features and labels for training
train_features = []
train_labels = []
for data_batch, labels_batch in train_generator:
    train_features.append(data_batch)
    train_labels.append(labels_batch)
    if len(train_features) * train_generator.batch_size >= train_generator.samples:
        break

train_features = np.concatenate(train_features)
train_labels = np.concatenate(train_labels)
train_features = train_features.reshape((train_features.shape[0], -1))

# Extract features and labels for validation
val_features = []
val_labels = []
for data_batch, labels_batch in validation_generator:
    val_features.append(data_batch)
    val_labels.append(labels_batch)
    if len(val_features) * validation_generator.batch_size >= validation_generator.samples:
        break

val_features = np.concatenate(val_features)
val_labels = np.concatenate(val_labels)
val_features = val_features.reshape((val_features.shape[0], -1))

# Extract features and labels for testing
test_features = []
test_labels = []
for data_batch, labels_batch in test_generator:
    test_features.append(data_batch)
    test_labels.append(labels_batch)
    if len(test_features) * test_generator.batch_size >= test_generator.samples:
        break

test_features = np.concatenate(test_features)
test_labels = np.concatenate(test_labels)
test_features = test_features.reshape((test_features.shape[0], -1))

# Build the k-NN model
model = KNeighborsClassifier(n_neighbors=5)

# Train the model
model.fit(train_features, train_labels)

# Validate the model
val_predictions = model.predict(val_features)
val_accuracy = accuracy_score(val_labels, val_predictions)
print(f'Validation accuracy: {val_accuracy * 100:.2f}%')

# Test the model
test_predictions = model.predict(test_features)
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# End timer and print the total time taken
end_time = time.time()
total_time = end_time - start_time
print(f'Total time taken: {total_time:.2f} seconds')
# ARCHIVAL SOURCE (2024). Absolute local file-system paths replaced with a placeholder.
# Set REPLACE_WITH_PROJECT_ROOT to a local project directory before attempting to run.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
    class_mode='binary'
)
validation_generator = val_datagen.flow_from_directory(
    r"REPLACE_WITH_PROJECT_ROOT\Combined\valid",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    r"REPLACE_WITH_PROJECT_ROOT\Combined\test",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

for data_batch, labels_batch in train_generator:
    print("Data batch shape:", data_batch.shape)
    print("Labels batch shape:", labels_batch.shape)
    break

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training the model for 20 epochs
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Test if model is working
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {test_acc * 100:.2f}%')

# End timer and print the total time taken
end_time = time.time()
total_time = end_time - start_time
print(f'Total time taken: {total_time:.2f} seconds')
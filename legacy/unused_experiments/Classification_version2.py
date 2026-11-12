# ARCHIVAL / NOT USED in final results (Autumn 2024).
# Original absolute data paths replaced with REPLACE_WITH_PROJECT_ROOT.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up data generators for training, validation, and test sets
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from files
train_generator_marvel = train_datagen.flow_from_directory(
    r"REPLACE_WITH_PROJECT_ROOT\Marvel\train_marvel",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
validation_generator_marvel = val_datagen.flow_from_directory(
    r"REPLACE_WITH_PROJECT_ROOT\Marvel\valid_marvel",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
test_generator_marvel = test_datagen.flow_from_directory(
    r"REPLACE_WITH_PROJECT_ROOT\Marvel\test_marvel",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
for data_batch, labels_batch in train_generator_marvel:
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

# Training the model
model.fit(
    train_generator_marvel,
    steps_per_epoch=train_generator_marvel.samples // train_generator_marvel.batch_size,
    epochs=10,
    validation_data=validation_generator_marvel,
    validation_steps=validation_generator_marvel.samples // validation_generator_marvel.batch_size
)

# Test if model is working
test_loss, test_acc = model.evaluate(test_generator_marvel, steps=test_generator_marvel.samples // test_generator_marvel.batch_size)
print(f'Test accuracy: {test_acc * 100:.2f}%')
# ARCHIVAL / NOT USED in final results (Autumn 2024).
# Original absolute data paths replaced with REPLACE_WITH_PROJECT_ROOT.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# Build the CNN model with exactly 11 layers
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),  # 1
    MaxPooling2D((2, 2)),                                              # 2
    Conv2D(64, (3, 3), activation='relu'),                             # 3
    MaxPooling2D((2, 2)),                                              # 4
    Conv2D(128, (3, 3), activation='relu'),                            # 5
    MaxPooling2D((2, 2)),                                              # 6
    Conv2D(128, (3, 3), activation='relu'),                            # 7
    MaxPooling2D((2, 2)),                                              # 8
    Flatten(),                                                         # 9
    Dense(512, activation='relu'),                                     # 10
    Dense(1, activation='sigmoid')                                     # 11
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training the model for 15 epochs
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Test if model is working
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {test_acc * 100:.2f}%')


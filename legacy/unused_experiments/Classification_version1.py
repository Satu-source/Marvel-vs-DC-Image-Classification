# ARCHIVAL / NOT USED in final results (Autumn 2024).
# Original absolute data paths replaced with REPLACE_WITH_PROJECT_ROOT.
# Import needed libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up data generators for training, validation, and test sets
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from files
#TRAIN SET 
train_generator_marvel = train_datagen.flow_from_directory(
    r"REPLACE_WITH_PROJECT_ROOT\Marvel\train_marvel", # Path to training set directory
    target_size=(150, 150), # Resize images to 150x150
    batch_size=32,
    class_mode='binary' # Binary classification
)
#VALIDATION SET 
validation_generator_marvel = val_datagen.flow_from_directory(
    r"REPLACE_WITH_PROJECT_ROOT\Marvel\valid_marvel", # Path to validation set directory
    target_size=(150, 150), # Resize images to 150x150
    batch_size=32,
    class_mode='binary' # Binary classification
)
#TEST SET 
test_generator_marvel = test_datagen.flow_from_directory(
    r"REPLACE_WITH_PROJECT_ROOT\Marvel\test_marvel", # Path to test set directory
    target_size=(150, 150), # Resize images to 150x150
    batch_size=32,
    class_mode='binary' # Binary classification
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)), # Convolutional layer with 32 filters
    MaxPooling2D((2, 2)), # Max pooling layer
    Conv2D(64, (3, 3), activation='relu'), # Convolutional layer with 64 filters
    MaxPooling2D((2, 2)), # Max pooling layer
    Conv2D(128, (3, 3), activation='relu'), # Convolutional layer with 128 filters
    MaxPooling2D((2, 2)), # Max pooling layer
    Flatten(), # Flatten the 3D feature maps to 1D feature vectors
    Dense(512, activation='relu'), # Fully connected layer with 512 neurons
    Dense(1, activation='sigmoid') # Output layer with a sigmoid activation function for binary classification
])

# Compile the model
model.compile(optimizer='adam', # Use Adam optimizer
              loss='binary_crossentropy', # Binary crossentropy loss function
              metrics=['accuracy']) # Monitor accuracy during training

# Training the model
model.fit(
    train_generator_marvel,
    steps_per_epoch=train_generator_marvel.samples // train_generator_marvel.batch_size,
    epochs=10, # Number of epochs
    validation_data=validation_generator_marvel,
    validation_steps=validation_generator_marvel.samples // validation_generator_marvel.batch_size
)

# test if model is working
test_loss, test_acc = model.evaluate(test_generator_marvel, steps=test_generator_marvel.samples // test_generator_marvel.batch_size)
print(f'Test accuracy: {test_acc * 100:.2f}%')

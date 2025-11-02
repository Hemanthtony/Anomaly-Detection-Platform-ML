import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set dataset path (same as in train_model.py)
dataset_path = r'C:\Users\Hemanth\.cache\kagglehub\datasets\karagwaanntreasure\plant-disease-detection\versions\1\Dataset'

# Data preprocessing (same as training, but no augmentation for evaluation)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Load validation data
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Important for evaluation
)

# Load the trained model
model = tf.keras.models.load_model('plant_disease_model.h5')

# Evaluate the model on validation data
loss, accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // validation_generator.batch_size)

print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

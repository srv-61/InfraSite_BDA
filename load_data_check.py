import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to your dataset folder
train_dir = 'path_to_your_train_datasets_folder'
test_dir = 'path_to_your_test_datasets_folder'

# ImageDataGenerator to load and preprocess images
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images to 224x224 (or any size you prefer)
    batch_size=32,  # Batch size for training
    class_mode='binary'  # For binary classification, use 'binary'. For multi-class, use 'categorical'.
)

# Load testing images
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'  # Change based on your classification type
)

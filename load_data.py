import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Correct path using a raw string or forward slashes
train_dir = r'D:\SOHAN\7TH SEM\Big Data & Deep Learning\InfraSite_BDA\Datasets\Train Datasets'
test_dir = r'D:\SOHAN\7TH SEM\Big Data & Deep Learning\InfraSite_BDA\Datasets\Test Datasets'

# Set up ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create train generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
)

# Create test generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
)

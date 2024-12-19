import os
import shutil

def organize_images_into_class_folder(dataset_dir):
    class_folder = os.path.join(dataset_dir, 'default_class')
    os.makedirs(class_folder, exist_ok=True)

    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            src_path = os.path.join(dataset_dir, filename)
            dest_path = os.path.join(class_folder, filename)
            shutil.move(src_path, dest_path)

# Organize both train and test datasets
train_dir = r'D:\SOHAN\7TH SEM\Big Data & Deep Learning\InfraSite_BDA\Datasets\Train Datasets'
test_dir = r'D:\SOHAN\7TH SEM\Big Data & Deep Learning\InfraSite_BDA\Datasets\Test Datasets'

organize_images_into_class_folder(train_dir)
organize_images_into_class_folder(test_dir)

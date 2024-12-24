import os
import cv2
import torch
import matplotlib
matplotlib.use('TkAgg')  # Set TkAgg backend to ensure Matplotlib displays the pie chart correctly
import matplotlib.pyplot as plt

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Automatically loads the latest version

# Define the directory containing images
image_directory = 'D:/SOHAN/7TH SEM/Big Data & Deep Learning/InfraSite_BDA/Datasets/Object_Creation'

# Define valid image extensions
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# List all valid image files in the directory
image_files = [f for f in os.listdir(image_directory) if any(f.lower().endswith(ext) for ext in valid_extensions)]
print(f"Valid images found: {image_files}")

# Dictionary to store the counts of each object type across all images
overall_counts = {}

def count_objects_and_plot(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return

    # Perform inference using the YOLOv5 model
    results = model(image)

    # Get the class names (e.g., 'tree', 'building', 'road')
    class_names = results.names

    # Get the predictions (objects detected in the image)
    predictions = results.pandas().xywh[0]

    # Count the occurrences of each detected class (e.g., tree, building, road)
    class_counts = predictions['name'].value_counts()

    # Update the overall counts dictionary
    for class_name, count in class_counts.items():
        if class_name in overall_counts:
            overall_counts[class_name] += count
        else:
            overall_counts[class_name] = count

    # Print the counts of each class for the current image
    print(f"Object counts for {image_path}:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")

    # Display results on the image (bounding boxes and labels)
    results.show()

# Process each image file in the directory
for image_name in image_files:
    image_path = os.path.join(image_directory, image_name)
    count_objects_and_plot(image_path)

# Print overall counts
print("\nOverall Object Counts Across All Images:")
for class_name, count in overall_counts.items():
    print(f"{class_name}: {count}")

# Ensure we have data to plot
if overall_counts:
    print("Generating pie chart...")  # Debugging step

    # Generate pie chart
    labels = list(overall_counts.keys())
    sizes = list(overall_counts.values())

    print(f"Labels: {labels}")
    print(f"Sizes: {sizes}")

    # Create a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title('Overall Object Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.

    # Make sure the chart displays
    plt.show()

    # Save the chart as a file
    plt.savefig("pie_chart.png")
    print("Pie chart saved as pie_chart.png")

else:
    print("No objects detected. Pie chart not generated.")

from ultralytics import YOLO
import os
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Replace with your model

# Define the test image folder path
test_image_folder = r"D:\SOHAN\7TH SEM\Big Data & Deep Learning\InfraSite_BDA\Datasets\Test Datasets\test_image"

# Check if the folder exists
if not os.path.exists(test_image_folder):
    raise FileNotFoundError(f"{test_image_folder} does not exist")

# Define desired classes
desired_classes = ["car", "tree", "building", "toilet", "road", "truck", "chair", "person"]

# Initialize a dictionary to store the count of predicted classes
class_counts = {cls: 0 for cls in desired_classes}

# Process each image in the folder
for image_name in os.listdir(test_image_folder):
    image_path = os.path.join(test_image_folder, image_name)
    if os.path.isfile(image_path):
        # Run inference
        results = model(image_path)

        # Parse and filter results
        filtered_results = []
        for box in results[0].boxes.data:
            class_id = int(box[5])  # Get class index
            class_name = model.names[class_id]  # Map to class name
            confidence = float(box[4])  # Get confidence score
            if class_name in desired_classes:
                filtered_results.append((class_name, confidence))
                # Increment the class count
                class_counts[class_name] += 1

        # Display the results
        print(f"Results for {image_name}:")
        if filtered_results:
            for obj in filtered_results:
                print(f"  - {obj[0]} (Confidence: {obj[1]:.2f})")
        else:
            print("  - No desired objects detected.")
        print()

# Plot the pie chart for the predicted class distribution
labels = list(class_counts.keys())
sizes = list(class_counts.values())
colors = plt.cm.Paired.colors  # Optional: you can choose a different color scheme

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title("Predicted Class Distribution")
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
plt.show()

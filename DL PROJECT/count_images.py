import os

# Set dataset path (same as in train_model.py)
dataset_path = r'C:\Users\Hemanth\.cache\kagglehub\datasets\karagwaanntreasure\plant-disease-detection\versions\1\Dataset'

# Function to count images in each class
def count_images_per_class(dataset_path):
    class_counts = {}
    total_images = 0
    healthy_count = 0
    diseased_count = 0

    if os.path.exists(dataset_path):
        for class_name in os.listdir(dataset_path):
            class_dir = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_dir):
                image_count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                class_counts[class_name] = image_count
                total_images += image_count

                # Check if healthy or diseased
                if 'healthy' in class_name.lower():
                    healthy_count += image_count
                else:
                    diseased_count += image_count

    return class_counts, total_images, healthy_count, diseased_count

# Get counts
class_counts, total_images, healthy_count, diseased_count = count_images_per_class(dataset_path)

print("Total images:", total_images)
print("Healthy images:", healthy_count)
print("Diseased images:", diseased_count)
print("\nImages per class:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

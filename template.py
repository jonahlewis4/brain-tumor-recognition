import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Define the target size for the images (width, height)
target_size = (256, 256)


def load_dataset(data_dir):
    """
    Loads images and labels from the given directory.
    Assumes a structure like:
      data_dir/
         images/  - contains image files
         labels/  - contains text files with labels (starting with '1' or '0')
    """
    images_folder = os.path.join(data_dir, 'images')
    labels_folder = os.path.join(data_dir, 'labels')
    data = []
    labels = []

    for filename in os.listdir(images_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(images_folder, filename)
            base_name = os.path.splitext(filename)[0]
            label_file = os.path.join(labels_folder, base_name + '.txt')

            if os.path.exists(label_file):
                # Process the image
                try:
                    img = Image.open(img_path).convert('L')
                    img = img.resize(target_size)
                    img_array = np.array(img)
                    data.append(img_array)
                except Exception as e:
                    print(f"Could not process image {img_path}: {e}")
                    continue

                # Read the label from the text file
                try:
                    with open(label_file, 'r') as f:
                        content = f.read().strip()
                        # Assume label is the first character
                        label = 1 if content.startswith('1') else 0
                        labels.append(label)
                except Exception as e:
                    print(f"Could not read label from {label_file}: {e}")
            else:
                print(f"Label file {label_file} not found for image {img_path}")

    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    return data, labels


# Define paths for your training and testing data directories.
# Update these paths to where your data actually is.
train_data_dir = 'brain-tumor/train'  # Update this to your train directory path
test_data_dir = 'brain-tumor/test'  # Update this to your test directory path

# Load the datasets.
train_images, train_labels = load_dataset(train_data_dir)
test_images, test_labels = load_dataset(test_data_dir)

print(f"Loaded {len(train_images)} train images and {len(test_images)} test images.")

# Preprocess the data:
# Reshape images to include the channel dimension (grayscale) and normalize pixel values.
train_images = train_images.reshape(-1, target_size[0], target_size[1], 1) / 255.0
test_images = test_images.reshape(-1, target_size[0], target_size[1], 1) / 255.0

# Build a simple CNN model using TensorFlow's Keras API.
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model.
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model.
history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set.
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)

# --- Additional Stats & Visualization ---

# Plot training & validation accuracy and loss by epoch.
epochs_range = range(1, len(history.history['accuracy']) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train & Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], label='Train Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train & Validation Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Generate predictions on the test set.
test_pred_prob = model.predict(test_images)
test_pred = (test_pred_prob > 0.5).astype("int32")

# Print a classification report.
print("Classification Report:")
print(classification_report(test_labels, test_pred))

# Print a confusion matrix.
print("Confusion Matrix:")
print(confusion_matrix(test_labels, test_pred))

# Plot ROC curve.
fpr, tpr, thresholds = roc_curve(test_labels, test_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
import os
import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# Set up the directories where your images are stored.
# Make sure to update these paths to where your actual image folders are located.
tumor_dir = 'archive/yes'       # Directory with brains having tumors
non_tumor_dir = 'archive/no' # Directory with brains without tumors

# Define the target size for the images (width, height). Feel free to adjust as needed.
target_size = (256, 256)

# Lists to hold the image data and corresponding labels.
data = []
labels = []

def process_images(directory, label):
    """
    Loads images from the specified directory, converts them to greyscale,
    resizes them to the target size, and appends them to the global lists.
    """
    for filename in os.listdir(directory):
        # Check for common image file extensions.
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            try:
                # Open the image, convert it to greyscale, and resize it.
                img = Image.open(img_path).convert('L')
                img = img.resize(target_size)
                # Convert the image to a NumPy array.
                img_array = np.array(img)
                data.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Could not process {img_path}: {e}")

# Process images from both directories.
process_images(tumor_dir, label=1)      # Label 1 indicates a tumor.
process_images(non_tumor_dir, label=0)  # Label 0 indicates no tumor.

# Create a Pandas DataFrame with the image data and labels.
df = pd.DataFrame({
    'image': data,
    'label': labels
})

# Split the DataFrame into training (80%) and testing (20%) sets.
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=65)

print(f"Number of training samples: {len(train_df)}")
print(f"Number of testing samples: {len(test_df)}")

# Preprocess the training and testing data
# Stack the images and reshape to include channel dimension, then normalize.
def prepare_data(df, target_size):
    images = np.stack(df['image'].values)
    images = images.reshape(-1, target_size[0], target_size[1], 1) / 255.0
    labels = np.array(df['label'].values)
    return images, labels

train_images, train_labels = prepare_data(train_df, target_size)
test_images, test_labels = prepare_data(test_df, target_size)

# Using keras we compile a CNN.
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
history = model.fit(train_images, train_labels, epochs=30, batch_size=30, validation_split=0.1)

# Evaluate the model on the test set.
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)
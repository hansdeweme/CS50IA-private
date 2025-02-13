import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split

EPOCHS = 15
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

def main():
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    images, labels = load_data(sys.argv[1])
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    model = get_model()

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True
    )
    
    model.fit(datagen.flow(x_train, y_train, batch_size=32),
              epochs=EPOCHS, validation_data=(x_test, y_test))

    model.evaluate(x_test, y_test, verbose=2)
    
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

def load_data(data_dir):
    images, labels = [], []
    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                try:
                    img_path = os.path.join(category_path, filename)
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                    images.append(image)
                    labels.append(category)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    return images, labels

def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    main()

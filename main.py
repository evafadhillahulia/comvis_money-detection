import os
import cv2
import pygame
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D


# ================== ADJUST CONTRAST ==================
def adjust_contrast(image, alpha=1.5, beta=0):
    """
    Adjust the contrast and brightness of an image.
    Parameters:
        image (np.array): Input image (grayscale or color).
        alpha (float): Contrast control (1.0-3.0).
        beta (int): Brightness control (0-100).
    Returns:
        np.array: Adjusted image.
    """
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

# ================== PREPROCESSING GAMBAR ==================
def preprocess_image(image_path, size=(128, 128), contrast_alpha=1.5, contrast_beta=0):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image {image_path}")
        return None
    image = cv2.resize(image, size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Grayscale
    image = adjust_contrast(image, alpha=contrast_alpha, beta=contrast_beta)  # Adjust contrast
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image

def load_dataset(data_dir):
    images = []
    labels = []
    label_map = {}
    current_label = 0
    
    # Iterate through folders for each class
    for subdir in os.listdir(data_dir):
        subpath = os.path.join(data_dir, subdir)
        if os.path.isdir(subpath):
            print(f"Loading class '{subdir}'...")
            label_map[current_label] = subdir  # Map label to folder name
            
            # Load each image in the folder
            for file in os.listdir(subpath):
                if file.endswith(".jpg") or file.endswith(".png"):
                    img_path = os.path.join(subpath, file)
                    image = preprocess_image(img_path)
                    if image is not None:
                        images.append(image)
                        labels.append(current_label)
            current_label += 1
    
    return np.array(images), np.array(labels), label_map

# ================== BUILD CNN MODEL ==================
def build_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ================== TRAIN AND TEST ==================
def train_and_evaluate(model, X, y, label_map):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))
    print("Training complete.")
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    return history

# ================== PLAY AUDIO ==================
def play_audio(predicted_label, label_map, audio_dir="audio_files"):
    if predicted_label in label_map:
        audio_file = os.path.join(audio_dir, f"{label_map[predicted_label]}.mp3")
        print(f"Playing audio: {audio_file}")
        if os.path.exists(audio_file):
            pygame.mixer.init()
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():  # Wait for audio to finish playing
                continue
        else:
            print(f"Audio file not found for label '{label_map[predicted_label]}'")
    else:
        print(f"Label '{predicted_label}' not found in label_map.")

# ================== PREDICT IMAGE ==================
def predict_image(model, label_map, image_path, audio_dir="audio_files", contrast_alpha=1.5, contrast_beta=0):
    image = preprocess_image(image_path, contrast_alpha=contrast_alpha, contrast_beta=contrast_beta)
    if image is None:
        return None
    image = image.reshape(1, 128, 128, 1)
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)
    
    print(f"Predicted Nominal: {label_map[predicted_label]}")

    # Play the corresponding audio
    play_audio(predicted_label, label_map, audio_dir)
    
    return predicted_label

# ================== DISPLAY IMAGE ==================
def display_prediction(image_path, predicted_nominal):
    test_image = cv2.imread(image_path)
    if test_image is None:
        print(f"Error: Failed to load image {image_path}")
        return
    cv2.putText(test_image, f"Nominal: {predicted_nominal}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    plt.imshow(test_image_rgb)
    plt.axis('off')
    plt.show()

# ================== TEST PREDICTION ON DATASET ==================
def test_on_dataset(model, label_map, test_images_dir="dataset/test", contrast_alpha=1.5, contrast_beta=0):
    test_images = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) if f.endswith((".jpg", ".png"))]
    
    for image_path in test_images:
        predicted_label = predict_image(model, label_map, image_path, contrast_alpha=contrast_alpha, contrast_beta=contrast_beta)
        if predicted_label is not None:
            predicted_nominal = label_map[predicted_label]
            display_prediction(image_path, predicted_nominal)


# ================== MAIN ==================
if __name__ == "__main__":
    data_dir = "dataset/train"
    X, y, label_map = load_dataset(data_dir)
    X = X.reshape(X.shape[0], 128, 128, 1)
    y = to_categorical(y, num_classes=len(label_map))
    input_shape = (128, 128, 1)
    model = build_model(input_shape, len(label_map))
    train_and_evaluate(model, X, y, label_map)
    model.save("money_detector.keras")
    print("Model saved as 'money_detector.keras'.")

    # Test using images from a pre-existing test dataset
    test_on_dataset(model, label_map, test_images_dir="dataset/test", contrast_alpha=1.5, contrast_beta=0)

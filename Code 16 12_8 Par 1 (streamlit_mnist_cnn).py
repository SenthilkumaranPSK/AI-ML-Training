import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import numpy as np
from PIL import Image
import os
import random
import matplotlib.pyplot as plt

MODEL_PATH = "mnist_cnn_model.h5"
CLASSES = [str(i) for i in range(10)]

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        st.info("Loading saved MNIST model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        (_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        test_images = test_images.astype('float32') / 255.0
        test_images = test_images.reshape(-1, 28, 28, 1)
        test_labels = to_categorical(test_labels, num_classes=10)
    else:
        st.info("Training new MNIST model (first run only)...")
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.astype('float32') / 255.0
        test_images = test_images.astype('float32') / 255.0
        train_images = train_images.reshape(-1, 28, 28, 1)
        test_images = test_images.reshape(-1, 28, 28, 1)
        train_labels = to_categorical(train_labels, num_classes=10)
        test_labels = to_categorical(test_labels, num_classes=10)
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        # Early stopping callback for better generalization
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
        model.fit(
            train_images, train_labels,
            epochs=30, batch_size=128, validation_split=0.1, verbose=1,
            callbacks=[early_stop]
        )
        model.save(MODEL_PATH)
        st.success("Model trained & saved successfully!")
    return model, test_images, test_labels

def show_image_and_probs(image_array, predictions, true_label=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.imshow(image_array.reshape(28, 28), cmap='gray')
    ax1.axis('off')
    if true_label is not None:
        ax1.set_title(f"True Label: {true_label}")
    else:
        ax1.set_title("Uploaded Image")
    ax2.bar(range(10), predictions, color='skyblue')
    ax2.set_xticks(range(10))
    ax2.set_xticklabels(CLASSES)
    ax2.set_ylim([0, 1])
    ax2.set_title("Class Probabilities")
    st.pyplot(fig)

st.title("MNIST Digit Classification with CNN (High Accuracy)")
st.write("Upload a **handwritten digit image** or choose a **random test sample** to see predictions.")

model, test_images, test_labels = load_or_train_model()

# Calculate and display test accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
st.success(f"Test Accuracy: {test_acc * 100:.2f}%")

if st.button("Pick a Random Test Image"):
    idx = random.randint(0, len(test_images) - 1)
    sample_img = test_images[idx]
    sample_label = np.argmax(test_labels[idx])
    predictions = model.predict(sample_img.reshape(1, 28, 28, 1))[0]
    top_indices = predictions.argsort()[-3:][::-1]
    st.subheader(f"Predicted Digit: **{CLASSES[np.argmax(predictions)]}**")
    for i in top_indices:
        st.write(f"**{CLASSES[i]}** — {predictions[i]*100:.2f}%")
    show_image_and_probs(sample_img, predictions, true_label=sample_label)

uploaded_file = st.file_uploader("Upload an image (digit 0-9)...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Grayscale
    image_resized = image.resize((28, 28))
    img_array = np.array(image_resized).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    predictions = model.predict(img_array)[0]
    top_indices = predictions.argsort()[-3:][::-1]
    st.subheader(f"Predicted Digit: **{CLASSES[np.argmax(predictions)]}**")
    for i in top_indices:
        st.write(f"**{CLASSES[i]}** — {predictions[i]*100:.2f}%")
    show_image_and_probs(img_array, predictions)

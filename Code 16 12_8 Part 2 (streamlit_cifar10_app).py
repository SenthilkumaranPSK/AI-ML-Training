# streamlit_cifar10_app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os

# CIFAR-10 classes
CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

MODEL_PATH = "cifar10_cnn_model.h5"

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        st.info("Loading saved model...")
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        st.info("Training new model... please wait (first run only).")
        model = models.Sequential([
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),

            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),

            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])

        # Load CIFAR-10
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
        train_images, test_images = train_images / 255.0, test_images / 255.0

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        model.fit(
            train_images, train_labels,
            epochs=15,
            batch_size=64,
            validation_data=(test_images, test_labels),
            verbose=1
        )

        model.save(MODEL_PATH)
        st.success("Model trained & saved successfully!")
    return model

# Streamlit App
st.title("CIFAR-10 Image Classification with CNN")
st.write("Upload an image (any size) and get predictions. Model trained on the CIFAR-10 dataset.")

model = load_or_train_model()

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess
    image_resized = image.resize((32, 32))
    img_array = np.array(image_resized) / 255.0
    img_array = img_array.reshape(1, 32, 32, 3)

    # Prediction
    predictions = model.predict(img_array)[0]  # shape (10,)
    predicted_class = CLASSES[np.argmax(predictions)]

    # Top-3 Predictions
    top_indices = predictions.argsort()[-3:][::-1]

    st.subheader(f"Main Prediction: {predicted_class}")
    st.write("Top-3 Predictions")
    for i in top_indices:
        st.write(f"{CLASSES[i]} — {predictions[i] * 100:.2f}%")

    # Probability Chart
    st.bar_chart(predictions)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model (make sure the .h5 file is in the correct path)
MODEL = tf.keras.models.load_model("potatoes.h5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

st.set_page_config(page_title="Potato Disease Classification", layout="centered")
st.title("Potato Disease Classification")
st.write("Upload a potato leaf image to predict if it's Healthy, Early Blight, or Late Blight.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image: resize and (optionally) normalize
    # Change 256, 256 to your model's input size if different
    image = image.resize((256, 256))
    img_array = np.array(image)
    # Optional normalization (uncomment if your model expects input scaled 0-1)
    # img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, 0)  # Add batch dimension

    # Prediction
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    # Show results
    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
    # Optional: Probability chart
    st.bar_chart(predictions[0])

    st.markdown("---")
    st.caption("Built with Streamlit and TensorFlow")

import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
import network
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps

# Ambil path folder saat ini (lokasi file app.py)
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "trained_network.pkl")

# Load model
net = network.Network.load(MODEL_PATH)

st.set_page_config(page_title="Digit Classifier", layout="centered")
st.titel("Tebak Tipis-Tipis")

CANVAS_SIZE = 280
DIGIT_SIZE = 28

# Layout: Make columns equal width for equal sized displays
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown("#### Input")
    canvas_result = st_canvas(
        fill_color="#FFFFFF",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=True,
    )

# Helper: crop + center like MNIST using PIL
def crop_and_center_pil(pil_img):
    gray = pil_img.convert("L")  # Convert to grayscale
    img_array = np.array(gray)
    coords = np.argwhere(img_array < 255)  # find all non-white pixels

    if coords.shape[0] == 0:
        return np.zeros((28, 28))

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = img_array[y0:y1, x0:x1]

    # Resize to fit in 20x20 box
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    resized = Image.fromarray(cropped).resize(new_size, resample=Image.Resampling.BILINEAR)

    # Place resized digit in center of 28x28
    result = Image.new("L", (28, 28), 0)
    x_offset = (28 - new_size[0]) // 2
    y_offset = (28 - new_size[1]) // 2
    result.paste(resized, (x_offset, y_offset))
    return np.array(result) / 255.0

# Prediction
if st.button("Cek hasil"):
    if canvas_result.image_data is not None:
        rgba = canvas_result.image_data.astype(np.uint8)
        pil_img = Image.fromarray(rgba)
        processed = crop_and_center_pil(pil_img)
        input_vector = processed.reshape(784, 1)
        output = net.feedforward(input_vector)
        prediction = int(np.argmax(output))

        with right_col:
            st.markdown("#### Output")
            display_img = Image.fromarray((processed * 255).astype(np.uint8)).resize(
                (CANVAS_SIZE, CANVAS_SIZE), resample=Image.Resampling.NEAREST)
            st.image(display_img, width=CANVAS_SIZE, channels="L")

        st.markdown(f"### Prediksi: **{prediction}**")
        st.bar_chart(output.flatten(), height=200)
    else:
        st.warning("Tulis angka dulu brader.")

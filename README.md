
# 🧠 Digit Classifier Neural Network (MNIST + Streamlit)

This project is a simple yet powerful digit recognition app built with:

- 🖌️ Interactive drawing canvas (using `streamlit-drawable-canvas`)
- 🧠 Neural network from scratch (based on Michael Nielsen's book)
- 🧪 Trained on the MNIST dataset
- 🚀 Deployed with Streamlit Cloud

![Screenshot](./screenshot.png) <!-- optional: attach your own screenshot -->

---

## 🚀 Live Demo

👉 Try it live: [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)

---

## 📦 Features

- Draw digits (0–9) in the browser
- Neural network predicts your digit in real-time
- Preprocessing mimics MNIST format (28×28 grayscale)
- Visual confidence bar chart for predictions

---

## 🧠 Tech Stack

- Python
- NumPy
- Streamlit
- OpenCV
- Pillow
- Matplotlib
- Custom-built neural net (no TensorFlow/PyTorch)

---

## 🛠 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/digit-classifier-streamlit.git
cd digit-classifier-streamlit

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py                  # Streamlit main app
├── network.py              # Neural network class
├── mnist_loader.py         # MNIST dataset loader
├── trained_network.pkl     # Pre-trained model weights
├── requirements.txt
└── .streamlit/
    └── config.toml         # Streamlit theme config
```

---

## 🔐 Secrets (for deployment)

When deploying to Streamlit Cloud, define your secrets in the UI using TOML format:

```toml
DB_USERNAME = "youruser"
DB_TOKEN = "yourtoken"

[firebase]
api_key = "your_api_key"
```

Access them inside `app.py`:

```python
import streamlit as st
user = st.secrets["DB_USERNAME"]
api = st.secrets["firebase"]["api_key"]
```

---

## 🧾 Credits

- Based on [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
- Canvas powered by [`streamlit-drawable-canvas`](https://github.com/andfanilo/streamlit-drawable-canvas)

---

## 📄 License

MIT License © 2025 Azyzia

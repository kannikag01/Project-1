import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import tensorflow as tf

# ============================================================
# Load Image Model
# ============================================================
@st.cache_resource
def load_image_model():
    return load_model("fake_image_detector.h5", compile=False)

image_model = load_image_model()

# ============================================================
# Load Fine-Tuned BERT Text Model
# ============================================================
@st.cache_resource
def load_text_model():
    bert_checkpoint = r"C:\image\script\results\checkpoint-2638"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        bert_checkpoint, local_files_only=True
    )
    model.eval()
    return tokenizer, model

text_tokenizer, text_model = load_text_model()

# ============================================================
# Load Social Media LSTM Text Model
# (Corrected: MUST load .keras model)
# ============================================================
@st.cache_resource
def load_social_text_model():
    return tf.keras.models.load_model("social_text_model.keras", compile=False)

social_model = load_social_text_model()

# ============================================================
# Streamlit UI
# ============================================================
st.title("🧠 Fake Content Detector Suite")
st.write("Detect whether the content is AI-generated or real.")

st.sidebar.title("Select Mode")
mode = st.sidebar.radio("Choose Detection Type", ["Image", "Text (BERT)", "Social Media Text (LSTM)"])

# ============================================================
# IMAGE DETECTION
# ============================================================
if mode == "Image":
    st.header("🖼 Image Fake Detector")

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = image_model.predict(img_array)[0][0]
        label = "✅ Real Image" if prediction >= 0.6 else "❌ Fake Image"
        confidence = prediction if prediction >= 0.6 else 1 - prediction

        st.subheader(label)
        st.write(f"Confidence: {confidence * 100:.2f}%")


# ============================================================
# TEXT DETECTION WITH BERT
# ============================================================
elif mode == "Text (BERT)":
    st.header("📰 Text Fake-News Detector (BERT)")
    user_text = st.text_area("Enter news text or article content:")

    if st.button("Analyze Text"):
        if not user_text.strip():
            st.error("Please enter some text.")
        else:
            encodings = text_tokenizer(
                user_text.strip(),
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = text_model(**encodings)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            real_prob, fake_prob = probs[0], probs[1]

            if fake_prob > 0.6:
                label = "❌ Fake News"
                confidence = fake_prob
            elif real_prob > 0.6:
                label = "✅ Real News"
                confidence = real_prob
            else:
                label = "⚠ Uncertain Result"
                confidence = max(fake_prob, real_prob)

            st.subheader(label)
            st.write(f"Confidence: {confidence * 100:.2f}%")
            st.caption(f"(Real: {real_prob*100:.2f}%, Fake: {fake_prob*100:.2f}%)")


# ============================================================
# SOCIAL MEDIA TEXT CLASSIFIER (LSTM) WITH IMAGE + TEXT INPUT
# ============================================================
elif mode == "Social Media Text (LSTM)":
    st.header("📱 Social Media Fake-News Detector (LSTM Model)")

    # Image upload box (optional, not used by LSTM)
    uploaded_image = st.file_uploader(
        "Upload related image (optional)", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

    # Text box
    user_text = st.text_area("Enter the social media text to analyze:")

    # Analyze button
    if st.button("Analyze Social Media Content"):
        if not user_text.strip():
            st.error("Please enter some text.")
        else:
            try:
                # Model expects raw strings (TextVectorization is inside model)
                input_data = tf.constant([user_text], dtype=tf.string)

                # Run prediction
                prediction = social_model.predict(input_data)[0][0]

                real_prob = float(prediction)
                fake_prob = 1.0 - real_prob

                # Decision logic
                if real_prob >= 0.6:
                    label = "✅ Real News"
                    confidence = real_prob
                elif fake_prob >= 0.6:
                    label = "❌ Fake News"
                    confidence = fake_prob
                else:
                    label = "⚠ Uncertain Result"
                    confidence = max(real_prob, fake_prob)

                st.subheader(label)
                st.write(f"Confidence: {confidence * 100:.2f}%")
                st.caption(
                    f"(Real: {real_prob*100:.2f}%, Fake: {fake_prob*100:.2f}%)"
                )

            except Exception as e:
                st.error("Model error. Your LSTM model may not be saved correctly.")
                st.error(str(e))
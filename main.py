import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.convnext import preprocess_input, ConvNeXtXLarge
from PIL import Image
import pickle
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
import requests
from io import BytesIO
from datetime import datetime
import pytz

# Define PST timezone
pst = pytz.timezone("America/Los_Angeles")

# ESP32 IP Address
ESP32_IP = "https://golden-viable-salmon.ngrok-free.app"

# Sample image URL
IMAGE_URL = "https://osuwheat.com/wp-content/uploads/2013/05/armyworm-2-royer-2007.jpg"

CLASS_LABEL = [
    "Bird cherry-oat aphid", "Cerodonta Denticornis", "English grain aphid",
    "Green bug", "Longlegged spider mite", "Penthaleus major", "Wheat blossom midge",
    "Wheat phloeothrips", "Wheat sawfly"
]
IMG_SIZE = (224, 224)

@st.cache_resource
def get_ConvNeXtXLarge_model():
    """Load the ConvNeXtXLarge model with ImageNet weights."""
    base_model = ConvNeXtXLarge(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    x = GlobalAveragePooling2D()(base_model.output)
    return Model(inputs=base_model.input, outputs=x)

@st.cache_resource
def load_sklearn_models(model_path):
    """Load the trained MLP classification model."""
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)

def featurization(image, model):
    """Extract deep learning features from the image using ConvNeXtXLarge."""
    img = image.resize(IMG_SIZE)  # Resize image
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_batch = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_preprocessed = preprocess_input(img_batch)
    return model.predict(img_preprocessed)

# Load models
ConvNeXtXLarge_featurized_model = get_ConvNeXtXLarge_model()
classification_model = load_sklearn_models("mlp_best_model.pkl")

# Sidebar navigation
selected_tab = st.sidebar.radio("", ["Pest Prediction", "History", "About"])

removal_suggestion = ""

def bird_cherry_oat_aphid():
    removal_suggestion="For chemical control: Pyrethroids, Neonicotinoids, Organophosphates, Flonicamid\nFor biological control: Lacewigs, Hoverflies, Parasitic wasps"
    st.write(removal_suggestion)
def cerodonta_denticornis():
    removal_suggestion="For chemical control: Pyrethroids, Neonicotinoids, Spinosad, Abamectin\nFor biological control: Parasitic wasps, Predatory beetles"
    st.write(removal_suggestion)
def english_grain_aphid():
    removal_suggestion="For chemical control: Pyrethroids, Neonicotinoids, Sulfoxafor, Flonicamid\nFor biological control: Parasitic wasps"
    st.write(removal_suggestion)
def green_bug():
    removal_suggestion="For chemical control: Pyrethroids, Neonicotinoids, Organophosphates, Flonicamid, cultural\nFor biological control: Ladybugs, Lacewigs, Parasitic wasps"
    st.success(removal_suggestion)
def longlegged_spider_mite():
    removal_suggestion="For chemical control: Abamectin, Bifenazate, Hexythiazox, Spiromesifen\nFor Biological control: Predatory mites, Fungal pathogens"
    st.write(removal_suggestion)
def penthaleus_major():
    removal_suggestion="For chemical control: Organophosphates, Pyrethroids, Neonicotinoids, Etoxazole\nFor biological control: Predatory mites, Rove beetles"
    st.write(removal_suggestion)
def wheat_blossom_midge():
    removal_suggestion="For chemical control: Pyrethroids, Neonicotinoids, Spinosad, Chlorantraniliprole\nFor biological control: Parasitic wasps"
    st.write(removal_suggestion)
def wheat_phloeothrips():
    removal_suggestion="For chemical control: Pyrethroids, Neonicotinoids,Sulfoxafor, Spinosad\nFor biological control: Predatory mites, Minute pirate mites"
    st.write(removal_suggestion)
def wheat_sawfly():
    removal_suggestion="For Chemical control: Pyrethroids, Neonicotinoids, Spinosyns, Carbamates\nFor biological control: Parasitic wasps"
    st.write(removal_suggestion)

# About Tab
if selected_tab == "About":
    st.title("About")
    st.image(IMAGE_URL, use_container_width=True)
    st.write("This app connects to an ESP32 device to fetch and display predictions along with captured images.")
    st.write("It helps in monitoring and analyzing real-time data from the ESP32 device.")

# Predictions Tab
elif selected_tab == "Pest Prediction":
    st.title("Pest Prediction")

    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button("Get Prediction"):
        try:
            # Fetch prediction from ESP32
            pred_url = f"{ESP32_IP}/prediction"
            response = requests.get(pred_url)
            if response.status_code == 200:
                prediction = response.json().get("prediction", "").strip().lower()
                timestamp = datetime.now(pst).strftime("%m-%d-%Y %I:%M:%S %p")
                st.write(f"Timestamp: {timestamp}")
            else:
                st.error("Failed to get prediction data")
                prediction = ""

            # Fetch image from ESP32
            img_url = f"{ESP32_IP}/image"
            response = requests.get(img_url)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))

                # Feature extraction and classification
                with st.spinner("Processing..."):
                    image_features = featurization(image, ConvNeXtXLarge_featurized_model)
                    model_predict = classification_model.predict(image_features)
                    result_label = CLASS_LABEL[int(model_predict[0])]
                    st.success(f"Model Prediction: {result_label}")
                    st.image(image, use_container_width=True, caption="")

                    st.session_state.history.append((timestamp, result_label, image))
                    if result_label=="Bird Cherry-Oat Aphid":
                        bird_cherry_oat_aphid()
                    elif result_label=="Cerodonta Denticornis":
                        cerodonta_denticornis()
                    elif result_label=="English Grain Aphid":
                        english_grain_aphid()
                    elif result_label=="Green Bug":
                        green_bug()
                    elif result_label=="Longlegged Spider Mite":
                        longlegged_spider_mite()
                    elif result_label=="Penthaleus Major":
                        penthaleus_major()
                    elif result_label=="Wheat Blossom Midge":
                        wheat_blossom_midge()
                    elif result_label=="Wheat Phloeothrips":
                        wheat_phloeothrips()
                    elif result_label=="Wheat Sawfly":
                        wheat_sawfly()

            else:
                st.error("Failed to retrieve image")

        except Exception as e:
            st.error(f"Error: {e}")

# History Tab
elif selected_tab == "History":
    st.title("History")

    if "history" in st.session_state and st.session_state.history:
        for entry in st.session_state.history:
            timestamp, label, image = entry
            with st.expander(f"{label} at {timestamp}"):
                st.write(f"{removal_suggestion}")
                st.image(image, use_container_width=True)
    else:
        st.write("No history available.")
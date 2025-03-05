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
chem_control_text = "#### 🧪 **Chemical Control:**"
bio_control_text = "#### 🦠 **Biological Control:**"

def bird_cherry_oat_aphid():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(chem_control_text)
        st.markdown("- Pyrethroids  \n- Neonicotinoids  \n- Organophosphates  \n- Flonicamid")

    with col2:
        st.markdown(bio_control_text)
        st.markdown("- Lacewings  \n- Hoverflies  \n- Parasitic wasps")

def cerodonta_denticornis():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(chem_control_text)
        st.markdown("- Pyrethroids  \n- Neonicotinoids  \n- Spinosad  \n- Abamectin")

    with col2:
        st.markdown(bio_control_text)
        st.markdown("- Parasitic wasps  \n- Predatory beetles")

def english_grain_aphid():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(chem_control_text)
        st.markdown("- Pyrethroids  \n- Neonicotinoids  \n- Sulfoxaflor  \n- Flonicamid")

    with col2:
        st.markdown(bio_control_text)
        st.markdown("- Parasitic wasps")

def green_bug():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(chem_control_text)
        st.markdown("- Pyrethroids  \n- Neonicotinoids  \n- Organophosphates  \n- Flonicamid")

    with col2:
        st.markdown(bio_control_text)
        st.markdown("- Ladybugs  \n- Lacewings  \n- Parasitic wasps")

def longlegged_spider_mite():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(chem_control_text)
        st.markdown("- Abamectin  \n- Bifenazate  \n- Hexythiazox  \n- Spiromesifen")

    with col2:
        st.markdown(bio_control_text)
        st.markdown("- Predatory mites  \n- Fungal pathogens")

def penthaleus_major():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(chem_control_text)
        st.markdown("- Organophosphates  \n- Pyrethroids  \n- Neonicotinoids  \n- Etoxazole")

    with col2:
        st.markdown(bio_control_text)
        st.markdown("- Predatory mites  \n- Rove beetles")

def wheat_blossom_midge():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(chem_control_text)
        st.markdown("- Pyrethroids  \n- Neonicotinoids  \n- Spinosad  \n- Chlorantraniliprole")

    with col2:
        st.markdown(bio_control_text)
        st.markdown("- Parasitic wasps")

def wheat_phloeothrips():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(chem_control_text)
        st.markdown("- Pyrethroids  \n- Neonicotinoids  \n- Sulfoxaflor  \n- Spinosad")

    with col2:
        st.markdown(bio_control_text)
        st.markdown("- Predatory mites  \n- Minute pirate bugs")

def wheat_sawfly():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(chem_control_text)
        st.markdown("- Pyrethroids  \n- Neonicotinoids  \n- Spinosyns  \n- Carbamates")

    with col2:
        st.markdown(bio_control_text)
        st.markdown("- Parasitic wasps")

def suggestions(result_label):
    st.markdown("### **Pest Removal Suggestions:**")
    if result_label.lower()=="bird cherry-oat aphid":
        bird_cherry_oat_aphid()
    elif result_label.lower()=="cerodonta denticornis":
        cerodonta_denticornis()
    elif result_label.lower()=="english grain aphid":
        english_grain_aphid()
    elif result_label.lower()=="green bug":
        green_bug()
    elif result_label.lower()=="longlegged spider mite":
        longlegged_spider_mite()
    elif result_label.lower()=="penthaleus major":
        penthaleus_major()
    elif result_label.lower()=="wheat blossom midge":
        wheat_blossom_midge()
    elif result_label.lower()=="wheat phloeothrips":
        wheat_phloeothrips()
    elif result_label.lower()=="wheat sawfly":
        wheat_sawfly()

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
                st.markdown(f"### Timestamp: {timestamp}")
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
                    # Get probability scores 
                    probabilities = classification_model.predict_proba(image_features)[0] 
                    max_prob_index = np.argmax(probabilities) 
                    result_label = CLASS_LABEL[max_prob_index] 
                    confidence_score = probabilities[max_prob_index] * 100 # Convert to percentage
                    #result_label = CLASS_LABEL[int(model_predict[0])]
                    if confidence_score >= 50:
                        st.success(f"### Pest: {result_label}")
                        st.markdown(f"#### Confidence Score: {confidence_score}")
                        st.image(image, use_container_width=True, caption="")
                        st.session_state.history.insert(0,(timestamp, result_label, image))
                        suggestions(result_label)
                    else:
                        st.success(f"No wheat pest detected")
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
            with st.expander(f"🪲 {label} at {timestamp}"):
                st.write(f"{removal_suggestion}")
                st.image(image, use_container_width=True)
                suggestions(label)
    else:
        st.write("No history available.")
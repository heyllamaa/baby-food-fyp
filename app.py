import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. THE BRAIN: Load your trained MobileNetV2 model
@st.cache_resource
def load_brain():
    return tf.keras.models.load_model('augmented_model_80_20.h5')

model = load_brain()

# 2. THE FACE: Build the user interface on the screen
st.title("👶 PureeCheck AI")
st.subheader("Infant Food Texture Safety Classifier")

camera_photo = st.camera_input("Point camera at baby food and tap to capture")

# 3. THE CONNECTION: Pass the photo to the Brain
if camera_photo is not None:
    img = Image.open(camera_photo).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    
    prediction_scores = model.predict(img_array)
    final_class_index = np.argmax(prediction_scores)
    
    # 4. THE OUTPUT: Display the clinical safety alerts
    st.divider()
    if final_class_index == 0:
        st.success("✅ SAFE TEXTURE: Smooth Puree (Suitable for 4-6+ months)")
    elif final_class_index == 1:
        st.success("✅ SAFE TEXTURE: Mashed/Lumpy (Suitable for 6-9+ months)")
    elif final_class_index == 2:
        st.error("⚠️ CAUTION: Solid/Minced texture - Requires active chewing")
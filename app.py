import streamlit as st
import torch
from PIL import Image
from timeit import default_timer as timer
from model import create_effnetb2_model

# Page config
st.set_page_config(page_title="FoodVision Mini", page_icon="🍕")

class_names = ['pizza', 'steak', 'sushi']

# Cache model so it loads only once
@st.cache_resource
def load_model():
    model, transforms = create_effnetb2_model()
    model.load_state_dict(
        torch.load(
            'pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth',
            map_location=torch.device('cpu')
        )
    )
    model.eval()
    return model, transforms

effnetb2, effnetb2_transforms = load_model()

st.title("FoodVision Mini 🍕🥩🍣")
st.write("A fine-tuned EfficientNetB2 classifier to detect pizza, steak, or sushi.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    start_time = timer()
    transformed_img = effnetb2_transforms(img).unsqueeze(0)

    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(transformed_img), dim=1)[0]

    pred_time = round(timer() - start_time, 4)

    st.subheader(f"Predictions (took {pred_time}s):")
    for idx, class_name in enumerate(class_names):
        prob = float(pred_probs[idx])
        st.write(f"**{class_name.capitalize()}**: {prob * 100:.2f}%")
        st.progress(prob)

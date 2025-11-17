import streamlit as st
import os
import torch
import cv2 as cv
import numpy as np
from PIL import Image
import sys
from io import BytesIO  # Moved import to top

# Add the parent directory to path to import modules
# Ensure 'models' and 'utils' folders are in the parent directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.definitions.transformer_net import TransformerNet
import utils.utils as utils

# Page config
st.set_page_config(
    page_title="Minimal Style Transfer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_BINARIES_PATH = os.path.join(os.path.dirname(__file__), 'models', 'binaries')
IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406])
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225])

# --- Core ML/Processing Functions (Unchanged) ---

@st.cache_resource
def load_model(model_path, device):
    """Load and cache the stylization model"""
    stylization_model = TransformerNet().to(device)
    training_state = torch.load(model_path, map_location=device)
    state_dict = training_state["state_dict"]
    stylization_model.load_state_dict(state_dict, strict=True)
    stylization_model.eval()
    # Return model, not training_state, to save memory if unused
    return stylization_model

def prepare_image(image, target_width, device):
    """Prepare uploaded image for model input"""
    img = np.array(image)
    current_height, current_width = img.shape[:2]
    new_width = target_width
    new_height = int(current_height * (new_width / current_width))
    img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    mean = torch.tensor(IMAGENET_MEAN_1, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD_1, dtype=torch.float32).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor.to(device)

def post_process_image(img_tensor):
    """Convert model output back to displayable image"""
    img = img_tensor.cpu().numpy()[0]
    mean = IMAGENET_MEAN_1.reshape(-1, 1, 1)
    std = IMAGENET_STD_1.reshape(-1, 1, 1)
    img = (img * std) + mean
    img = (np.clip(img, 0., 1.) * 255).astype(np.uint8)
    img = np.moveaxis(img, 0, 2)
    return img

def get_available_models():
    """Get list of available model files"""
    if not os.path.exists(MODEL_BINARIES_PATH):
        return []
    models = [f for f in os.listdir(MODEL_BINARIES_PATH)
              if f.endswith('.pth') or f.endswith('.pt')]
    return sorted(models)

def get_model_display_name(model_name):
    """Convert model filename to display name"""
    name_map = {
        'candy.pth': '🍬 Candy',
        'starry_v3.pth': '🌟 Starry Night',
        'mosaic_4e5_e2.pth': '🎨 Mosaic',
        'edtaonisl_9e5_33k.pth': '🌈 Abstract'
    }
    return name_map.get(model_name, model_name.split('.')[0].replace('_', ' ').title())

# --- Main Streamlit App ---

def main():
    st.title("🎨 Neural Style Transfer")
    st.write("Transform your photos into artistic masterpieces using standard Streamlit components.")

    # Check if models directory exists
    available_models = get_available_models()
    if not available_models:
        st.error(f"⚠️ No models found in {MODEL_BINARIES_PATH}")
        st.info("Please ensure you have downloaded the pretrained models to the models/binaries folder.")
        return

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        st.subheader("1. Choose Style Model")
        model_options = {get_model_display_name(m): m for m in available_models}
        selected_display = st.selectbox(
            "Choose your artistic style:",
            list(model_options.keys())
        )
        selected_model = model_options[selected_display]
        
        # Image width slider
        st.subheader("2. Adjust Image Size")
        img_width = st.slider(
            "Output width (pixels):",
            min_value=256,
            max_value=1024,
            value=500,
            step=64,
            help="Larger sizes take more time ⏱️"
        )
        
        # Device info
        st.subheader("Device Info")
        device_icon = "🚀" if device.type == "cuda" else "💻"
        st.info(f"{device_icon} Computing on: {device.type.upper()}")

    # --- Main Content Area ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 Upload Your Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Original Image", use_column_width=True)

    with col2:
        st.header("✨ Stylized Result")
        
        if uploaded_file is not None:
            # Stylize button
            if st.button(f"🎨 Transform with {selected_display}"):
                with st.spinner("✨ Creating your masterpiece..."):
                    try:
                        # Load model
                        model_path = os.path.join(MODEL_BINARIES_PATH, selected_model)
                        stylization_model = load_model(model_path, device)
                        
                        # Prepare image
                        img_tensor = prepare_image(image, img_width, device)
                        
                        # Stylize
                        with torch.no_grad():
                            stylized_tensor = stylization_model(img_tensor)
                        
                        # Post-process
                        stylized_img = post_process_image(stylized_tensor)
                        
                        # Display result
                        st.image(stylized_img, caption="Stylized Image", use_column_width=True)
                        st.success("✅ Your artwork is ready!")
                        

                        # Download button
                        result_pil = Image.fromarray(stylized_img)
                        buf = BytesIO()
                        result_pil.save(buf, format='JPEG', quality=95)
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="💾 Download Artwork",
                            data=byte_im,
                            file_name=f"stylized_{selected_model.split('.')[0]}.jpg",
                            mime="image/jpeg"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error during stylization: {str(e)}")
                        st.info("💡 Try reducing the image width or using a different model.")
        else:
            # Placeholder
            st.info("👈 Upload an image and click 'Transform' to see the result here.")

if __name__ == "__main__":
    main()
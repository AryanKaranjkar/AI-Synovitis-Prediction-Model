import io
import json
from pathlib import Path

import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms


st.set_page_config(page_title="Synovitis Classifier", layout="centered")

# ---------- Config ----------
DEFAULT_ARCH = "efficientnetv2_rw_m.agc_in1k"
DEFAULT_NUM_CLASSES = 2
DEFAULT_IMG_SIZE = 512
DEFAULT_CLASS_NAMES = ["NoSynovitis", "Synovitis"]

# ---------- Preprocessing ----------
class SquarePadToWhite:
    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas

def build_transform(img_size: int = DEFAULT_IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        SquarePadToWhite(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

# ---------- Model loading ----------
@st.cache_resource(show_spinner=True)
def load_model(weights_path: str, arch: str, num_classes: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(weights_path, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state

    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, device

# ---------- Prediction ----------
@torch.no_grad()
def predict_image(model, device, image: Image.Image, img_size: int, tta: bool = False):
    tf = build_transform(img_size)
    x = tf(image).unsqueeze(0).to(device)

    if not tta:
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].tolist()
            pred = int(torch.argmax(logits, dim=1).item())
        return probs, pred

    # simple TTA: identity + hflip + vflip + hvflip
    flips = [lambda z: z, torch.flip, torch.flip, torch.flip]
    dims = [(), (3,), (2,), (2, 3,)]
    agg = torch.zeros(DEFAULT_NUM_CLASSES, device=device)
    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
        for f, d in zip(flips, dims):
            z = f(x, d) if d else x
            logits = model(z)
            agg += torch.softmax(logits, dim=1)[0]
    agg /= len(flips)
    probs = agg.tolist()
    pred = int(torch.argmax(agg).item())
    return probs, pred

# ---------- Sidebar ----------
st.sidebar.title("Settings")

# Model file
default_model_path = "model_hothead.pt"
# model_path = st.sidebar.text_input("Model file", value=default_model_path, help="Path to weights file (state_dict or {'model': state_dict})") 
model_path = st.sidebar.selectbox(
"Model File",
["model_hothead.pt", "model_footloose.pt"], help="Path to weights file (state_dict or {'model': state_dict})",placeholder = "select"
)
# Try to auto-read model_info.json if present
arch = DEFAULT_ARCH
num_classes = DEFAULT_NUM_CLASSES
info_path = Path("model_info.json")
if info_path.exists():
    try:
        info = json.loads(info_path.read_text())
        arch = info.get("model_name", arch)
        num_classes = int(info.get("num_classes", num_classes))
    except Exception:
        pass

arch = st.sidebar.text_input("Model architecture", value=arch, help="timm model name")
img_size = st.sidebar.number_input("Image size", min_value=224, max_value=1024, value=DEFAULT_IMG_SIZE, step=32)
use_tta = st.sidebar.checkbox("Enable TTA (flips)", value=False)
st.sidebar.caption("Device: **{}**".format("CUDA" if torch.cuda.is_available() else "CPU"))

# ---------- Main UI ----------
st.title("🩺 Synovitis Classifier (Standalone)")
st.write("Upload an image. The app will pad to square (white), resize to the chosen size, and classify.")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","tif","tiff"])

# Load model lazily when needed
model = None
device = "cpu"
load_error = None
if Path(model_path).exists():
    try:
        model, device = load_model(model_path, arch, num_classes)
    except Exception as e:
        load_error = str(e)
else:
    load_error = f"Model file not found at: {model_path}"

if load_error:
    st.error(load_error)
else:
    if uploaded is not None:
        # Read image
        img_bytes = uploaded.read()
        image = Image.open(io.BytesIO(img_bytes))

        # Show original and preprocessed preview
        st.subheader("Preview")
        st.image(image, caption="Original", use_column_width=True)

        st.subheader("Prediction")
        with st.spinner("Running inference..."):
            probs, pred = predict_image(model, device, image, img_size=img_size, tta=use_tta)

        class_names = DEFAULT_CLASS_NAMES[:num_classes]
        label = class_names[pred] if pred < len(class_names) else str(pred)
        st.success(f"**Prediction:** {label}")

    else:
        st.info("👆 Upload an image to begin.")

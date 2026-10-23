import hashlib
import io
import os
from pathlib import Path
import socket
from typing import Any

import requests
import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision import transforms
from torchvision.models import vgg16

# Configure optimal PyTorch intra-op threads for CPU execution
if hasattr(os, "cpu_count") and os.cpu_count():
    torch.set_num_threads(os.cpu_count())



# ----------------------------- App configuration -----------------------------
st.set_page_config(
    page_title="PolymerCode AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
# Find models directory in either local frontend/models or project root models/
MODEL_DIR = ROOT / "models" if (ROOT / "models").exists() else ROOT.parent / "models"
CLASS_NAMES = ["HDPE", "LDPE", "OTHER", "PET", "PP", "PS", "PVC"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
API_BASE = "http://127.0.0.1:8000"

MODEL_OPTIONS = {
    "VGG-16 (Recommended · 65.4% Test Acc · 0.64 F1)": {"key": "vgg16", "file": "vgg16_best.pt"},
    "Custom CNN (Baseline · 44.4% Test Acc · 0.41 F1)": {"key": "custom_cnn", "file": "custom_cnn_best.pt"},
}



RESIN_GUIDE: dict[str, dict[str, Any]] = {
    "PET": {
        "code": 1,
        "full_name": "Polyethylene terephthalate",
        "steps": [
            "Empty the item completely and separate it from other materials.",
            "Rinse away residue; remove caps or labels if local guidance requires it.",
            "Place it in the collection stream that accepts this PET item.",
            "At the recycling facility, PET is sorted, shredded, washed, and reprocessed into flakes.",
        ],
        "note": "Acceptance varies for bottles, trays, coloured PET, and mixed packaging.",
    },
    "HDPE": {
        "code": 2,
        "full_name": "High-density polyethylene",
        "steps": [
            "Empty the container and separate it from other resin types.",
            "Rinse off residue and handle caps or labels according to local guidance.",
            "Place the item in a collection stream that accepts this HDPE product.",
            "At the recycling facility, HDPE is sorted, ground, washed, and reprocessed.",
        ],
        "note": "Acceptance depends on the item and local recycling facilities.",
    },
    "PVC": {
        "code": 3,
        "full_name": "Polyvinyl chloride",
        "steps": [
            "Keep PVC separate from ordinary mixed-plastic recycling.",
            "Identify the specific product and check for an approved specialist collection route.",
            "Deliver it only to a facility that explicitly accepts that PVC product.",
            "Do not burn or melt PVC at home.",
        ],
        "note": "PVC is not accepted in many household plastic streams. Never burn or heat it at home.",
    },
    "LDPE": {
        "code": 4,
        "full_name": "Low-density polyethylene",
        "steps": [
            "Separate LDPE film or bags from rigid plastic containers.",
            "Empty and clean the film as required by the collection program.",
            "Use a dedicated film or flexible-plastic drop-off point if available.",
            "If the program does not accept the item, follow local waste guidance.",
        ],
        "note": "Plastic films often need a separate collection stream and may not belong in curbside bins.",
    },
    "PP": {
        "code": 5,
        "full_name": "Polypropylene",
        "steps": [
            "Empty the item and separate PP from other plastic types.",
            "Rinse away food or product residue.",
            "Check whether the local program accepts this PP item.",
            "Accepted PP is sorted, washed, shredded, and reprocessed into recycled material.",
        ],
        "note": "Acceptance varies by item shape, additives, and local sorting capability.",
    },
    "PS": {
        "code": 6,
        "full_name": "Polystyrene",
        "steps": [
            "Identify whether the item is rigid polystyrene or expanded foam.",
            "Keep it clean and dry if required by the collection service.",
            "Use a dedicated polystyrene or foam collection point if available.",
            "If no specialist route accepts it, follow local waste guidance. Do not burn it.",
        ],
        "note": "Polystyrene foam is not accepted by many curbside recycling programs.",
    },
    "OTHER": {
        "code": 7,
        "full_name": "Other or mixed resin category",
        "steps": [
            "Check the product label or documentation for the specific resin or material blend.",
            "Do not assume it can follow the PET, HDPE, or another single-resin route.",
            "Ask the local recycling authority or a specialist take-back program about suitable handling.",
            "If the material cannot be identified or accepted, follow local waste guidance.",
        ],
        "note": "Code 7 covers varied plastics and multi-layer materials; the code alone does not guarantee recyclability.",
    },
}


# ------------------------------- Model classes -------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def build_model(model_key: str, num_classes: int = 7) -> nn.Module:
    if model_key == "custom_cnn":
        return CustomCNN(num_classes=num_classes)
    if model_key in {"vgg16", "vgg16_balanced"}:
        model = vgg16(weights=None)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
        return model
    raise ValueError(f"Unknown model: {model_key}")


@st.cache_resource(show_spinner=False)
def load_model(model_key: str, checkpoint_path_str: str, modified_time: float):
    """Load local checkpoint and return model + preprocessing as fallback."""
    checkpoint_path = Path(checkpoint_path_str)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: models/{checkpoint_path.name} in {MODEL_DIR}."
        )

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = None
    saved_classes = None
    saved_class_to_idx = None
    if isinstance(checkpoint, dict):
        for state_key in ("model_state_dict", "state_dict", "model"):
            candidate = checkpoint.get(state_key)
            if isinstance(candidate, dict) and candidate:
                state_dict = candidate
                break
        saved_classes = checkpoint.get("classes") or checkpoint.get("class_names")
        saved_class_to_idx = checkpoint.get("class_to_idx")
        if state_dict is None and checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a supported model state dictionary.")

    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    if saved_class_to_idx:
        classes = [name for name, _ in sorted(saved_class_to_idx.items(), key=lambda pair: pair[1])]
    elif isinstance(saved_classes, (list, tuple)):
        classes = list(saved_classes)
    else:
        classes = CLASS_NAMES.copy()

    classes = [str(name).strip().upper() for name in classes]
    model = build_model(model_key, num_classes=len(classes))
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cpu":
        model.to(memory_format=torch.channels_last)
    model.eval()

    # Pre-warm model to eliminate first-inference cold start latency
    with torch.inference_mode():
        dummy = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device)
        if device.type == "cpu":
            dummy = dummy.to(memory_format=torch.channels_last)
        _ = model(dummy)

    preprocessing = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return model, classes, device, preprocessing


def predict_local(image: Image.Image, model: nn.Module, classes: list[str], device: torch.device, preprocessing):
    tensor = preprocessing(image.convert("RGB")).unsqueeze(0).to(device)
    if device.type == "cpu":
        tensor = tensor.to(memory_format=torch.channels_last)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()
    ranking = torch.argsort(probabilities, descending=True).tolist()
    results = []
    for class_index in ranking:
        label = classes[class_index]
        results.append({
            "label": label,
            "probability": float(probabilities[class_index].item()),
        })
    return results


@st.cache_data(ttl=15, show_spinner=False)
def check_fastapi_backend() -> bool:
    try:
        # Non-blocking fast socket check to avoid 1.2s timeout on reruns
        with socket.create_connection(("127.0.0.1", 8000), timeout=0.12):
            res = requests.get(f"{API_BASE}/api/v1/health", timeout=0.4)
            return res.status_code == 200
    except Exception:
        return False


def predict_via_fastapi(image: Image.Image, model_key: str) -> list[dict[str, Any]]:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    files = {"image": ("symbol.jpg", buf, "image/jpeg")}
    data = {"model": model_key}

    res = requests.post(f"{API_BASE}/api/v1/predict", files=files, data=data, timeout=8.0)
    if res.status_code != 200:
        detail = res.json().get("detail", f"Backend error ({res.status_code})")
        raise RuntimeError(f"FastAPI Backend: {detail}")

    payload = res.json()
    ranked = payload.get("ranked_predictions", [])
    results = []
    for item in ranked:
        results.append({
            "label": item["class"],
            "probability": float(item["score"]),
        })
    return results


# --------------------------------- Styling ------------------------------------
st.markdown(
    """
    <style>
      :root {
        --pc-bg: #000000;
        --pc-panel: #101F2B;
        --pc-panel-2: #142735;
        --pc-panel-soft: #0C1A24;
        --pc-text: #EAF2F5;
        --pc-muted: #9AAEB8;
        --pc-line: #263D49;
        --pc-accent: #A78BFA;
        --pc-accent-strong: #C4B5FD;
        --pc-accent-soft: #261F3E;
        --pc-warn: #E4B45C;
        --pc-green: #34D399;
      }
      html, body, [data-testid="stAppViewContainer"], .stApp {
        background: radial-gradient(ellipse at 8% 0%, rgba(139,92,246,.10), transparent 34%),
                    radial-gradient(ellipse at 92% 12%, rgba(196,181,253,.055), transparent 28%),
                    var(--pc-bg) !important;
        color: var(--pc-text) !important;
      }
      [data-testid="stAppViewContainer"] .main { background: transparent !important; }
      ::selection { background: rgba(167,139,250,.35); color: #FFFFFF; }
      [data-testid="stHeader"] { background: transparent !important; }
      [data-testid="stToolbar"] { right: 1rem; }
      .block-container { max-width: 1320px; padding-top: .35rem !important; padding-bottom: 1.4rem; }
      #MainMenu, footer { visibility: hidden; }
      .stApp, .stApp p, .stApp label, .stApp span, .stApp div,
      [data-testid="stMarkdownContainer"] { color: var(--pc-text); }
      [data-testid="stMarkdownContainer"] p { line-height: 1.55; }
      h1, h2, h3, h4 { color: var(--pc-text) !important; letter-spacing: -.025em; }
      hr { border-color: var(--pc-line) !important; }

      .pc-header { display:flex; align-items:center; justify-content:space-between; margin:0 0 .9rem; padding:0 0 .7rem; border-bottom:1px solid rgba(167,139,250,.20); }
      .pc-brand { display:flex; align-items:center; gap:11px; }
      .pc-mark { width:38px; height:38px; flex:0 0 38px; border-radius:12px; display:grid; place-items:center; background:linear-gradient(145deg,#D8CCFF,#8B5CF6); color:#04151C !important; font-size:19px; font-weight:950; box-shadow:0 6px 18px rgba(167,139,250,.20); }
      .pc-title { margin:0; font-weight:850; font-size:1.3rem; letter-spacing:-.04em; line-height:1.1; color:var(--pc-text) !important; }
      .pc-pill { display:flex; align-items:center; gap:7px; padding:5px 12px; border-radius:99px; font-size:.78rem; font-weight:600; border:1px solid rgba(167,139,250,.3); background:#141F30; }
      .pc-pill-dot { width:8px; height:8px; border-radius:50%; }
      .pc-pill.online .pc-pill-dot { background:#10B981; box-shadow:0 0 8px #10B981; }
      .pc-pill.offline .pc-pill-dot { background:#F59E0B; }
      .pc-section { font-size:1.06rem; font-weight:800; margin:0 0 .9rem; color:var(--pc-text); }
      .pc-small { color:var(--pc-muted) !important; font-size:.82rem; line-height:1.5; }
      .pc-panel { border:1px solid rgba(148,163,184,.16); border-radius:18px; padding:1.15rem; background:linear-gradient(145deg,rgba(19,31,45,.96),rgba(12,24,36,.96)); margin-bottom:.85rem; box-shadow:0 16px 36px rgba(0,0,0,.20); backdrop-filter:blur(8px); }
      .pc-prediction { display:flex; align-items:center; gap:15px; padding:.75rem 0 .6rem; }
      .pc-code { flex:0 0 70px; height:70px; width:70px; border-radius:17px; display:grid; place-items:center; background:var(--pc-accent-soft); border:1px solid #514075; color:var(--pc-accent-strong) !important; font-size:1.9rem; font-weight:900; box-shadow:inset 0 1px rgba(255,255,255,.025); }
      .pc-polymer { margin:0; font-size:1.65rem; font-weight:850; line-height:1.2; letter-spacing:-.035em; color:var(--pc-text) !important; }
      .pc-full { margin:.35rem 0 0; color:var(--pc-muted) !important; font-size:.86rem; }
      .pc-step { display:flex; align-items:flex-start; gap:11px; padding:.68rem 0; border-bottom:1px solid rgba(154,174,184,.13); font-size:.87rem; line-height:1.55; color:#DCE7EC !important; }
      .pc-step:last-child { border-bottom:0; padding-bottom:.1rem; }
      .pc-stepnum { flex:0 0 25px; height:25px; border-radius:8px; display:grid; place-items:center; background:var(--pc-accent-soft); border:1px solid #514075; color:var(--pc-accent-strong) !important; font-size:.76rem; font-weight:850; }
      .pc-note { margin-top:.95rem; border-radius:12px; padding:.8rem .9rem; background:linear-gradient(135deg,#211D32,#192332); border:1px solid rgba(167,139,250,.22); color:#D6D1E5 !important; font-size:.81rem; line-height:1.55; }
      .pc-confidence-head { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin:.9rem 0 .55rem; font-size:.86rem; color:var(--pc-muted) !important; }
      .pc-confidence-value { color:var(--pc-accent-strong) !important; font-size:1.05rem; font-weight:850; }
      .pc-muted { color:var(--pc-muted) !important; }
      .pc-prob-row { display:flex; align-items:center; justify-content:space-between; padding:4px 0; font-size:.82rem; }
      .pc-prob-name { font-weight:650; }
      .pc-prob-pct { color:var(--pc-accent-strong); font-weight:700; }

      /* Native Streamlit widgets */
      .stSelectbox label, .stRadio label, .stFileUploader label, .stCameraInput label { color:var(--pc-muted) !important; font-size:.82rem !important; font-weight:650 !important; }
      div[data-baseweb="select"] > div { background:#0D1D27 !important; border:1px solid var(--pc-line) !important; border-radius:12px !important; color:var(--pc-text) !important; min-height:45px; }
      div[data-baseweb="select"] * { color:var(--pc-text) !important; }
      ul[role="listbox"] { background:var(--pc-panel) !important; border:1px solid var(--pc-line) !important; }
      li[role="option"] { background:var(--pc-panel) !important; color:var(--pc-text) !important; }
      li[role="option"]:hover, li[aria-selected="true"] { background:var(--pc-accent-soft) !important; }
      div[role="radiogroup"] { gap:.5rem; }
      div[role="radiogroup"] label { background:#0D1D27; border:1px solid var(--pc-line); border-radius:10px; padding:.48rem .75rem; color:var(--pc-text) !important; }
      div[role="radiogroup"] label:has(input:checked) { background:var(--pc-accent-soft); border-color:#6D5AA8; }
      div[data-testid="stFileUploader"] section { border:1.5px dashed #65538D !important; border-radius:16px !important; background:linear-gradient(145deg,#17172A,#101A28) !important; padding:1.35rem 1rem !important; }
      div[data-testid="stFileUploader"] section:hover { border-color:var(--pc-accent) !important; }
      div[data-testid="stFileUploader"] section > div { color:var(--pc-text) !important; }
      div[data-testid="stFileUploader"] small { color:var(--pc-muted) !important; }
      div[data-testid="stFileUploader"] button, div[data-testid="stCameraInput"] button { background:#201B32 !important; color:var(--pc-text) !important; border:1px solid #4A3F6A !important; border-radius:9px !important; }
      div[data-testid="stFileUploader"] button:hover, div[data-testid="stCameraInput"] button:hover { border-color:var(--pc-accent) !important; color:var(--pc-accent-strong) !important; }
      div[data-testid="stImage"] img { border-radius:15px; border:1px solid var(--pc-line); }
      div.stButton > button[kind="primary"] { background:#8B5CF6 !important; color:#FFFFFF !important; border:0 !important; border-radius:12px !important; min-height:2.85rem; font-weight:850 !important; box-shadow:0 8px 22px rgba(139,92,246,.30); transition:background .15s ease, box-shadow .15s ease; }
      div.stButton > button[kind="primary"]:hover { background:#7C3AED !important; color:#FFFFFF !important; border:0 !important; box-shadow:0 10px 26px rgba(139,92,246,.38); }
      div.stButton > button[kind="secondary"] { background:#142631 !important; color:var(--pc-text) !important; border:1px solid var(--pc-line) !important; border-radius:12px !important; }
      div.stButton > button:disabled { opacity:.45; }
      div[data-testid="stProgress"] > div { background:#2D2545 !important; border-radius:99px; }
      div[data-testid="stProgress"] > div > div { background:linear-gradient(90deg,#8B5CF6,#D8CCFF) !important; border-radius:99px; }
      div[data-testid="stAlert"] { background:#192A34 !important; border:1px solid #394C56 !important; border-radius:12px !important; color:var(--pc-text) !important; }
      div[data-testid="stAlert"] p { color:var(--pc-text) !important; }
      div[data-testid="stSpinner"] p { color:var(--pc-muted) !important; }
      @media(max-width:700px) { .block-container { padding-top:.2rem !important; } .pc-title { font-size:1.2rem; } .pc-header { margin-bottom:.8rem; padding-bottom:.6rem; } .pc-panel { padding:1rem; } .pc-polymer { font-size:1.4rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header with FastAPI backend status indicator
backend_online = check_fastapi_backend()
pill_class = "online" if backend_online else "offline"
pill_text = "FastAPI Backend Online" if backend_online else "Local PyTorch Engine"

st.markdown(
    f"""
    <div class="pc-header">
      <div class="pc-brand">
        <div class="pc-mark">P</div>
        <h1 class="pc-title">PolymerCode AI</h1>
      </div>
      <div class="pc-pill {pill_class}">
        <span class="pc-pill-dot"></span>
        <span>{pill_text}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([0.95, 1.05], gap="large")

with left:
    st.markdown('<div class="pc-section">Image input</div>', unsafe_allow_html=True)
    selected_model_name = st.selectbox("Select Model Architecture", list(MODEL_OPTIONS.keys()), index=0)
    selected_model = MODEL_OPTIONS[selected_model_name]

    source = st.radio(
        "Image source",
        ["Upload / drag and drop", "Camera"],
        horizontal=True,
        label_visibility="collapsed",
        key="source_choice",
    )

    selected_file = None
    if source == "Upload / drag and drop":
        selected_file = st.file_uploader(
            "Drop image here or browse",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
            key="polymer_upload",
        )
    else:
        selected_file = st.camera_input(
            "Capture resin symbol",
            label_visibility="collapsed",
            key="polymer_camera",
        )

    current_image = None
    current_hash = None
    if selected_file is not None:
        try:
            raw_bytes = selected_file.getvalue()
            current_hash = hashlib.sha256(raw_bytes).hexdigest()
            img = ImageOps.exif_transpose(Image.open(selected_file)).convert("RGB")
            # Downscale large images (e.g. 12MP phone captures) to speed up display and processing
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
            current_image = img
            st.image(current_image, use_container_width=True)
        except (UnidentifiedImageError, OSError, ValueError):
            st.error("Unable to read this image. Try another image.")
            current_image, current_hash = None, None

    predict_clicked = st.button(
        "Identify polymer",
        type="primary",
        use_container_width=True,
        disabled=current_image is None,
    )

    if predict_clicked and current_image is not None and current_hash is not None:
        engine_used = "FastAPI Backend" if backend_online else "Local Model"
        try:
            with st.spinner(f"Identifying polymer using {selected_model_name} ({engine_used})..."):
                if backend_online:
                    try:
                        predictions = predict_via_fastapi(current_image, selected_model["key"])
                    except Exception as api_err:
                        # Fallback to local inference if backend fails during call
                        checkpoint_path = MODEL_DIR / selected_model["file"]
                        model, classes, device, preprocessing = load_model(
                            selected_model["key"], str(checkpoint_path), checkpoint_path.stat().st_mtime if checkpoint_path.exists() else 0.0
                        )
                        predictions = predict_local(current_image, model, classes, device, preprocessing)
                        engine_used = "Local Model (Fallback)"
                else:
                    checkpoint_path = MODEL_DIR / selected_model["file"]
                    model, classes, device, preprocessing = load_model(
                        selected_model["key"], str(checkpoint_path), checkpoint_path.stat().st_mtime if checkpoint_path.exists() else 0.0
                    )
                    predictions = predict_local(current_image, model, classes, device, preprocessing)

            st.session_state["polymercode_prediction"] = {
                "key": f"{current_hash}:{selected_model['key']}",
                "model_name": selected_model_name,
                "engine_used": engine_used,
                "predictions": predictions,
            }
            st.session_state.pop("polymercode_error", None)
        except (FileNotFoundError, RuntimeError, ValueError, KeyError, OSError) as exc:
            st.session_state.pop("polymercode_prediction", None)
            st.session_state["polymercode_error"] = str(exc)

    if st.session_state.get("polymercode_error"):
        st.error(st.session_state["polymercode_error"])

with right:
    st.markdown('<div class="pc-section">Prediction</div>', unsafe_allow_html=True)
    saved = st.session_state.get("polymercode_prediction")
    expected_key = f"{current_hash}:{selected_model['key']}" if current_hash else None

    if saved and expected_key and saved.get("key") == expected_key:
        top = saved["predictions"][0]
        label = top["label"]
        info = RESIN_GUIDE[label]
        confidence = top["probability"]

        st.markdown(
            f'<div class="pc-panel"><div class="pc-small">Predicted polymer · via {saved.get("engine_used", "AI Engine")}</div><div class="pc-prediction"><div class="pc-code">{info["code"]}</div><div><p class="pc-polymer">{label}</p><p class="pc-full">{info["full_name"]}</p></div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="pc-confidence-head"><span>Confidence score</span><span class="pc-confidence-value">{confidence:.1%}</span></div>',
            unsafe_allow_html=True,
        )
        st.progress(max(0.0, min(1.0, confidence)))
        if confidence < 0.55 or (len(saved["predictions"]) > 1 and confidence - saved["predictions"][1]["probability"] < 0.12):
            st.warning("Low-confidence prediction. Use a clear close-up of the printed resin symbol and verify the result.")

        with st.expander("Detailed Class Probability Distribution", expanded=False):
            for item in saved["predictions"]:
                c_lbl = item["label"]
                c_prob = item["probability"]
                st.markdown(
                    f'<div class="pc-prob-row"><span class="pc-prob-name">{c_lbl} · {RESIN_GUIDE[c_lbl]["full_name"]}</span><span class="pc-prob-pct">{c_prob:.1%}</span></div>',
                    unsafe_allow_html=True,
                )
                st.progress(max(0.0, min(1.0, c_prob)))

        st.markdown('<div class="pc-panel"><div class="pc-section">Recycling procedure</div>', unsafe_allow_html=True)
        for index, step in enumerate(info["steps"], start=1):
            st.markdown(
                f'<div class="pc-step"><span class="pc-stepnum">{index}</span><span>{step}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown(f'<div class="pc-note">{info["note"]}<br><br>Check local recycling rules.</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="pc-panel"><div class="pc-small">Predicted polymer</div><div class="pc-prediction"><div class="pc-code">—</div><div><p class="pc-polymer">Waiting for image</p><p class="pc-full">The result appears here.</p></div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="pc-panel"><div class="pc-section">Recycling procedure</div><div class="pc-small">Shown after polymer identification.</div></div>',
            unsafe_allow_html=True,
        )

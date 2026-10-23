"""FastAPI Backend for PolymerCode AI.

Provides high-performance resin identification inference endpoints integrating:
1. VGG-16 (Recommended baseline: 61.7% test accuracy)
2. VGG-16 Balanced (Experimental class-balanced weights)
3. Custom CNN (Lightweight baseline: 35.8% test accuracy)
"""

from __future__ import annotations

import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path
import os
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision import transforms

# Set optimal intra-op threads for CPU execution
if hasattr(os, "cpu_count") and os.cpu_count():
    torch.set_num_threads(os.cpu_count())


# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model_architectures import CustomCNN, build_vgg16

MODEL_DIR = ROOT_DIR / "models"
CLASS_NAMES = ["HDPE", "LDPE", "OTHER", "PET", "PP", "PS", "PVC"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "vgg16": {
        "key": "vgg16",
        "name": "VGG-16",
        "file": "vgg16_best.pt",
        "arch": "vgg16",
        "description": "Standard transfer learning model. Highest tested accuracy (65.43%, macro-F1 0.6425).",
        "recommended": True,
    },
    "vgg16_balanced": {
        "key": "vgg16_balanced",
        "name": "VGG-16 Balanced",
        "file": "vgg16_balanced_best.pt",
        "arch": "vgg16",
        "description": "Trained with balanced class loss weighting (validation macro-F1 0.4413).",
        "recommended": False,
    },
    "custom_cnn": {
        "key": "custom_cnn",
        "name": "Custom CNN",
        "file": "custom_cnn_best.pt",
        "arch": "custom_cnn",
        "description": "Lightweight custom 4-block convolutional network (44.44% test accuracy, macro-F1 0.4139).",
        "recommended": False,
    },
}

RESIN_INFO: Dict[str, Dict[str, Any]] = {
    "PET": {
        "code": 1,
        "name": "PET",
        "full_name": "Polyethylene terephthalate",
        "steps": [
            "Empty the container completely and separate from non-PET caps and labels where required.",
            "Rinse residue with water; adhere to local municipal sorting rules.",
            "The recycling facility sorts, shreds, washes, and extrudes PET into reusable pellets/flakes.",
            "Send only accepted PET items (e.g. standard beverage bottles) to recycling or deposit streams.",
        ],
        "note": "PET bottles have widespread recycling programs, but acceptance varies for thermoform trays and colored items.",
    },
    "HDPE": {
        "code": 2,
        "name": "HDPE",
        "full_name": "High-density polyethylene",
        "steps": [
            "Empty containers completely and separate from different resin types.",
            "Rinse residue thoroughly; handle caps and pumps according to local rules.",
            "Recycling facilities sort, shred, clean, and melt HDPE for durable goods and piping.",
            "Place in designated rigid plastics recycling stream.",
        ],
        "note": "HDPE milk jugs and detergent bottles are widely recyclable. Always check colored vs natural guidance.",
    },
    "PVC": {
        "code": 3,
        "name": "PVC",
        "full_name": "Polyvinyl chloride",
        "steps": [
            "Keep PVC strictly separate from standard PET, HDPE, and curbside collection bins.",
            "Identify the product type (e.g. pipes, siding) and check for specialist construction take-back centers.",
            "Deliver only to facilities licensed to handle vinyl products.",
            "Never burn, heat, or melt PVC at home as it releases hazardous chlorine compounds.",
        ],
        "note": "PVC is generally NOT accepted in household curbside bins. Dedicated industrial take-back is required.",
    },
    "LDPE": {
        "code": 4,
        "name": "LDPE",
        "full_name": "Low-density polyethylene",
        "steps": [
            "Separate flexible LDPE films, wraps, and grocery bags from rigid plastic bins.",
            "Ensure films are clean, dry, and free of food or adhesives.",
            "Take to designated supermarket film drop-off points rather than curbside carts.",
            "If no film collection service is available, follow local residual waste instructions.",
        ],
        "note": "Thin plastic films jam automated sorting machinery and should rarely go in standard curbside bins.",
    },
    "PP": {
        "code": 5,
        "name": "PP",
        "full_name": "Polypropylene",
        "steps": [
            "Empty and rinse PP containers (yogurt tubs, bottle caps, takeout boxes).",
            "Remove sealing foils, food residue, and non-PP labels.",
            "Sort into accepted plastic recycling streams where local infrastructure supports PP.",
            "Check local curbside guidelines as PP acceptance varies by municipality.",
        ],
        "note": "PP acceptance has expanded in recent years, but check whether black or rigid forms are accepted locally.",
    },
    "PS": {
        "code": 6,
        "name": "PS",
        "full_name": "Polystyrene",
        "steps": [
            "Distinguish between rigid polystyrene (cutlery, CD cases) and expanded foam (Styrofoam).",
            "Clean off all food debris and grease.",
            "Drop off foam packaging at specialist recycling depots if available in your area.",
            "Do not place polystyrene foam in curbside recycling unless explicitly instructed.",
        ],
        "note": "Polystyrene foam is lightweight and difficult to recycle economically. Never incinerate open-air.",
    },
    "OTHER": {
        "code": 7,
        "name": "OTHER",
        "full_name": "Other or mixed resin category",
        "steps": [
            "Inspect product markings or technical datasheets for multilayer or specialty polymer compositions (e.g. polycarbonate, acrylic, PLA).",
            "Do not mix with single-stream PET or HDPE bins.",
            "Contact the manufacturer or local recycling center for specialized disposal options.",
            "If unidentifiable or non-recyclable, dispose through designated general waste.",
        ],
        "note": "Code 7 encompasses multilayer plastics, bioplastics, and specialty resins; recyclability is non-standard.",
    },
}


class ModelManager:
    """Manages loading, caching, and inference for the 3 resin classification models."""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded_models: Dict[str, nn.Module] = {}
        self.preprocessing = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def normalize_model_key(self, raw_name: Optional[str]) -> str:
        if not raw_name:
            return "vgg16"
        clean = raw_name.lower().strip().replace("-", "").replace(" ", "_")
        if "custom" in clean or "cnn" in clean:
            return "custom_cnn"
        if "balanced" in clean:
            return "vgg16_balanced"
        if "vgg" in clean:
            return "vgg16"
        if clean in MODEL_REGISTRY:
            return clean
        return "vgg16"

    def get_model(self, model_key: str) -> nn.Module:
        key = self.normalize_model_key(model_key)
        if key in self._loaded_models:
            return self._loaded_models[key]

        info = MODEL_REGISTRY.get(key)
        if not info:
            raise ValueError(f"Unknown model key '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}")

        checkpoint_path = self.models_dir / info["file"]
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path.name} in {self.models_dir}")

        arch = info["arch"]
        if arch == "custom_cnn":
            model: nn.Module = CustomCNN(num_classes=7)
        elif arch == "vgg16":
            model = build_vgg16(num_classes=7)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

        # Strip 'module.' prefix if saved with DataParallel
        if isinstance(state_dict, dict) and all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.to(self.device)
        if self.device.type == "cpu":
            model.to(memory_format=torch.channels_last)
        model.eval()

        # Pre-warm model to eliminate cold-start latency
        with torch.inference_mode():
            dummy = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=self.device)
            if self.device.type == "cpu":
                dummy = dummy.to(memory_format=torch.channels_last)
            _ = model(dummy)

        self._loaded_models[key] = model
        return model

    def load_all_models(self) -> Dict[str, bool]:
        """Preload all 3 models into memory."""
        status_map = {}
        for key in MODEL_REGISTRY.keys():
            try:
                self.get_model(key)
                status_map[key] = True
            except Exception as e:
                print(f"[WARN] Failed to load model {key}: {e}", file=sys.stderr)
                status_map[key] = False
        return status_map

    def predict(self, image: Image.Image, model_name: str) -> Dict[str, Any]:
        normalized_key = self.normalize_model_key(model_name)
        model = self.get_model(normalized_key)

        # Preprocess image
        tensor = self.preprocessing(image.convert("RGB")).unsqueeze(0).to(self.device)
        if self.device.type == "cpu":
            tensor = tensor.to(memory_format=torch.channels_last)

        with torch.inference_mode():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

        # Class probabilities mapping
        probabilities = {cls: float(prob) for cls, prob in zip(CLASS_NAMES, probs)}

        # Ranking
        ranked_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        top_idx = ranked_indices[0]
        top_class = CLASS_NAMES[top_idx]
        top_confidence = float(probs[top_idx])

        ranked_predictions = [
            {
                "class": CLASS_NAMES[i],
                "score": float(probs[i]),
                "code": RESIN_INFO[CLASS_NAMES[i]]["code"],
                "full_name": RESIN_INFO[CLASS_NAMES[i]]["full_name"],
            }
            for i in ranked_indices
        ]

        info = RESIN_INFO[top_class]

        return {
            "class_name": top_class,
            "predicted_class": top_class,
            "class_index": top_idx,
            "confidence": top_confidence,
            "score": top_confidence,
            "probabilities": probabilities,
            "ranked_predictions": ranked_predictions,
            "resin_info": info,
            "model_used": normalized_key,
            "model_label": MODEL_REGISTRY[normalized_key]["name"],
            "device": str(self.device),
        }


# Global manager instance
manager = ModelManager(MODEL_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: preload models
    print(f"Loading PolymerCode AI models from: {MODEL_DIR}")
    status = manager.load_all_models()
    print(f"Model initialization complete: {status}")
    yield


app = FastAPI(
    title="PolymerCode AI API",
    description="High-accuracy plastic resin-code classification and recycling guidelines powered by PyTorch & FastAPI.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for all clients (Streamlit frontend, web UI, mobile)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="API Root")
def get_root():
    return {
        "service": "PolymerCode AI API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/api/v1/health",
            "models": "/api/v1/models",
            "resins": "/api/v1/resins",
            "predict": "/api/v1/predict (POST)",
        },
    }


@app.get("/api/v1/health", summary="Health Check")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": len(manager._loaded_models) > 0,
        "loaded_models": list(manager._loaded_models.keys()),
        "available_models": list(MODEL_REGISTRY.keys()),
        "device": str(manager.device),
        "cuda_available": torch.cuda.is_available(),
    }


@app.get("/api/v1/models", summary="List Available Models")
def list_models():
    models_data = []
    for key, item in MODEL_REGISTRY.items():
        models_data.append({
            "key": key,
            "name": item["name"],
            "architecture": item["arch"],
            "file": item["file"],
            "description": item["description"],
            "recommended": item["recommended"],
            "loaded": key in manager._loaded_models,
        })
    return {"models": models_data, "count": len(models_data)}


@app.get("/api/v1/resins", summary="List Resin Codes & Guidelines")
def list_resins():
    return {"resins": RESIN_INFO, "total_classes": len(RESIN_INFO)}


@app.post("/api/v1/predict", summary="Predict Polymer from Image")
async def predict_polymer(
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    model: str = Form("vgg16"),
):
    upload_file = image or file
    if upload_file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image uploaded. Please provide an image file under 'image' or 'file' key.",
        )

    content_type = upload_file.content_type or ""
    if not (content_type.startswith("image/") or upload_file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload a PNG, JPG, JPEG, or WEBP image.",
        )

    try:
        image_bytes = await upload_file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image payload received.",
            )

        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to decode image: {str(exc)}",
        )

    try:
        result = manager.predict(pil_image, model_name=model)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model asset missing: {str(exc)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)

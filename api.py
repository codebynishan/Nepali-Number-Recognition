import io
from typing import Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Query
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms, models

# ======================================================
# CONFIG
# ======================================================
MODEL_PATH = "../models/nepali_mobilenetv2.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
CONF_THRESHOLD = 0.7
API_KEY = "MY_API_KEY"

# ======================================================
# LOAD MODEL
# ======================================================
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
class_names = checkpoint["class_names"]
num_classes = len(class_names)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# ======================================================
# IMAGE TRANSFORM
# ======================================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================================================
# FASTAPI APP
# ======================================================
app = FastAPI(title="Nepali Digit Classification API")

# ======================================================
# CORE INFERENCE
# ======================================================
def _preprocess(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    return image

def model_predict(image_bytes: bytes):
    """
    Standard classifier prediction (no agent)
    """
    image = _preprocess(image_bytes)
    with torch.no_grad():
        logits = model(image)
        pred_idx = torch.argmax(logits, dim=1).item()
    return {"class": class_names[pred_idx]}

def agent_predict(image_bytes: bytes):
    """
    Agent-based prediction:
    - Uses confidence threshold
    - Detects out-of-distribution samples
    """
    image = _preprocess(image_bytes)
    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    if confidence.item() < CONF_THRESHOLD:
        return {"class": "unknown class"}

    return {"class": class_names[pred_idx.item()]}

# ======================================================
# API ENDPOINT
# ======================================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None),
    use_agent: bool = Query(False, description="Enable agent-based prediction")
):
    # ---------------- SECURITY ----------------
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # ---------------- VALIDATION ----------------
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Invalid image format")

    image_bytes = await file.read()

    # ---------------- INFERENCE ----------------
    if use_agent:
        result = agent_predict(image_bytes)
    else:
        result = model_predict(image_bytes)

    return JSONResponse(content=result)

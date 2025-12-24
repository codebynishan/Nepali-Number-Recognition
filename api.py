import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import JSONResponse
from typing import Optional
import torch
from torchvision import transforms, models
from PIL import Image
import io


MODEL_PATH = "../models/nepali_mobilenetv2.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
CONF_THRESHOLD = 0.7  
API_KEY = "mysecretapikey"  


checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
class_names = checkpoint["class_names"]
num_classes = len(class_names)

model = models.mobilenet_v2(weights=None)  # Don't load pretrained here
model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


app = FastAPI(title="Nepali Digit Classification API")

def predict_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image)
        pred = torch.argmax(outputs, dim=1)
        pred_class = class_names[pred.item()]
        return {"class": pred_class}


# -----------------------------
# ENDPOINT
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), x_api_key: Optional[str] = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    image_bytes = await file.read()
    result = predict_image(image_bytes)
    return JSONResponse(content=result)

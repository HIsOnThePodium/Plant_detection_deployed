from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
import numpy as np

app = FastAPI()

# ===== LOAD MODEL =====
MODEL_PATH = "plant_disease_model_1_latest.pt"

model = torch.load(MODEL_PATH, map_location="cpu")
model.eval()

def preprocess_pil(img: Image.Image) -> torch.Tensor:
    """
    Preprocess CƠ BẢN để chạy được.
    Nếu lúc train bạn resize size khác, nói mình chỉnh lại.
    """
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))      # (3, H, W)
    x = torch.tensor(arr).unsqueeze(0)      # (1, 3, H, W)
    return x

@app.get("/")
def home():
    return {"status": "ok", "model": MODEL_PATH}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))

    x = preprocess_pil(img)

    with torch.no_grad():
        y = model(x)

    # classification phổ biến
    if isinstance(y, torch.Tensor) and y.ndim == 2:
        pred = int(torch.argmax(y, dim=1).item())
        scores = y.squeeze(0).tolist()
        return {
            "pred": pred,
            "scores": scores
        }

    return {"error": "Unexpected model output", "type": str(type(y))}

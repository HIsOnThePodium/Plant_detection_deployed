import os
import csv
from flask import Flask, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
import CNN

# ================== READ CSV (NO PANDAS) ==================
def read_csv_dicts(path):
    with open(path, newline="", encoding="ISO-8859-1") as f:
        return list(csv.DictReader(f))

disease_info = read_csv_dicts("disease_info.csv")
supplement_info = read_csv_dicts("supplement_info.csv")

# ================== LOAD MODEL ==================
model = CNN.CNN(39)
model.load_state_dict(
    torch.load("plant_disease_model_1_latest.pt", map_location="cpu")
)
model.eval()

# ================== PREDICTION ==================
def prediction(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_data)
        index = torch.argmax(output).item()

    return index

# ================== FLASK APP ==================
app = Flask(__name__)

# ================== ROUTES ==================
@app.route("/")
def home_page():
    return render_template("home.html")

@app.route("/contact")
def contact():
    return render_template("contact-us.html")

@app.route("/index")
def ai_engine_page():
    return render_template("index.html")

@app.route("/mobile-device")
def mobile_device_detected_page():
    return render_template("mobile-device.html")

@app.route("/submit", methods=["POST"])
def submit():
    image = request.files["image"]
    filename = image.filename

    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, filename)
    image.save(file_path)

    pred = prediction(file_path)

    title = disease_info[pred]["disease_name"]
    description = disease_info[pred]["description"]
    prevent = disease_info[pred]["Possible Steps"]
    image_url = disease_info[pred]["image_url"]

    supplement_name = supplement_info[pred]["supplement name"]
    supplement_image_url

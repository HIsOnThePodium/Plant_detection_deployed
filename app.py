import os
import csv
from flask import Flask, render_template, request
from PIL import Image
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
state = torch.load("plant_disease_model_1_latest.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()

# (optional) giảm RAM thêm chút
torch.set_num_threads(1)

# ================== PREDICTION ==================
def prediction(image_path):
    image = Image.open(image_path).convert("RGB").resize((224, 224))

    # PIL -> numpy float32 -> torch tensor (1,3,224,224)
    arr = np.array(image, dtype=np.float32) / 255.0   # H,W,C
    arr = np.transpose(arr, (2, 0, 1))                # C,H,W
    input_data = torch.from_numpy(arr).unsqueeze(0)   # 1,C,H,W

    with torch.no_grad():
        output = model(input_data)
        return int(torch.argmax(output).item())

# ================== FLASK APP ==================
app = Flask(__name__)

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
    supplement_image_url = supplement_info[pred]["supplement image"]
    supplement_buy_link = supplement_info[pred]["buy link"]

    return render_template(
        "submit.html",
        title=title,
        desc=description,
        prevent=prevent,
        image_url=image_url,
        pred=pred,
        sname=supplement_name,
        simage=supplement_image_url,
        buy_link=supplement_buy_link,
    )

@app.route("/market")
def market():
    return render_template(
        "market.html",
        supplement_image=[row["supplement image"] for row in supplement_info],
        supplement_name=[row["supplement name"] for row in supplement_info],
        disease=[row["disease_name"] for row in disease_info],
        buy=[row["buy link"] for row in supplement_info],
    )

if __name__ == "__main__":
    app.run(debug=True)

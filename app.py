import os
import csv
import requests
from flask import Flask, render_template, request

# ================== READ CSV (NO PANDAS) ==================
def read_csv_dicts(path):
    with open(path, newline="", encoding="ISO-8859-1") as f:
        return list(csv.DictReader(f))

disease_info = read_csv_dicts("disease_info.csv")
supplement_info = read_csv_dicts("supplement_info.csv")

# ================== HUGGING FACE API ==================
HF_PREDICT_URL = "https://dogiahien-model.hf.space/predict"

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
    # HTML input của bạn đang name="image"
    image = request.files["image"]
    filename = image.filename

    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, filename)
    image.save(file_path)

    # ===== gọi HuggingFace để predict =====
    image.stream.seek(0)  # reset con trỏ đọc file
    files = {"file": (filename, image.read(), image.mimetype or "image/jpeg")}
    resp = requests.post(HF_PREDICT_URL, files=files, timeout=60)
    data = resp.json()

    # HF trả: {"pred_idx":..., "pred_label":..., "confidence":...}
    pred = int(data["pred_idx"])

    # ===== phần dưới giữ nguyên logic của bạn =====
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

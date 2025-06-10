from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import cv2, uuid, os, base64
from pathlib import Path

# Carrega modelo uma única vez (yolov8n → rápido. Troque por yolov8s/l/x se quiser mais acurácia)
model = YOLO("yolov8n.pt")

# IDs COCO de veículos: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_IDS = [2, 3, 5, 7]

app = Flask(__name__)
UPLOAD_DIR = Path("static/uploads")
OUT_DIR    = Path("static/outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True,    exist_ok=True)

def allowed(name):  # extensões básicas
    return name.lower().endswith((".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    if not file or not allowed(file.filename):
        return "Arquivo não suportado", 400

    # Salva arquivo
    uid = f"{uuid.uuid4()}{Path(file.filename).suffix}"
    in_path  = UPLOAD_DIR / uid
    file.save(in_path)

    # Decide se é imagem ou vídeo
    if in_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
        count, out_name = process_image(in_path)
        return render_template("result.html",
                              is_video=False,
                              img_data=out_name,
                              count=count)

    else:
        count, out_path = process_video(in_path)
        return render_template("result.html",
                              is_video=True,
                              video_src=url_for('static', filename=f"outputs/{out_path.name}"),
                              count=count)

def process_image(path: Path):
    results = model.predict(source=str(path),
                            conf=0.3,
                            classes=VEHICLE_IDS,
                            verbose=False)[0]

    # contagem
    count = len(results.boxes)
    # converte imagem anotada para base64 p/ embutir no HTML
    annotated = results.plot()
    b64 = base64.b64encode(cv2.imencode(".jpg", annotated)[1]).decode()
    data_uri = f"data:image/jpeg;base64,{b64}"
    return count, data_uri

def process_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    w,h,fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                        cv2.CAP_PROP_FRAME_HEIGHT,
                                        cv2.CAP_PROP_FPS))
    out_name = f"{uuid.uuid4()}.mp4"
    out_path = OUT_DIR / out_name
    writer = cv2.VideoWriter(str(out_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps, (w,h))
    seen_ids = set()

    # usa modo track do YOLO para ID persistente
    for frame, result in model.track(source=str(path),
                                    classes=VEHICLE_IDS,
                                    conf=0.3,
                                    persist=True,
                                    stream=True):   # stream=True → generator
        # result contains .boxes.id quando persist=True
        if result.boxes.id is not None:
            seen_ids.update(result.boxes.id.cpu().numpy().astype(int).tolist())

        writer.write(result.plot())
    writer.release()
    cap.release()
    return len(seen_ids), out_path

if __name__ == "__main__":
    # Escuta em todas as interfaces de rede, porta 5000
    app.run(host="0.0.0.0", port=5000, debug=True)

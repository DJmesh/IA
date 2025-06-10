"""
YOLO + Flask – detecção de veículos (car, motorcycle, bus, truck).

Requer: ultralytics >= 8.1, opencv-python-headless >= 4.8,
        Flask >= 2.3, Pillow.
"""

from flask import Flask, render_template, request, jsonify, url_for
from ultralytics import YOLO
import cv2
import uuid
import json
from pathlib import Path
import base64
import os

# ──────────────────── CONFIGURAÇÃO ────────────────────
MODEL_WEIGHTS = "yolov8n.pt"          # troque p/ yolov8s|m|l se quiser +acurácia
VEHICLE_IDS    = [2, 3, 5, 7]         # car, motorcycle, bus, truck (COCO)
VEHICLE_NAMES  = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

ALLOWED_IMG   = (".jpg", ".jpeg", ".png")
ALLOWED_VIDEO = (".mp4", ".mov", ".avi", ".mkv", ".webm")

BASE_DIR   = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUT_DIR    = BASE_DIR / "static" / "outputs"
for p in (UPLOAD_DIR, OUT_DIR):
    p.mkdir(parents=True, exist_ok=True)

app   = Flask(__name__)
model = YOLO(MODEL_WEIGHTS)   # carrega 1x

# ──────────────────── UTILIDADES ────────────────────
def _is_allowed(name: str) -> bool:
    return name.lower().endswith(ALLOWED_IMG + ALLOWED_VIDEO)

def _save_json(data: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _cls_counter(cls_list):
    """Recebe lista numpy de ids de classe e devolve dict {id:count}."""
    counts = {}
    for cid in cls_list:
        cid = int(cid)
        counts[cid] = counts.get(cid, 0) + 1
    return counts

# ──────────────────── PROCESSADORES ────────────────────
def process_image(src_path: Path, out_dir: Path) -> dict:
    res = model.predict(
        source=str(src_path),
        classes=VEHICLE_IDS,
        conf=0.3,
        verbose=False
    )[0]

    counts_raw = _cls_counter(res.boxes.cls.cpu().numpy())
    counts = {VEHICLE_NAMES[k]: v for k, v in counts_raw.items()}
    total  = sum(counts.values())

    annotated = res.plot()  # ndarray BGR
    out_path  = out_dir / "annotated.jpg"
    cv2.imwrite(str(out_path), annotated)

    # data-uri para exibir inline no HTML
    b64 = base64.b64encode(cv2.imencode(".jpg", annotated)[1]).decode()
    img_data_uri = f"data:image/jpeg;base64,{b64}"

    meta = {
        "type": "image",
        "original": src_path.name,
        "annotated": out_path.name,
        "counts": counts,
        "total": total
    }
    return meta, img_data_uri


def process_video(src_path: Path, out_dir: Path) -> dict:
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError("Falha ao abrir vídeo para leitura.")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    out_path = out_dir / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    seen = {cid: set() for cid in VEHICLE_IDS}

    # Ultralytics tracking generator  :contentReference[oaicite:0]{index=0}
    for result in model.track(
            source=str(src_path),
            stream=True,
            classes=VEHICLE_IDS,
            conf=0.3,
            persist=True):
        # result é ultralytics.engine.results.Results
        annotated = result.plot()
        writer.write(annotated)

        if result.boxes.id is not None:  # ids disponíveis
            for cls_id, track_id in zip(
                    result.boxes.cls.cpu().numpy(),
                    result.boxes.id.cpu().numpy()):
                cid, tid = int(cls_id), int(track_id)
                if cid in seen:
                    seen[cid].add(tid)

    cap.release()
    writer.release()

    counts = {VEHICLE_NAMES[cid]: len(ids) for cid, ids in seen.items()}
    total  = sum(counts.values())

    meta = {
        "type": "video",
        "original": src_path.name,
        "annotated": out_path.name,
        "fps": fps,
        "resolution": [w, h],
        "counts": counts,
        "total": total
    }
    return meta


# ──────────────────── ROTAS ────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")   # deixe seu template + bonito aqui


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file or not _is_allowed(file.filename):
        return "Arquivo não suportado", 400

    # cria pastas exclusivas para este upload
    uid        = str(uuid.uuid4())
    upload_dir = UPLOAD_DIR / uid
    out_dir    = OUT_DIR / uid
    upload_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # salva arquivo original
    in_path = upload_dir / file.filename
    file.save(in_path)

    try:
        if in_path.suffix.lower() in ALLOWED_IMG:
            meta, img_data_uri = process_image(in_path, out_dir)
            meta_path = out_dir / "metadata.json"
            _save_json(meta, meta_path)

            # JSON direto?
            if request.form.get("mode") == "json" or request.args.get("json"):
                meta["annotated_url"] = url_for(
                    'static',
                    filename=f"outputs/{uid}/{meta['annotated']}",
                    _external=True)
                return jsonify(meta)

            # HTML
            return render_template(
                "result.html",
                is_video=False,
                img_data=img_data_uri,
                counts=meta["counts"],
                total=meta["total"]
            )

        else:  # vídeo
            meta = process_video(in_path, out_dir)
            meta_path = out_dir / "metadata.json"
            _save_json(meta, meta_path)

            if request.form.get("mode") == "json" or request.args.get("json"):
                meta["annotated_url"] = url_for(
                    'static',
                    filename=f"outputs/{uid}/{meta['annotated']}",
                    _external=True)
                return jsonify(meta)

            video_url = url_for(
                'static',
                filename=f"outputs/{uid}/{meta['annotated']}"
            )
            return render_template(
                "result.html",
                is_video=True,
                video_src=video_url,
                counts=meta["counts"],
                total=meta["total"]
            )

    except Exception as e:
        # Se algo falhar, limpe a pasta p/ não deixar lixo
        try:
            for p in (upload_dir, out_dir):
                if p.exists():
                    for f in p.iterdir():
                        f.unlink()
                    p.rmdir()
        finally:
            raise e  # Flask mostrará no debug

# health-check simples para orquestrações Docker/K8s
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


# ──────────────────── ENTRADA PRINCIPAL ────────────────────
if __name__ == "__main__":
    # Todas as interfaces, porta 5000
    app.run(host="0.0.0.0", port=5000, debug=True)

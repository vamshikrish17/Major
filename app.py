import os
import uuid
import random
import base64

import cv2
import numpy as np
import torch
from flask import Flask, request, render_template, url_for
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_MODEL_PATH = "yolov8n.pt"          # YOLOv8 nano, auto-downloads
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth" # <-- make sure this file exists

app = Flask(__name__)
# Increase max upload size and form data memory size to 100 Megabytes
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['MAX_FORM_MEMORY_SIZE'] = 100 * 1024 * 1024

print(f"Using device: {DEVICE}")
print("Loading YOLOv8 model...")
yolo_model = YOLO(YOLO_MODEL_PATH)

print("Loading SAM model...")
sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

try:
    CLASS_NAMES = yolo_model.model.names
except AttributeError:
    CLASS_NAMES = yolo_model.names


# ---------------- CORE LOGIC ----------------
def detect_and_segment(image_bgr, conf_thr=0.35):
    """
    Runs YOLOv8 detection + SAM segmentation concurrently on a BGR image.
    Uses Batched Tensor Processing and Semantic-Aware Prompts (Boxes + Center-Points)
    to radically optimize performance and fix spatial ambiguity from the existing literature.
    """
    import torch
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 1. YOLO detection (Returns PyTorch tensors directly on DEVICE)
    results = yolo_model(image_rgb)[0]
    boxes = results.boxes.xyxy
    scores = results.boxes.conf
    class_ids = results.boxes.cls

    # Filter tensors heavily by confidence threshold
    valid_idx = scores >= conf_thr
    boxes = boxes[valid_idx]
    scores = scores[valid_idx]
    class_ids = class_ids[valid_idx]

    out = image_bgr.astype(np.float32)
    segments = []

    if len(boxes) == 0:
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out, segments

    # 2. Prepare SAM Image Block
    sam_predictor.set_image(image_rgb)

    # 3. Concurrent Multi-Object Batch Processing (Inspired by Kim et al. 2025)
    batch_boxes = sam_predictor.transform.apply_boxes_torch(boxes, image_rgb.shape[:2]).to(DEVICE)

    # 4. Semantic-Aware Multi-Prompting (Inspired by SaSAM / Wei et al. 2024)
    # Formulate precise Hybrid Prompts: Bounding Box anchored exactly with the Object Center-of-Mass.
    points = torch.zeros((len(boxes), 1, 2), device=DEVICE)
    points[:, 0, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # Center X
    points[:, 0, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # Center Y

    batch_points = sam_predictor.transform.apply_coords_torch(points, image_rgb.shape[:2]).to(DEVICE)
    batch_labels = torch.ones((len(boxes), 1), dtype=torch.int, device=DEVICE)

    # 5. Execute batched prediction entirely in parallel (Zero looping across SAM instances)
    masks, iou_preds, _ = sam_predictor.predict_torch(
        point_coords=batch_points,
        point_labels=batch_labels,
        boxes=batch_boxes,
        multimask_output=False
    )
    
    # 6. Offload output batch back to system memory for pixel operations
    masks_torch = masks[:, 0, :, :] # Extract (N, H, W) bool tensor array
    masks_np = masks_torch.cpu().numpy()
    boxes_np = boxes.int().cpu().numpy()
    scores_np = scores.cpu().numpy()
    class_ids_np = class_ids.cpu().numpy()

    h, w = image_bgr.shape[:2]

    # Re-iterate securely on CPU output matrices solely for OpenCV Dashboard rendering
    for i in range(len(masks_np)):
        mask = masks_np[i]

        if mask.sum() < 10:
            continue

        x1, y1, x2, y2 = boxes_np[i]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        # Store robust mask + meta for isolated PNG export framework
        cls_id = int(class_ids_np[i])
        score = float(scores_np[i])
        label = CLASS_NAMES.get(cls_id, str(cls_id))

        segments.append({
            "mask": mask,  # Complete dense-pixel image mask
            "box": [x1, y1, x2, y2],
            "label": label,
            "score": score,
        })

        # 7. Front-end Visualization (Colored semantic overlay generation)
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        alpha = 0.6

        mask_bool = mask.astype(bool)
        out[mask_bool] = (
            (1 - alpha) * out[mask_bool] +
            alpha * np.array(color, dtype=np.float32)
        )

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{label} {score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, segments


def save_segments_as_pngs(original_bgr, segments):
    """
    For each segment:
      - create transparent PNG (object only, no background, no overlay)
      - cropped to bounding box
    Returns list of dicts: {url, label, score}
    """
    h, w = original_bgr.shape[:2]
    segment_urls = []

    for i, seg in enumerate(segments):
        mask = seg["mask"].astype(bool)
        x1, y1, x2, y2 = seg["box"]

        # Build 4-channel BGRA full-size image
        seg_rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Copy original BGR where mask is True
        seg_rgba[mask, 0:3] = original_bgr[mask]
        seg_rgba[mask, 3] = 255  # opaque where object

        # Crop to bounding box to keep only area around object
        seg_cropped = seg_rgba[y1:y2+1, x1:x2+1]

        # Save to disk
        seg_name = f"{uuid.uuid4().hex}_seg{i}.png"
        seg_path = os.path.join(RESULT_FOLDER, seg_name)
        cv2.imwrite(seg_path, seg_cropped)

        segment_urls.append({
            "url": url_for("static", filename=f"results/{seg_name}"),
            "label": seg["label"],
            "score": seg["score"],
        })

    return segment_urls


# ---------------- HTML TEMPLATE ----------------
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>YOLO + SAM Web GUI</title>
    <style>
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            margin: 0;
            padding: 0;
            background: #0f172a;
            color: #e5e7eb;
        }
        .container {
            max-width: 1100px;
            margin: 30px auto;
            padding: 20px;
            background: #020617;
            border-radius: 16px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
        }
        h1 {
            text-align: center;
            margin-bottom: 10px;
        }
        p.subtitle {
            text-align: center;
            color: #9ca3af;
            margin-top: 0;
            margin-bottom: 20px;
        }
        form {
            border: 1px dashed #4b5563;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .form-row {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .form-column {
            flex: 1 1 300px;
        }
        input[type=file] {
            margin: 10px 0;
        }
        button, .btn {
            padding: 8px 16px;
            background: #22c55e;
            border: none;
            color: #020617;
            border-radius: 999px;
            font-weight: 600;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover, .btn:hover {
            background: #16a34a;
        }
        .btn-secondary {
            background: #1d4ed8;
            color: #e5e7eb;
        }
        .btn-secondary:hover {
            background: #1e40af;
        }
        .btn-danger {
            background: #ef4444;
            color: #fee2e2;
        }
        .btn-danger:hover {
            background: #b91c1c;
        }
        .note {
            font-size: 12px;
            color: #9ca3af;
            margin-top: 6px;
        }
        .images {
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            justify-content: center;
            margin-top: 20px;
        }
        .card {
            background: #020617;
            border-radius: 12px;
            padding: 12px;
            border: 1px solid #1f2937;
            flex: 1 1 300px;
            max-width: 450px;
        }
        .card h2, .card h3 {
            font-size: 18px;
            margin-top: 0;
        }
        img {
            max-width: 100%;
            border-radius: 8px;
        }
        .footer {
            text-align: center;
            color: #6b7280;
            font-size: 12px;
            margin-top: 10px;
        }
        a.download-link {
            display: inline-block;
            margin-top: 6px;
            font-size: 13px;
            color: #60a5fa;
        }
        .segments-section {
            margin-top: 24px;
        }
        .segment-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            justify-content: center;
        }
        .webcam-container {
            border-radius: 12px;
            border: 1px solid #1f2937;
            padding: 12px;
            background: #020617;
        }
        video {
            width: 100%;
            max-height: 260px;
            border-radius: 8px;
            background: #000;
        }
        .webcam-buttons {
            display: flex;
            gap: 10px;
            margin-top: 10px;
            flex-wrap: wrap;
        }
        .top-actions {
            display: flex;
            justify-content: flex-end;
            margin-top: 10px;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>YOLOv8 + SAM Image Segmentation</h1>
    <p class="subtitle">Upload an image or use webcam → detect objects → segment with SAM → download clean cutouts.</p>

    <form id="mainForm" method="post" enctype="multipart/form-data">
        <div class="form-row">
            <div class="form-column">
                <h3>Upload Image</h3>
                <label>Select an image file (JPG/PNG):</label><br>
                <input type="file" name="image" accept="image/*"><br><br>
                <button type="submit">Run on Uploaded Image</button>
                <p class="note">If both file and webcam are used, webcam capture will take priority.</p>
            </div>

            <div class="form-column">
                <div class="webcam-container">
                    <h3>Webcam</h3>
                    <video id="webcam" autoplay playsinline></video>
                    <canvas id="webcamCanvas" style="display:none;"></canvas>
                    <div class="webcam-buttons">
                        <button type="button" class="btn btn-secondary" onclick="startWebcam()">Start Webcam</button>
                        <button type="button" class="btn" onclick="captureWebcam()">Capture &amp; Segment</button>
                    </div>
                    <p class="note">Allow camera permission in your browser.</p>
                </div>
            </div>
        </div>

        <!-- hidden field to send webcam image (base64) -->
        <input type="hidden" name="webcam_data" id="webcamData">
    </form>

    {% if result_url %}
    <div class="top-actions">
        <button type="button" class="btn btn-danger" onclick="clearResults()">✕ Clear recent segmentation</button>
    </div>

    <div class="images">
        {% if original_url %}
        <div class="card">
            <h2>Original</h2>
            <img src="{{ original_url }}" alt="Original image">
        </div>
        {% endif %}
        <div class="card">
            <h2>Segmented Preview</h2>
            <img src="{{ result_url }}" alt="Result image">
            <br>
            <a class="download-link" href="{{ result_url }}" download>⬇ Download preview image</a>
        </div>
    </div>
    {% endif %}

    {% if segment_urls %}
    <div class="segments-section">
        <h2>Individual Segments (no background)</h2>
        <p class="note">Each PNG has transparent background and original colors (no overlay), similar to remove.bg.</p>
        <div class="segment-grid">
            {% for seg in segment_urls %}
            <div class="card">
                <h3>{{ seg.label }} ({{ "%.2f"|format(seg.score) }})</h3>
                <img src="{{ seg.url }}" alt="Segment image">
                <br>
                <a class="download-link" href="{{ seg.url }}" download>⬇ Download segment PNG</a>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}

    <div class="footer">
        YOLOv8 for detection + SAM for segmentation | Runs locally on your laptop.
    </div>
</div>

<script>
let webcamStream = null;

function startWebcam() {
    if (webcamStream) return;
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            webcamStream = stream;
            const video = document.getElementById('webcam');
            video.srcObject = stream;
        })
        .catch(function(err) {
            alert("Error accessing webcam: " + err);
        });
}

function captureWebcam() {
    const video = document.getElementById('webcam');
    if (!video || video.readyState !== 4) {
        alert("Webcam not ready. Click 'Start Webcam' and wait for preview.");
        return;
    }
    const canvas = document.getElementById('webcamCanvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/png');
    document.getElementById('webcamData').value = dataURL;
    document.getElementById('mainForm').submit();
}

function clearResults() {
    // Simple way: reload page with GET to clear recent result display
    window.location.href = "/";
}
</script>

</body>
</html>
"""


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    original_url = None
    result_url = None
    segment_urls = None

    if request.method == "POST":
        img = None
        upload_name = None

        # 1) Check if webcam data is provided
        webcam_data = request.form.get("webcam_data", "").strip()
        if webcam_data:
            # Expecting format: "data:image/png;base64,...."
            try:
                header, encoded = webcam_data.split(",", 1)
                img_bytes = base64.b64decode(encoded)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                ext = ".png"
                upload_name = f"{uuid.uuid4().hex}{ext}"
                upload_path = os.path.join(UPLOAD_FOLDER, upload_name)
                cv2.imwrite(upload_path, img)
            except Exception as e:
                print("Error decoding webcam image:", e)
                img = None

        # 2) If no webcam image, try file upload
        if img is None:
            if "image" not in request.files:
                return render_template("index.html",
                                              original_url=None,
                                              result_url=None,
                                              segment_urls=None)

            file = request.files["image"]
            if file.filename == "":
                return render_template("index.html",
                                              original_url=None,
                                              result_url=None,
                                              segment_urls=None)

            ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
            upload_name = f"{uuid.uuid4().hex}{ext}"
            upload_path = os.path.join(UPLOAD_FOLDER, upload_name)
            file.save(upload_path)

            img = cv2.imread(upload_path)

        if img is None:
            return render_template("index.html",
                                          original_url=None,
                                          result_url=None,
                                          segment_urls=None)

        # Run YOLO + SAM
        result_img, segments = detect_and_segment(img, conf_thr=0.35)

        # Save preview result image
        result_name = f"{uuid.uuid4().hex}.png"
        result_path = os.path.join(RESULT_FOLDER, result_name)
        cv2.imwrite(result_path, result_img)

        # Save individual segments as transparent PNGs
        segment_urls = save_segments_as_pngs(img, segments) if segments else None

        original_url = url_for("static", filename=f"uploads/{upload_name}")
        result_url = url_for("static", filename=f"results/{result_name}")

    return render_template(
        "index.html",
        original_url=original_url,
        result_url=result_url,
        segment_urls=segment_urls,
    )


if __name__ == "__main__":
    # Access in browser at: http://127.0.0.1:1726
    app.run(host="0.0.0.0", port=1726, debug=True)
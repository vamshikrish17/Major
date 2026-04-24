import argparse
import os
import random
import cv2
import numpy as np
import torch

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor


# -------------------------------------
# Helper: Check if file is an image
# -------------------------------------
def is_image_file(filename):
    ext = filename.lower().split(".")[-1]
    return ext in ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]


# -------------------------------------
# Load YOLO + SAM models
# -------------------------------------
def load_models(yolo_model_path, sam_checkpoint, device):
    print("\nLoading YOLO model...")
    yolo = YOLO(yolo_model_path)

    print("Loading SAM model...")
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    print("Models loaded successfully.\n")
    return yolo, predictor


# -------------------------------------
# YOLO Detection + SAM Segmentation
# -------------------------------------
def detect_and_segment(image_bgr, yolo, predictor, device="cpu", conf_thr=0.35):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ---------- 1. YOLO Detection ----------
    results = yolo(image_rgb)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)

    try:
        class_names = yolo.model.names
    except:
        class_names = yolo.names

    predictor.set_image(image_rgb)

    out = image_bgr.astype(np.float32)

    for box, score, cls_id in zip(boxes, scores, class_ids):
        if score < conf_thr:
            continue

        x1, y1, x2, y2 = box.astype(int)

        # ---------- 2. SAM Segmentation ----------
        box_prompt = np.array([x1, y1, x2, y2])
        masks, iou_preds, _ = predictor.predict(
            box=box_prompt[None, :],
            multimask_output=False
        )
        mask = masks[0]

        if mask.sum() < 10:
            continue

        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

        alpha = 0.6
        mask_bool = mask.astype(bool)

        out[mask_bool] = (
            (1 - alpha) * out[mask_bool] +
            alpha * np.array(color, dtype=np.float32)
        )

        label = class_names[int(cls_id)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{label} {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return np.clip(out, 0, 255).astype(np.uint8)


# -------------------------------------
# Webcam Processing
# -------------------------------------
def run_webcam(yolo, predictor, device="cpu", conf_thr=0.35, cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("\n📷 Webcam started — Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = detect_and_segment(frame, yolo, predictor, device=device, conf_thr=conf_thr)
        cv2.imshow("YOLO + SAM (Press q to exit)", result_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------------
# MAIN FUNCTION
# -------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--folder", help="Folder containing multiple images")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--sam_ckpt", required=True, help="Path to SAM checkpoint (.pth)")
    parser.add_argument("--yolo_model", default="yolov8n.pt", help="YOLOv8 model name")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--output", default="output.png", help="Output file name")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")

    # Load models once
    yolo, predictor = load_models(args.yolo_model, args.sam_ckpt, device)

    # ----------- Webcam Mode -----------
    if args.webcam:
        run_webcam(yolo, predictor, device=device, conf_thr=args.conf)
        return

    # ----------- Folder Mode -----------
    if args.folder:
        folder = args.folder
        out_dir = os.path.join(folder, "segmented")
        os.makedirs(out_dir, exist_ok=True)

        files = [f for f in os.listdir(folder) if is_image_file(f)]

        print(f"📁 Found {len(files)} images in folder: {folder}\n")

        for f in files:
            img_path = os.path.join(folder, f)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Skipping unreadable file: {f}")
                continue

            result = detect_and_segment(img, yolo, predictor, device=device, conf_thr=args.conf)
            save_path = os.path.join(out_dir, f"seg_{f}")
            cv2.imwrite(save_path, result)

            print(f"✔ Saved: {save_path}")

        print("\n🎉 Folder processing complete!\n")
        return

    # ----------- Single Image Mode -----------
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")

        result = detect_and_segment(img, yolo, predictor, device=device, conf_thr=args.conf)
        cv2.imwrite(args.output, result)

        print(f"✔ Saved output to {args.output}")
        return

    print("❗ Please provide --image OR --folder OR --webcam")


# -------------------------------------
# Entry Point
# -------------------------------------
if __name__ == "__main__":
    main()
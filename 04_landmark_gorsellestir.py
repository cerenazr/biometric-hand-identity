"""
04_visualize_landmarks.py
--------------------------
Dataset'ten ornek goruntuleri alip MediaPipe landmark'larini uzerine cizer.
Her aspect (dorsal, palmar) ve her el (sol, sag) icin birer ornek uretir.
Cikti: output/plots/landmarks_*.png

Calistirma: python 04_visualize_landmarks.py
"""

import os
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data", "11k_hands")
PLOTS_DIR  = os.path.join(BASE_DIR, "output", "plots")
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

os.makedirs(PLOTS_DIR, exist_ok=True)

# MediaPipe el iskeleti baglantilari
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # Basparmak
    (0,5),(5,6),(6,7),(7,8),          # Isaret
    (5,9),(9,10),(10,11),(11,12),     # Orta
    (9,13),(13,14),(14,15),(15,16),   # Yuzsuz
    (13,17),(17,18),(18,19),(19,20),  # Serce
    (0,17),(5,9),(9,13),(13,17),      # Avuc ici
]

FINGERTIP_IDX = {4, 8, 12, 16, 20}
WRIST_IDX     = 0


def _find_image_root(base):
    for root, _, files in os.walk(base):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                return root
    return base


def draw_landmarks(img_bgr, landmarks):
    h, w = img_bgr.shape[:2]
    pts  = [(int(l.x * w), int(l.y * h)) for l in landmarks]
    out  = img_bgr.copy()

    # Iskelet baglantilari
    for a, b in CONNECTIONS:
        cv2.line(out, pts[a], pts[b], (0, 200, 0), 2, cv2.LINE_AA)

    # Landmark noktalari
    for i, (x, y) in enumerate(pts):
        if i in FINGERTIP_IDX:
            color, r = (0, 80, 255), 7    # Parmak ucu — kirmizi
        elif i == WRIST_IDX:
            color, r = (255, 140, 0), 7   # Bilek — turuncu
        else:
            color, r = (255, 255, 255), 5 # Diger — beyaz
        cv2.circle(out, (x, y), r, color, -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), r, (0, 0, 0), 1, cv2.LINE_AA)

    return out


def process_and_save(img_name, subj_id, aspect, detector, image_root):
    path = os.path.join(image_root, img_name)
    with open(path, "rb") as f:
        img_bgr = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return False

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result  = detector.detect(mp_img)
    if not result.hand_landmarks:
        return False

    annotated = draw_landmarks(img_bgr, result.hand_landmarks[0])
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Orijinal Goruntu", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(annotated_rgb)
    axes[1].set_title("MediaPipe Landmark Tespiti (21 nokta)", fontsize=12)
    axes[1].axis("off")

    fig.suptitle(
        f"Kisi {subj_id} — {aspect.title()} | {img_name}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    safe_aspect = aspect.replace(" ", "_")
    save_path = os.path.join(PLOTS_DIR, f"landmarks_{safe_aspect}_subject{subj_id}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {os.path.basename(save_path)}")
    return True


def main():
    csv_path = os.path.join(DATA_DIR, "HandInfo.csv")
    if not os.path.exists(csv_path):
        print("HATA: HandInfo.csv bulunamadi.")
        return
    if not os.path.exists(MODEL_PATH):
        print("HATA: hand_landmarker.task bulunamadi.")
        return

    info       = pd.read_csv(csv_path)
    image_root = _find_image_root(DATA_DIR)

    with open(MODEL_PATH, "rb") as f:
        model_bytes = f.read()
    detector = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3,
        )
    )

    # Her aspect ve el tarafı kombinasyonu için birer örnek seç
    targets = [
        ("dorsal right",  "El Sirtu — Sag"),
        ("dorsal left",   "El Sirtu — Sol"),
        ("palmar right",  "Avuc Ici — Sag"),
        ("palmar left",   "Avuc Ici — Sol"),
    ]

    print("Landmark gorsellestirmesi olusturuluyor...\n")
    for aspect_key, aspect_label in targets:
        subset = info[info["aspectOfHand"].str.lower() == aspect_key.lower()]
        if subset.empty:
            print(f"  '{aspect_key}' icin goruntu bulunamadi, atlaniyor.")
            continue

        # Her deneme için farklı bir kişi dene, MediaPipe tespiti başarana kadar
        success = False
        for _, row in subset.sample(frac=1, random_state=42).head(20).iterrows():
            ok = process_and_save(
                str(row["imageName"]).strip(),
                str(row["id"]),
                aspect_label,
                detector,
                image_root,
            )
            if ok:
                success = True
                break
        if not success:
            print(f"  '{aspect_label}' icin tespit basarisiz.")

    detector.close()

    # Grid ozet: 4 goruntu tek figurde
    landmark_files = sorted([
        f for f in os.listdir(PLOTS_DIR)
        if f.startswith("landmarks_") and f.endswith(".png")
    ])

    if len(landmark_files) >= 2:
        n    = len(landmark_files)
        cols = 2
        rows = (n + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
        axes = np.array(axes).flatten()

        for i, fname in enumerate(landmark_files):
            img = plt.imread(os.path.join(PLOTS_DIR, fname))
            axes[i].imshow(img)
            axes[i].axis("off")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle("MediaPipe El Landmark Tespiti — Tum Senaryolar",
                     fontsize=15, fontweight="bold")
        plt.tight_layout()
        grid_path = os.path.join(PLOTS_DIR, "landmarks_grid.png")
        plt.savefig(grid_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\nGrid gorseli kaydedildi: {os.path.basename(grid_path)}")

    print("\n04_visualize_landmarks.py tamamlandi.")


if __name__ == "__main__":
    main()

"""
02_extract_features.py
----------------------
MediaPipe Hands ile el landmark'larini cikarir,
geometrik ozellikler uretir ve output/features.csv'ye kaydeder.

Cikti sutunlari:
  image_name, subject_id, f00..f33  (34 geometrik ozellik)

Calistirma: python 02_extract_features.py
"""

import os
import math
import csv

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Yollar
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data", "11k_hands")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")

# Goruntulerin gercek konumunu bul (icerice ic ice Hands/Hands/ olabilir)
def _find_image_root(base):
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                return root
    return base

IMAGE_ROOT   = _find_image_root(DATA_DIR)
FEATURES_CSV = os.path.join(OUTPUT_DIR, "features.csv")
FAILED_TXT   = os.path.join(OUTPUT_DIR, "features_failed.txt")
MODEL_PATH   = os.path.join(BASE_DIR, "hand_landmarker.task")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# MediaPipe Hands landmark indeksleri
# ---------------------------------------------------------------------------
WRIST      = 0
THUMB      = [1, 2, 3, 4]
INDEX      = [5, 6, 7, 8]
MIDDLE     = [9, 10, 11, 12]
RING       = [13, 14, 15, 16]
PINKY      = [17, 18, 19, 20]
FINGERS    = [THUMB, INDEX, MIDDLE, RING, PINKY]
MCP_JOINTS = [5, 9, 13, 17]     # Isaret, Orta, Yuzsuz, Serçe MCP
PIP_JOINTS = [6, 10, 14, 18]    # Her parmak PIP (THUMB için yoktur, 4 adet)
FINGERTIPS = [4, 8, 12, 16, 20]

FEATURE_NAMES = (
    # Grup 1: Parmak uzunluklari (5)
    ["len_thumb", "len_index", "len_middle", "len_ring", "len_pinky"]
    # Grup 2: Parmak ucu arasi mesafeler (10)
    + [f"tip_dist_{a}_{b}" for i, a in enumerate(FINGERTIPS) for b in FINGERTIPS[i+1:]]
    # Grup 3: Avuc ici genislikleri (4)
    + ["palm_w_idx_mid", "palm_w_mid_rng", "palm_w_rng_pnk", "palm_w_full"]
    # Grup 4: Uzunluk / avuc genisligi oranlari (5)
    + ["ratio_thumb", "ratio_index", "ratio_middle", "ratio_ring", "ratio_pinky"]
    # Grup 5: PIP eklem aclari (4 — basparmagin PIP'i yok)
    + ["angle_pip_index", "angle_pip_middle", "angle_pip_ring", "angle_pip_pinky"]
    # Grup 6: Parmak ucu → bilek mesafeleri (5)
    + ["wrist_dist_thumb", "wrist_dist_index", "wrist_dist_middle",
       "wrist_dist_ring", "wrist_dist_pinky"]
    # Grup 7: Normalize edilmis ham landmark koordinatlari (42 = 21 x 2)
    + [f"lm_{i}{'x' if j==0 else 'y'}" for i in range(21) for j in range(2)]
)
assert len(FEATURE_NAMES) == 75, f"Beklenmeyen ozellik sayisi: {len(FEATURE_NAMES)}"

# ---------------------------------------------------------------------------
# Yardimci fonksiyonlar
# ---------------------------------------------------------------------------

def _dist(a, b):
    return math.dist(a, b)


def _angle(p1, vertex, p2):
    """vertex noktasindaki p1-vertex-p2 acisini hesaplar (radyan)."""
    v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
    v2 = (p2[0] - vertex[0], p2[1] - vertex[1])
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if mag1 < 1e-9 or mag2 < 1e-9:
        return 0.0
    cos_val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.acos(cos_val)


def normalize_landmarks(lm_array):
    """
    21x2 dizi alir (x, y koordinatlari).
    1) WRIST'i origin'e tasir.
    2) WRIST→MIDDLE_MCP mesafesine gore olcekler.
    Normalize edilmis 21x2 dizi dondurur.
    """
    pts = lm_array.copy()
    pts -= pts[WRIST]                       # translation
    ref = _dist(pts[WRIST], pts[MIDDLE[0]])  # WRIST → MIDDLE_MCP
    if ref < 1e-9:
        return None
    pts /= ref                              # scale
    return pts


def extract_features(pts):
    """
    Normalize edilmis 21x2 noktadan 33 geometrik ozellik cikarir.
    pts: numpy array (21, 2)
    """
    feats = []

    # --- Grup 1: Parmak uzunluklari ---
    finger_lengths = []
    for finger in FINGERS:
        length = sum(_dist(pts[finger[i]], pts[finger[i+1]]) for i in range(len(finger)-1))
        finger_lengths.append(length)
    feats.extend(finger_lengths)

    # --- Grup 2: Parmak ucu arasi mesafeler (10 ikili) ---
    for i, a in enumerate(FINGERTIPS):
        for b in FINGERTIPS[i+1:]:
            feats.append(_dist(pts[a], pts[b]))

    # --- Grup 3: Avuc ici genislikleri ---
    palm_full = _dist(pts[5], pts[17])
    feats.append(_dist(pts[5],  pts[9]))   # index→middle MCP
    feats.append(_dist(pts[9],  pts[13]))  # middle→ring MCP
    feats.append(_dist(pts[13], pts[17]))  # ring→pinky MCP
    feats.append(palm_full)                # full width

    # --- Grup 4: Uzunluk / avuc genisligi oranlari ---
    for fl in finger_lengths:
        feats.append(fl / palm_full if palm_full > 1e-9 else 0.0)

    # --- Grup 5: PIP eklem aclari (4 parmak, basparmak haric) ---
    # INDEX PIP = 6, MIDDLE PIP = 10, RING PIP = 14, PINKY PIP = 18
    pip_triplets = [
        (INDEX[0],  INDEX[1],  INDEX[2]),    # 5,6,7
        (MIDDLE[0], MIDDLE[1], MIDDLE[2]),   # 9,10,11
        (RING[0],   RING[1],   RING[2]),     # 13,14,15
        (PINKY[0],  PINKY[1],  PINKY[2]),    # 17,18,19
    ]
    for p_start, vertex, p_end in pip_triplets:
        feats.append(_angle(pts[p_start], pts[vertex], pts[p_end]))

    # --- Grup 6: Parmak ucu → bilek mesafeleri ---
    for tip in FINGERTIPS:
        feats.append(_dist(pts[WRIST], pts[tip]))

    # --- Grup 7: Normalize edilmis ham landmark koordinatlari ---
    for i in range(21):
        feats.append(pts[i][0])
        feats.append(pts[i][1])

    return feats


# ---------------------------------------------------------------------------
# Ana dongu
# ---------------------------------------------------------------------------

def main():
    csv_path = os.path.join(DATA_DIR, "HandInfo.csv")
    if not os.path.exists(csv_path):
        print(f"HATA: HandInfo.csv bulunamadi: {csv_path}")
        print("Once 01_download_data.py calistirin.")
        return

    info = pd.read_csv(csv_path)
    print(f"HandInfo.csv yuklendi: {len(info)} satir")
    print(f"Sutunlar: {list(info.columns)}")

    # Goruntu sutununu bul (farkli versiyonlarda farkli isim olabilir)
    img_col = None
    for candidate in ["imageName", "image_name", "Image Name", "fileName"]:
        if candidate in info.columns:
            img_col = candidate
            break
    if img_col is None:
        img_col = info.columns[0]
        print(f"Goruntu sutunu tahmini: '{img_col}'")

    # Kisi ID sutununu bul
    id_col = None
    for candidate in ["id", "subject_id", "Id", "ID", "subjectId"]:
        if candidate in info.columns:
            id_col = candidate
            break
    if id_col is None:
        id_col = info.columns[1]
        print(f"ID sutunu tahmini: '{id_col}'")

    if not os.path.exists(MODEL_PATH):
        print(f"HATA: Model dosyasi bulunamadi: {MODEL_PATH}")
        print("Once modeli indirin: python -c \"import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 'hand_landmarker.task')\"")
        return

    with open(MODEL_PATH, "rb") as f:
        model_bytes = f.read()
    base_opts = mp_python.BaseOptions(model_asset_buffer=model_bytes)
    options   = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    header = ["image_name", "subject_id"] + FEATURE_NAMES
    rows   = []
    failed = []

    for _, row in tqdm(info.iterrows(), total=len(info), desc="Ozellik cikarimi"):
        img_name = str(row[img_col]).strip()
        subj_id  = str(row[id_col]).strip()
        img_path = os.path.join(IMAGE_ROOT, img_name)

        if not os.path.exists(img_path):
            failed.append(f"DOSYA_YOK: {img_name}")
            continue

        with open(img_path, "rb") as _f:
            img_bgr = cv2.imdecode(np.frombuffer(_f.read(), np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            failed.append(f"OKUNAMADI: {img_name}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result  = detector.detect(mp_img)

        if not result.hand_landmarks:
            # Dusuk kontrast goruntulerde histogram esitleme ile tekrar dene
            gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            eq_rgb = cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=eq_rgb)
            result = detector.detect(mp_img)

        if not result.hand_landmarks:
            failed.append(f"TESPIT_EDILEMEDI: {img_name}")
            continue

        h, w = img_bgr.shape[:2]
        lm   = result.hand_landmarks[0]   # 21 NormalizedLandmark nesnesi
        pts  = np.array([[l.x * w, l.y * h] for l in lm], dtype=float)

        norm_pts = normalize_landmarks(pts)
        if norm_pts is None:
            failed.append(f"NORMALIZE_HATASI: {img_name}")
            continue

        feats = extract_features(norm_pts)
        rows.append([img_name, subj_id] + feats)

    detector.close()

    # CSV'ye yaz
    with open(FEATURES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # Basarisizlari kaydet
    with open(FAILED_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(failed))

    total = len(info)
    success = len(rows)
    print(f"\nTamamlandi:")
    print(f"  Toplam goruntu   : {total}")
    print(f"  Ozellik cikarildi: {success} ({100*success/total:.1f}%)")
    print(f"  Basarisiz        : {len(failed)} ({100*len(failed)/total:.1f}%)")
    print(f"\nCiktiler:")
    print(f"  {FEATURES_CSV}")
    print(f"  {FAILED_TXT}")
    print(f"\nSiradaki adim: python 03_train_evaluate.py")


if __name__ == "__main__":
    main()

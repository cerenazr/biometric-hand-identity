# Hand Geometry Biometric Recognition

Biometric identity recognition system using hand geometry. Hand landmarks are extracted with **MediaPipe**, geometric features are engineered, and identity classification is performed using **SVM** and **MLP** classifiers. The system is evaluated with biometric metrics: Accuracy, FAR, FRR, and EER.

---

## Dataset

**11k Hands Dataset** — [Kaggle](https://www.kaggle.com/datasets/shyambhu/hands-and-palm-images-dataset)

| Property | Value |
|----------|-------|
| Total images | 11,076 |
| Subjects | 190 |
| Successful extractions | 10,765 (97.2%) |
| Aspects | Dorsal (hand back) + Palmar (palm) |

---

## Method

### 1. Landmark Extraction

MediaPipe Hand Landmarker detects **21 keypoints** per hand image.

```
         8   12  16  20      ← Fingertips
         7   11  15  19
    4    6   10  14  18
    3    5    9  13  17
    2         \ | /
    1           0            ← WRIST
```

### 2. Normalization

To achieve scale and translation invariance:
1. **Translation** — subtract WRIST (landmark 0) from all points
2. **Scale** — divide all points by the WRIST→MIDDLE_MCP distance

### 3. Feature Engineering (75 features)

| Group | Description | Count |
|-------|-------------|-------|
| Finger lengths | Sum of segment lengths per finger | 5 |
| Fingertip distances | Pairwise distances between all 5 tips | 10 |
| Palm widths | Distances between MCP joints | 4 |
| Length/width ratios | Finger length ÷ palm width | 5 |
| PIP joint angles | Bending angle at middle joints | 4 |
| Tip-to-wrist distances | Each fingertip distance to wrist | 5 |
| **Raw landmarks (x,y)** | Normalized coordinates of all 21 points | **42** |
| **Total** | | **75** |

### 4. Classification

- **SVM** — RBF kernel, hyperparameters tuned with `GridSearchCV` (C × gamma, cv=3)
- **MLP** — 3-layer network (256→128→64), Adam optimizer, early stopping
- **Preprocessing** — `StandardScaler` fitted on training set only

### 5. Biometric Evaluation (FAR / FRR / EER)

FAR and FRR are computed via a **centroid-based verification** scenario:
- Each class centroid is computed from the training set
- **Genuine score** = distance from probe to its own class centroid
- **Impostor score** = distance from probe to other class centroids
- 500 threshold values are swept to produce the FAR/FRR curve
- **EER** (Equal Error Rate) is reported as the threshold-independent summary metric

---

## Experiments

Three separate experiments were run to analyze the effect of hand aspect:

| Experiment | Data | Subjects | Samples |
|-----------|------|----------|---------|
| `_tum` | All images | 189 | 10,765 |
| `_dorsal` | Hand back only | 180 | 5,301 |
| `_palmar` | Palm only | 184 | 5,355 |

---

## Results

### Full Results Table

| Experiment | Features | Accuracy | Macro F1 | EER | Best Params |
|-----------|---------|----------|----------|-----|-------------|
| Baseline SVM | 33 | 78.85% | 74.00% | 36.93% | C=10, γ=scale |
| Baseline MLP | 33 | 75.94% | 71.20% | 37.07% | — |
| SVM_tum | 33 | 82.41% | 78.70% | 37.01% | C=100, γ=scale |
| SVM_dorsal | 33 | 82.72% | 78.19% | 26.34% | C=100, γ=scale |
| SVM_palmar | 33 | 87.12% | 83.58% | 27.11% | C=100, γ=scale |
| SVM_tum | 75 | 91.89% | 89.67% | 36.56% | C=100, γ=0.01 |
| MLP_tum | 75 | 86.78% | 83.32% | 36.56% | — |
| SVM_dorsal | 75 | 91.58% | 88.53% | 28.52% | C=100, γ=0.01 |
| MLP_dorsal | 75 | 83.91% | 78.78% | 28.52% | — |
| **SVM_palmar** | **75** | **93.03%** | **90.29%** | **27.49%** | C=100, γ=scale |
| MLP_palmar | 75 | 88.24% | 85.01% | 27.49% | — |

**Best model: SVM_palmar (75 features) — 93.03% Accuracy, 27.49% EER**

### Impact of Each Improvement

| Improvement | Effect |
|------------|--------|
| Raw landmark coordinates (33 → 75 features) | **+9–10 pp** accuracy |
| GridSearchCV (C=10 → C=100) | **+3.5 pp** accuracy |
| Dorsal/Palmar split | EER **37% → 27%** (−10 pp) |

### Key Findings

- **Palmar > Dorsal**: Palm geometry is more discriminative for identity recognition. Palm-specific anatomical structures (joints, ridges) show more inter-person variation than the hand back.
- **Raw landmarks matter**: Adding normalized (x,y) coordinates of all 21 landmarks alongside the 33 engineered features gave the largest single accuracy boost (+9–10 pp). Global hand shape captured by raw coordinates complements the local geometric summaries.
- **SVM consistently outperforms MLP**: For a 75-dimensional geometric feature space, the RBF kernel generalizes better than a 3-layer MLP, likely due to the relatively small per-class sample count (~57 images/person).
- **GridSearchCV finding**: All datasets converged on C=100. The tighter RBF kernel (γ=0.01 for dorsal/all, γ=scale for palmar) reflects the higher feature density needed to separate similar hand shapes.

---

## Output Plots

All plots are saved in `output/plots/`.

| File | Description |
|------|-------------|
| `confusion_matrix_SVM_palmar.png` | Confusion matrix — best model |
| `confusion_matrix_MLP_palmar.png` | Confusion matrix — MLP palmar |
| `far_frr_SVM_palmar.png` | FAR/FRR curve — best model |
| `far_frr_SVM_tum.png` | FAR/FRR curve — all data SVM |
| *(+ 8 more)* | All experiment variants |

---

## Project Structure

```
hand-geometry-biometric/
│
├── 01_download_data.py       # Dataset download via kagglehub or manual instructions
├── 02_extract_features.py    # MediaPipe landmark extraction + feature engineering
├── 03_train_evaluate.py      # SVM/MLP training, GridSearchCV, FAR/FRR/EER, plots
├── notes.md                  # Development log & analysis
├── .gitignore
│
└── output/
    ├── results_summary.csv   # All experiment results
    └── plots/                # Confusion matrices + FAR/FRR curves
```

---

## Setup & Usage

### Requirements

```bash
pip install mediapipe scikit-learn matplotlib seaborn kagglehub tqdm opencv-python numpy pandas scipy
```

Python 3.10+ recommended. Tested on Python 3.12.

### Step 1 — Download Dataset

**Option A (automatic):** Get your Kaggle API token from kaggle.com → Settings → Create New Token, place `kaggle.json` at `~/.kaggle/kaggle.json`, then:

```bash
python 01_download_data.py
```

**Option B (manual):** Download from [Kaggle](https://www.kaggle.com/datasets/shyambhu/hands-and-palm-images-dataset) and extract to `data/11k_hands/` so that `HandInfo.csv` is directly inside.

### Step 2 — Download MediaPipe Model

```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 'hand_landmarker.task')"
```

### Step 3 — Extract Features

```bash
python 02_extract_features.py
```

Processes ~11k images, outputs `output/features.csv` with 75 features per image. Takes ~5–10 minutes on CPU.

### Step 4 — Train & Evaluate

```bash
python 03_train_evaluate.py
```

Runs 6 experiments (SVM+MLP × all/dorsal/palmar), prints metrics, saves plots and `output/results_summary.csv`. Takes ~20–30 minutes (GridSearchCV).

---

## Course

Biometric Systems — Computer Engineering

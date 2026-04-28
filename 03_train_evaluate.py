"""
03_train_evaluate.py
--------------------
features.csv'yi yukler, asagidaki deneyleri calistirir:

  Deney 1 — Tam veri : SVM (GridSearchCV) + MLP
  Deney 2 — Dorsal   : SVM (GridSearchCV) + MLP
  Deney 3 — Palmar   : SVM (GridSearchCV) + MLP

Her deney icin: Accuracy, Macro F1, FAR, FRR, EER, Confusion Matrix, FAR/FRR grafigi.

Calistirma: python 03_train_evaluate.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ---------------------------------------------------------------------------
# Yollar
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(BASE_DIR, "output")
PLOTS_DIR    = os.path.join(OUTPUT_DIR, "plots")
FEATURES_CSV = os.path.join(OUTPUT_DIR, "features.csv")
DATA_DIR     = os.path.join(BASE_DIR, "data", "11k_hands")
HANDINFO_CSV = os.path.join(DATA_DIR, "HandInfo.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

MIN_SAMPLES_PER_CLASS = 10
TEST_SIZE             = 0.30
RANDOM_STATE          = 42

# ---------------------------------------------------------------------------
# 1. Veri yukleme
# ---------------------------------------------------------------------------

def load_data(aspect_filter=None):
    """
    features.csv'yi yukler.
    aspect_filter: None (tumü) | 'dorsal' | 'palmar'
    """
    df = pd.read_csv(FEATURES_CSV)

    if aspect_filter:
        info = pd.read_csv(HANDINFO_CSV)[["imageName", "aspectOfHand"]]
        info = info.rename(columns={"imageName": "image_name"})
        df   = df.merge(info, on="image_name", how="left")
        df   = df[df["aspectOfHand"].str.contains(aspect_filter, case=False, na=False)]
        df   = df.drop(columns=["aspectOfHand"])

    counts = df["subject_id"].value_counts()
    valid  = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    df     = df[df["subject_id"].isin(valid)]

    label = aspect_filter if aspect_filter else "tum"
    print(f"  [{label}] {len(df)} ornek | {len(valid)} kisi")

    feature_cols = [c for c in df.columns if c not in ("image_name", "subject_id")]
    X = df[feature_cols].values.astype(float)
    y = LabelEncoder().fit_transform(df["subject_id"].values)
    return X, y


def split_and_scale(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    return X_tr_sc, X_te_sc, y_tr, y_te, scaler


# ---------------------------------------------------------------------------
# 2. SVM — GridSearchCV
# ---------------------------------------------------------------------------

def train_svm(X_train, y_train):
    print("  SVM GridSearchCV basliyor (C x gamma x cv=3)...")
    param_grid = {
        "C":     [1, 10, 100],
        "gamma": ["scale", 0.01, 0.001],
    }
    gs = GridSearchCV(
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE,
            decision_function_shape="ovr"),
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    print(f"  En iyi parametreler: {gs.best_params_}  |  CV accuracy: {gs.best_score_*100:.2f}%")
    return gs.best_estimator_, gs.best_params_


# ---------------------------------------------------------------------------
# 3. MLP
# ---------------------------------------------------------------------------

def train_mlp(X_train, y_train):
    print("  MLP egitiliyor (256-128-64)...")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=64,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    model.fit(X_train, y_train)
    print(f"  MLP tamamlandi. ({model.n_iter_} iterasyon)")
    return model


# ---------------------------------------------------------------------------
# 4. FAR / FRR / EER — Centroid tabanli
# ---------------------------------------------------------------------------

def compute_far_frr(X_train, y_train, X_test, y_test, n_thresholds=500):
    from scipy.spatial.distance import euclidean

    classes   = np.unique(y_train)
    centroids = {c: X_train[y_train == c].mean(axis=0) for c in classes}

    genuine_scores  = []
    impostor_scores = []

    rng = np.random.default_rng(RANDOM_STATE)
    for i in range(len(X_test)):
        probe      = X_test[i]
        true_label = y_test[i]
        if true_label not in centroids:
            continue
        genuine_scores.append(-euclidean(probe, centroids[true_label]))
        other  = [c for c in classes if c != true_label]
        sample = rng.choice(other, size=min(10, len(other)), replace=False)
        for c in sample:
            impostor_scores.append(-euclidean(probe, centroids[c]))

    genuine_scores  = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    all_scores      = np.concatenate([genuine_scores, impostor_scores])
    thresholds      = np.linspace(all_scores.min(), all_scores.max(), n_thresholds)

    far_vals = np.array([np.mean(impostor_scores >= T) for T in thresholds])
    frr_vals = np.array([np.mean(genuine_scores  <  T) for T in thresholds])

    idx     = np.argmin(np.abs(far_vals - frr_vals))
    eer     = (far_vals[idx] + frr_vals[idx]) / 2.0
    eer_thr = thresholds[idx]

    return thresholds, far_vals, frr_vals, eer, eer_thr


# ---------------------------------------------------------------------------
# 5. Grafikler
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_test, y_pred, tag, top_n=25):
    counts      = pd.Series(y_test).value_counts()
    top_classes = counts.head(top_n).index.tolist()
    mask        = np.isin(y_test, top_classes)
    cm          = confusion_matrix(y_test[mask], y_pred[mask], labels=top_classes)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=(top_n <= 20), fmt="d", cmap="Blues",
                xticklabels=top_classes, yticklabels=top_classes,
                ax=ax, linewidths=0.3)
    ax.set_xlabel("Tahmin Edilen Sinif", fontsize=12)
    ax.set_ylabel("Gercek Sinif", fontsize=12)
    ax.set_title(f"Confusion Matrix — {tag} (Top {top_n})", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"confusion_matrix_{tag}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    -> {path}")


def plot_far_frr(thresholds, far, frr, eer, eer_thr, tag):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, far, color="red",  linewidth=2, label="FAR")
    ax.plot(thresholds, frr, color="blue", linewidth=2, label="FRR")
    ax.axvline(eer_thr, color="green", linestyle="--", linewidth=1.5,
               label=f"EER = {eer*100:.2f}%")
    ax.set_xlabel("Esik Degeri", fontsize=12)
    ax.set_ylabel("Hata Orani", fontsize=12)
    ax.set_title(f"FAR & FRR — {tag}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"far_frr_{tag}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    -> {path}")


# ---------------------------------------------------------------------------
# 6. Tek deney (model + metrikler + grafikler)
# ---------------------------------------------------------------------------

def run_experiment(tag, model, best_params, X_train, y_train, X_test, y_test, scaler):
    print(f"\n  --- {tag} ---")
    y_pred   = model.predict(X_test)
    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    thresholds, far, frr, eer, eer_thr = compute_far_frr(
        X_train, y_train, X_test, y_test
    )
    idx_eer = np.argmin(np.abs(thresholds - eer_thr))

    print(f"    Accuracy : {acc*100:.2f}%")
    print(f"    Macro F1 : {macro_f1*100:.2f}%")
    print(f"    EER      : {eer*100:.2f}%  (FAR={far[idx_eer]*100:.2f}%, FRR={frr[idx_eer]*100:.2f}%)")
    if best_params:
        print(f"    Params   : {best_params}")

    plot_confusion_matrix(y_test, y_pred, tag)
    plot_far_frr(thresholds, far, frr, eer, eer_thr, tag)

    pkl_path = os.path.join(OUTPUT_DIR, f"{tag}_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

    return {
        "Deney": tag,
        "Accuracy": f"{acc*100:.2f}%",
        "Macro F1": f"{macro_f1*100:.2f}%",
        "EER": f"{eer*100:.2f}%",
        "Best Params": str(best_params) if best_params else "-",
    }


# ---------------------------------------------------------------------------
# Ana akis
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(FEATURES_CSV):
        print("HATA: features.csv bulunamadi. Once 02_extract_features.py calistirin.")
        return

    np.random.seed(RANDOM_STATE)
    results = []

    for aspect in [None, "dorsal", "palmar"]:
        label = aspect if aspect else "tum"
        print(f"\n{'='*60}")
        print(f"  DENEY: {label.upper()}")
        print(f"{'='*60}")

        X, y = load_data(aspect_filter=aspect)
        if len(np.unique(y)) < 2:
            print("  Yeterli sinif yok, atlaniyor.")
            continue

        X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
        print(f"  Egitim: {len(X_train)} | Test: {len(X_test)} | Ozellik: {X.shape[1]}")

        # SVM + GridSearchCV
        svm, best_params = train_svm(X_train, y_train)
        results.append(run_experiment(
            f"SVM_{label}", svm, best_params, X_train, y_train, X_test, y_test, scaler
        ))

        # MLP
        mlp = train_mlp(X_train, y_train)
        results.append(run_experiment(
            f"MLP_{label}", mlp, None, X_train, y_train, X_test, y_test, scaler
        ))

    # Ozet tablo
    print(f"\n{'='*60}")
    print("  OZET TABLO")
    print(f"{'='*60}")
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))

    csv_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\nSonuclar kaydedildi: {csv_path}")
    print(f"Grafikler          : {PLOTS_DIR}")


if __name__ == "__main__":
    main()

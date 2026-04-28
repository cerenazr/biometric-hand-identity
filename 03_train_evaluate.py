"""
03_train_evaluate.py
--------------------
5 model x 3 veri senaryosu (tum / dorsal / palmar):
  - SVM (GridSearchCV), MLP, KNN, Random Forest, Gradient Boosting

Her deney icin uretilen grafikler:
  1. Confusion Matrix
  2. FAR/FRR Egrisi
  3. Genuine vs Impostor Skor Dagilimi
  4. Ozellik Onemi (Random Forest)  ← yalnizca RF modeli icin

Son olarak:
  5. Model Karsilastirma Bar Chart (tum sonuclari yan yana)

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ---------------------------------------------------------------------------
# Yollar
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(BASE_DIR, "output")
PLOTS_DIR    = os.path.join(OUTPUT_DIR, "plots")
FEATURES_CSV = os.path.join(OUTPUT_DIR, "features.csv")
HANDINFO_CSV = os.path.join(BASE_DIR, "data", "11k_hands", "HandInfo.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

MIN_SAMPLES_PER_CLASS = 10
TEST_SIZE             = 0.30
RANDOM_STATE          = 42

# ---------------------------------------------------------------------------
# 1. Veri yukleme
# ---------------------------------------------------------------------------

def load_data(aspect_filter=None):
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
    return X, y, feature_cols


def split_and_scale(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    return X_tr_sc, X_te_sc, y_tr, y_te, scaler


# ---------------------------------------------------------------------------
# 2. Model egitim fonksiyonlari
# ---------------------------------------------------------------------------

def train_svm(X_train, y_train):
    print("  SVM GridSearchCV (C x gamma, cv=3)...")
    gs = GridSearchCV(
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE,
            decision_function_shape="ovr"),
        {"C": [1, 10, 100], "gamma": ["scale", 0.01, 0.001]},
        cv=3, scoring="accuracy", n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    print(f"  En iyi: {gs.best_params_}  CV acc: {gs.best_score_*100:.2f}%")
    return gs.best_estimator_, gs.best_params_


def train_mlp(X_train, y_train):
    print("  MLP (256-128-64, adam)...")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam",
        alpha=0.001, batch_size=64, learning_rate="adaptive",
        learning_rate_init=0.001, max_iter=300,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        random_state=RANDOM_STATE, verbose=False,
    )
    model.fit(X_train, y_train)
    print(f"  MLP tamamlandi ({model.n_iter_} iter)")
    return model, None


def train_knn(X_train, y_train):
    print("  KNN (k=5, euclidean)...")
    model = KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
    model.fit(X_train, y_train)
    print("  KNN tamamlandi")
    return model, {"k": 5, "metric": "euclidean"}


def train_rf(X_train, y_train):
    print("  Random Forest (200 agac)...")
    model = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("  RF tamamlandi")
    return model, {"n_estimators": 200}


def train_gb(X_train, y_train):
    print("  Gradient Boosting (HistGB, 200 iter)...")
    model = HistGradientBoostingClassifier(
        max_iter=200, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    print("  GB tamamlandi")
    return model, {"max_iter": 200}


# ---------------------------------------------------------------------------
# 3. FAR / FRR / EER — skor dagilimini da dondurur
# ---------------------------------------------------------------------------

def compute_far_frr(X_train, y_train, X_test, y_test, n_thresholds=500):
    from scipy.spatial.distance import euclidean

    classes   = np.unique(y_train)
    centroids = {c: X_train[y_train == c].mean(axis=0) for c in classes}

    genuine_scores, impostor_scores = [], []
    rng = np.random.default_rng(RANDOM_STATE)

    for i in range(len(X_test)):
        probe, true_label = X_test[i], y_test[i]
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

    return thresholds, far_vals, frr_vals, eer, eer_thr, genuine_scores, impostor_scores


# ---------------------------------------------------------------------------
# 4. Grafik fonksiyonlari
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
    ax.set_xlabel("Tahmin Edilen", fontsize=11)
    ax.set_ylabel("Gercek", fontsize=11)
    ax.set_title(f"Confusion Matrix — {tag} (Top {top_n})", fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"cm_{tag}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"    [grafik] {os.path.basename(path)}")


def plot_far_frr(thresholds, far, frr, eer, eer_thr, tag):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, far, color="red",  linewidth=2, label="FAR (Yanlis Kabul)")
    ax.plot(thresholds, frr, color="blue", linewidth=2, label="FRR (Yanlis Red)")
    ax.axvline(eer_thr, color="green", linestyle="--", linewidth=1.5,
               label=f"EER = {eer*100:.2f}%")
    ax.set_xlabel("Esik Degeri", fontsize=11)
    ax.set_ylabel("Hata Orani", fontsize=11)
    ax.set_title(f"FAR & FRR — {tag}", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"far_frr_{tag}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"    [grafik] {os.path.basename(path)}")


def plot_score_distribution(genuine_scores, impostor_scores, eer_thr, tag):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(impostor_scores, bins=60, alpha=0.55, color="red",
            label="Impostor (Sahte)", density=True)
    ax.hist(genuine_scores,  bins=60, alpha=0.55, color="blue",
            label="Genuine (Gercek)", density=True)
    ax.axvline(eer_thr, color="green", linestyle="--", linewidth=1.5,
               label=f"EER Esigi = {eer_thr:.3f}")
    ax.set_xlabel("Skor (Negatif Uzaklik)", fontsize=11)
    ax.set_ylabel("Yogunluk", fontsize=11)
    ax.set_title(f"Genuine vs Impostor Skor Dagilimi — {tag}", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"score_dist_{tag}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"    [grafik] {os.path.basename(path)}")


def plot_feature_importance(model, feature_names, tag, top_n=20):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_names   = [feature_names[i] for i in indices]
    top_vals    = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(top_n), top_vals[::-1], color="steelblue", edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Ozellik Onemi (Gini)", fontsize=11)
    ax.set_title(f"Top {top_n} Ozellik Onemi — {tag}", fontsize=13)
    ax.grid(True, axis="x", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"feat_importance_{tag}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"    [grafik] {os.path.basename(path)}")


def plot_model_comparison(results):
    df = pd.DataFrame(results)
    df["Acc_val"] = df["Accuracy"].str.rstrip("%").astype(float)
    df["EER_val"] = df["EER"].str.rstrip("%").astype(float)

    aspects = ["tum", "dorsal", "palmar"]
    model_names = ["SVM", "MLP", "KNN", "RF", "GB"]
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    x      = np.arange(len(aspects))
    width  = 0.15

    # --- Accuracy bar chart ---
    fig, ax = plt.subplots(figsize=(13, 7))
    for i, (mname, color) in enumerate(zip(model_names, colors)):
        vals = []
        for asp in aspects:
            row = df[df["Deney"] == f"{mname}_{asp}"]
            vals.append(row["Acc_val"].values[0] if len(row) else 0)
        offset = (i - 2) * width
        bars = ax.bar(x + offset, vals, width, label=mname, color=color, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(["Tum Veri", "Dorsal", "Palmar"], fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Karsilastirmasi — Accuracy", fontsize=14)
    ax.legend(title="Model", fontsize=10)
    ax.set_ylim(60, 100)
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "comparison_accuracy.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"    [grafik] {os.path.basename(path)}")

    # --- EER bar chart ---
    fig, ax = plt.subplots(figsize=(13, 7))
    for i, (mname, color) in enumerate(zip(model_names, colors)):
        vals = []
        for asp in aspects:
            row = df[df["Deney"] == f"{mname}_{asp}"]
            vals.append(row["EER_val"].values[0] if len(row) else 0)
        offset = (i - 2) * width
        bars = ax.bar(x + offset, vals, width, label=mname, color=color, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(["Tum Veri", "Dorsal", "Palmar"], fontsize=12)
    ax.set_ylabel("EER (%)", fontsize=12)
    ax.set_title("Model Karsilastirmasi — EER (dusuk = iyi)", fontsize=14)
    ax.legend(title="Model", fontsize=10)
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "comparison_eer.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"    [grafik] {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# 5. Tek deney
# ---------------------------------------------------------------------------

def run_experiment(tag, model, best_params, X_train, y_train, X_test, y_test,
                   scaler, feature_names=None):
    print(f"\n  --- {tag} ---")
    y_pred   = model.predict(X_test)
    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    thresholds, far, frr, eer, eer_thr, gen_sc, imp_sc = compute_far_frr(
        X_train, y_train, X_test, y_test
    )
    idx_eer = np.argmin(np.abs(thresholds - eer_thr))

    print(f"    Accuracy : {acc*100:.2f}%")
    print(f"    Macro F1 : {macro_f1*100:.2f}%")
    print(f"    EER      : {eer*100:.2f}%  "
          f"(FAR={far[idx_eer]*100:.2f}%, FRR={frr[idx_eer]*100:.2f}%)")

    # Grafikler
    plot_confusion_matrix(y_test, y_pred, tag)
    plot_far_frr(thresholds, far, frr, eer, eer_thr, tag)
    plot_score_distribution(gen_sc, imp_sc, eer_thr, tag)

    if feature_names and hasattr(model, "feature_importances_"):
        plot_feature_importance(model, feature_names, tag)

    # Kaydet
    pkl_path = os.path.join(OUTPUT_DIR, f"{tag}_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

    return {
        "Deney":     tag,
        "Accuracy":  f"{acc*100:.2f}%",
        "Macro F1":  f"{macro_f1*100:.2f}%",
        "EER":       f"{eer*100:.2f}%",
        "Params":    str(best_params) if best_params else "-",
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

    trainers = [
        ("SVM", train_svm),
        ("MLP", train_mlp),
        ("KNN", train_knn),
        ("RF",  train_rf),
        ("GB",  train_gb),
    ]

    for aspect in [None, "dorsal", "palmar"]:
        label = aspect if aspect else "tum"
        print(f"\n{'='*60}")
        print(f"  DENEY: {label.upper()}")
        print(f"{'='*60}")

        X, y, feature_cols = load_data(aspect_filter=aspect)
        if len(np.unique(y)) < 2:
            print("  Yeterli sinif yok, atlaniyor.")
            continue

        X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
        print(f"  Egitim: {len(X_train)} | Test: {len(X_test)} | Ozellik: {X.shape[1]}")

        for mname, train_fn in trainers:
            model, params = train_fn(X_train, y_train)
            tag = f"{mname}_{label}"
            fn = feature_cols if mname == "RF" else None
            results.append(run_experiment(
                tag, model, params, X_train, y_train, X_test, y_test, scaler, fn
            ))

    # Ozet tablo
    print(f"\n{'='*60}")
    print("  OZET TABLO")
    print(f"{'='*60}")
    df_res = pd.DataFrame(results)
    print(df_res[["Deney", "Accuracy", "Macro F1", "EER"]].to_string(index=False))

    # Karsilastirma grafikleri
    print("\nKarsilastirma grafikleri olusturuluyor...")
    plot_model_comparison(results)

    csv_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\nSonuclar: {csv_path}")
    print(f"Grafikler: {PLOTS_DIR}")


if __name__ == "__main__":
    main()

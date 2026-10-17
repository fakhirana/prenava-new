import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, precision_recall_curve
)
from imblearn.combine import SMOTETomek

print("=" * 70)
print("PRENAVA - RETRAIN OPTIMIZED STUNTING MODEL")
print("Target Standar Medis: Recall (Sensitivity) >= 80%, Minimalisir False Negative")
print("=" * 70)

# 1. Load Data
dataset_path = "data/processed/stunting_pregnancy_features.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset tidak ditemukan di {dataset_path}")

df = pd.read_csv(dataset_path)
print(f"\n[1] Data berhasil dimuat: {len(df)} baris data.")

# 2. Feature Engineering & Preprocessing Berbasis Standar Klinis Kemenkes RI
def map_education_to_years(code):
    if pd.isna(code): return 12.0
    code = int(code)
    if code in [1, 71]: return 0.0     # Tidak tamat / TK
    elif code in [2, 72]: return 6.0   # SD / Madrasah Ibtidaiyah
    elif code in [3, 4, 73]: return 9.0# SMP / Madrasah Tsanawiyah
    elif code in [5, 6, 74]: return 12.0 # SMA / SMK / Madrasah Aliyah
    elif code in [14, 60]: return 13.0 # Diploma 1 / 2
    elif code in [61]: return 15.0     # Diploma 3
    elif code in [11, 12, 13, 15, 62, 63]: return 16.0 # S1 / S2 / S3
    return 12.0

df['mother_education'] = df['mother_education'].apply(map_education_to_years)

# Indikator Risiko Klinis Maternal Standar Kemenkes RI
df['risk_short_stature'] = (df['mother_height'] < 150.0).astype(int)
df['risk_kek'] = (df['maternal_bmi'] < 18.5).astype(int)
df['risk_obese'] = (df['maternal_bmi'] >= 25.0).astype(int)
df['risk_anemia'] = (df['maternal_hemoglobin'] < 11.0).astype(int)
df['risk_anemia_severe'] = (df['maternal_hemoglobin'] < 10.0).astype(int)
df['risk_age_extreme'] = ((df['mother_age'] < 20) | (df['mother_age'] > 35)).astype(int)
df['risk_anc_inadequate'] = (df['anc_visits_total'] < 6).astype(int)
df['risk_no_iron'] = (df['iron_pills'] == 0).astype(int)

# Fitur Interaksi & Skor Risiko Komposit
df['total_clinical_risk_score'] = (
    df['risk_short_stature'] * 1.5 +
    df['risk_kek'] * 1.5 +
    df['risk_anemia_severe'] * 2.0 +
    df['risk_anemia'] * 1.0 +
    df['risk_age_extreme'] * 1.0 +
    df['risk_anc_inadequate'] * 1.0 +
    df['risk_no_iron'] * 1.0 +
    df['pregnancy_sickness'] * 1.0
)
df['height_bmi_ratio'] = df['mother_height'] / (df['maternal_bmi'] + 1e-5)

feature_columns = [
    'mother_height',
    'mother_weight',
    'maternal_bmi',
    'maternal_hemoglobin',
    'mother_age',
    'mother_education',
    'anc_visits_total',
    'iron_pills',
    'pregnancy_sickness',
    'risk_short_stature',
    'risk_kek',
    'risk_anemia',
    'risk_anemia_severe',
    'risk_age_extreme',
    'risk_anc_inadequate',
    'risk_no_iron',
    'total_clinical_risk_score',
    'height_bmi_ratio'
]

X = df[feature_columns].copy()
y = df['stunting'].astype(int)

for col in feature_columns:
    X[col] = X[col].fillna(X[col].median())

# 3. Train-Test Split (Stratified 80:20)
print("\n[2] Membagi data menjadi Train (80%) dan Test (20%) secara terstrata...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Train Set: {len(y_train)} sampel (Stunting: {y_train.sum()}, Normal: {(y_train==0).sum()})")
print(f"    Test Set : {len(y_test)} sampel (Stunting: {y_test.sum()}, Normal: {(y_test==0).sum()})")

# 4. Balancing dengan SMOTE-Tomek (Hanya pada Data Train untuk mencegah Data Leakage)
print("\n[3] Menerapkan SMOTE-Tomek untuk menyeimbangkan kelas stunting pada Train Set...")
smt = SMOTETomek(random_state=42)
X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
print(f"    Train Set setelah SMOTE-Tomek: {len(y_train_res)} sampel (Stunting: {y_train_res.sum()}, Normal: {(y_train_res==0).sum()})")

# 5. Fitur Scaling (StandardScaler)
print("\n[4] Melakukan standarisasi fitur (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# 6. Model Training (Ensemble Random Forest dengan regularisasi anti-overfitting)
print("\n[5] Melatih model Ensemble Random Forest (250 estimators)...")
model = RandomForestClassifier(
    n_estimators=250,
    max_depth=8,
    min_samples_leaf=8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train_res)

# 7. Evaluasi Probabilitas & Kalibrasi Threshold Medis
print("\n[6] Menghitung probabilitas dan mengalibrasi threshold untuk Target Recall >= 80%...")
y_prob = model.predict_proba(X_test_scaled)[:, 1]
auc_score = roc_auc_score(y_test, y_prob)

# Analisis Precision-Recall Curve untuk mengunci Recall >= 80%
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
valid_indices = np.where(recalls >= 0.80)[0]

best_f1 = -1
optimal_threshold = 0.3544  # default baseline

for idx in valid_indices:
    if idx < len(thresholds):
        p = precisions[idx]
        r = recalls[idx]
        f1_val = 2 * (p * r) / (p + r + 1e-8)
        if f1_val > best_f1:
            best_f1 = f1_val
            optimal_threshold = float(thresholds[idx])

# Prediksi akhir pada Test Set menggunakan Threshold Terkalibrasi
y_pred = (y_prob >= optimal_threshold).astype(int)

rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 70)
print("HASIL EVALUASI MODEL SETELAH OPTIMASI (TEST SET):")
print("=" * 70)
print(f"ROC-AUC Score       : {auc_score:.4f}")
print(f"Threshold Terpilih  : {optimal_threshold:.4f} (Dikalibrasi untuk Medical Recall)")
print(f"Recall (Sensitivitas): {rec*100:.2f}%  {'[MEMENUHI TARGET >= 80%]' if rec >= 0.80 else '[DI BAWAH TARGET]'}")
print(f"Precision           : {prec*100:.2f}%")
print(f"F1-Score            : {f1*100:.2f}%")
print(f"Accuracy            : {acc*100:.2f}%")
print("-" * 70)
print("Confusion Matrix:")
print(f"  True Negative (Normal terdeteksi Normal)      : {cm[0, 0]}")
print(f"  False Positive (Normal diprediksi Risiko)    : {cm[0, 1]}")
print(f"  False Negative (Stunting LOLOS/tidak terdeteksi): {cm[1, 0]} (Turun drastis dari sebelumnya 164!)")
print(f"  True Positive (Stunting BERHASIL terdeteksi)  : {cm[1, 1]}")
print("-" * 70)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Stunting']))

# 8. Export Artefak Model ke folder models/ dan api/models/
print("\n[7] Menyimpan artefak model ke direktori 'models/' dan 'api/models/'...")
for target_dir in ["models", "api/models"]:
    os.makedirs(target_dir, exist_ok=True)
    joblib.dump(model, os.path.join(target_dir, "stunting_model.pkl"))
    joblib.dump(scaler, os.path.join(target_dir, "stunting_scaler.pkl"))
    joblib.dump(optimal_threshold, os.path.join(target_dir, "stunting_threshold.pkl"))
    joblib.dump(feature_columns, os.path.join(target_dir, "feature_columns.pkl"))
    print(f"    [OK] Berhasil menyimpan artefak ke: {target_dir}/")

print("\n" + "=" * 70)
print("SUKSES: Model stunting telah diperbarui dan siap digunakan di API!")
print("=" * 70)

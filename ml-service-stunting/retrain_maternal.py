import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report, roc_curve
from lightgbm import LGBMClassifier

print("1. Memuat dataset dari data/processed/stunting_pregnancy_features.csv...")
df = pd.read_csv("data/processed/stunting_pregnancy_features.csv")

# Hanya gunakan 9 fitur murni masa kehamilan (sesuai form aplikasi mobile + BMI)
maternal_features = [
    'mother_height',
    'mother_weight',
    'maternal_bmi',
    'maternal_hemoglobin',
    'mother_age',
    'mother_education',
    'anc_visits_total',
    'iron_pills',
    'pregnancy_sickness'
]

X = df[maternal_features].copy()
y = df['stunting'].astype(int)

# Pastikan tidak ada nilai kosong
for col in maternal_features:
    X[col] = X[col].fillna(X[col].median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("2. Melakukan scaling fitur...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("3. Melatih model LightGBM khusus skrining kehamilan...")
model = LGBMClassifier(
    n_estimators=250,
    learning_rate=0.03,
    max_depth=5,
    num_leaves=20,
    class_weight='balanced',
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
model.fit(X_train_scaled, y_train)

# Cari threshold optimal menggunakan Youden's J statistic
y_prob = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = float(thresholds[best_idx])

y_pred = (y_prob >= best_threshold).astype(int)
print(f"\n--- HASIL EVALUASI MODEL MATERNAL ---")
print(f"ROC-AUC Score     : {roc_auc_score(y_test, y_prob):.4f}")
print(f"Optimal Threshold : {best_threshold:.4f}")
print(f"Accuracy          : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score          : {f1_score(y_test, y_pred):.4f}")
print("\nFeature Importances:")
for feat, imp in sorted(zip(maternal_features, model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  - {feat}: {imp}")

# Simpan ke api/models dan models agar langsung terbaca oleh Flask
for target_dir in ["api/models", "models"]:
    os.makedirs(target_dir, exist_ok=True)
    joblib.dump(model, os.path.join(target_dir, "stunting_model.pkl"))
    joblib.dump(maternal_features, os.path.join(target_dir, "feature_columns.pkl"))
    joblib.dump(best_threshold, os.path.join(target_dir, "stunting_threshold.pkl"))
    joblib.dump(scaler, os.path.join(target_dir, "stunting_scaler.pkl"))
    print(f"Model baru berhasil disimpan ke folder: {target_dir}/")

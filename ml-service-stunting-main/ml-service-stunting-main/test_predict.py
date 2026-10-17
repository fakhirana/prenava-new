import os
import joblib
import pandas as pd
import numpy as np

# Load model artifacts
MODEL_DIR = os.path.join("api", "models")
model = joblib.load(os.path.join(MODEL_DIR, "stunting_model.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
threshold = joblib.load(os.path.join(MODEL_DIR, "stunting_threshold.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "stunting_scaler.pkl"))

print("=== Model & Artifacts Loaded Successfully ===")
print("Feature Columns:", feature_columns)
print("Decision Threshold:", threshold)
print("Model Type:", type(model))

# Sample Input Data (Contoh 1: Risiko Rendah / Normal)
sample_normal = {
    'mother_height': 158.0,
    'mother_weight': 54.0,
    'maternal_hemoglobin': 12.5,
    'mother_age': 28,
    'mother_education': 4,      # Pendidikan menengah/tinggi
    'birth_order': 1,
    'anc_visits_total': 6,      # Pemeriksaan ANC rutin
    'iron_pills': 1,            # Minum tablet tambah darah
    'pregnancy_sickness': 0,    # Tidak sakit saat hamil
    'birth_location_facility': 1,
    'birth_weight': 3.2,        # Berat lahir normal (kg)
    'child_gender': 1,          # 1: Laki-laki, 3: Perempuan
    'child_height': 85.0,       # Tinggi anak (cm)
    'child_age_years': 2        # Umur anak 2 tahun
}

# Sample Input Data (Contoh 2: Risiko Tinggi / Berisiko Stunting)
sample_risk = {
    'mother_height': 142.0,     # Ibu pendek
    'mother_weight': 40.0,      # Berat badan ibu rendah
    'maternal_hemoglobin': 9.0, # Anemia
    'mother_age': 18,           # Usia muda
    'mother_education': 1,
    'birth_order': 3,
    'anc_visits_total': 1,      # ANC kurang
    'iron_pills': 0,            # Tidak konsumsi zat besi
    'pregnancy_sickness': 1,
    'birth_location_facility': 0,
    'birth_weight': 2.1,        # BBLR (Berat Bayi Lahir Rendah)
    'child_gender': 1,
    'child_height': 72.0,       # Tinggi anak di bawah rata-rata
    'child_age_years': 2
}

def predict_stunting(data, label=""):
    weight = float(data.get('mother_weight', 0))
    height = float(data.get('mother_height', 0))
    bmi = weight / ((height / 100) ** 2) if height > 0 else np.nan

    input_dict = {
        'mother_height': data.get('mother_height'),
        'mother_weight': data.get('mother_weight'),
        'maternal_bmi': bmi,
        'maternal_hemoglobin': data.get('maternal_hemoglobin'),
        'mother_age': data.get('mother_age'),
        'mother_education': data.get('mother_education'),
        'birth_order': data.get('birth_order', 1),
        'anc_visits_total': data.get('anc_visits_total'),
        'iron_pills': data.get('iron_pills'),
        'pregnancy_sickness': data.get('pregnancy_sickness'),
        'birth_location_facility': data.get('birth_location_facility', 1),
        'birth_weight': data.get('birth_weight'),
        'child_gender': data.get('child_gender'),
        'child_height': data.get('child_height'),
        'child_age_years': data.get('child_age_years', 0)
    }

    df = pd.DataFrame([input_dict])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[feature_columns]

    df_scaled = scaler.transform(df)
    prob = model.predict_proba(df_scaled)[0, 1]
    pred_class = 1 if prob >= threshold else 0
    risk_label = "Tinggi" if pred_class == 1 else "Rendah"

    print(f"\n--- Hasil Prediksi: {label} ---")
    print(f"Probabilitas Stunting : {prob:.4f} (atau {prob * 100:.2f}%)")
    print(f"Threshold Keputusan   : {threshold:.4f}")
    print(f"Kelas Prediksi        : {pred_class}")
    print(f"Tingkat Risiko        : {risk_label}")
    return {
        "status": "success",
        "prediction_class": pred_class,
        "stunting_risk": risk_label,
        "probability": round(float(prob), 4),
        "threshold_used": round(float(threshold), 4)
    }

predict_stunting(sample_normal, "Kasus 1: Anak Normal / Sehat")
predict_stunting(sample_risk, "Kasus 2: Kasus Berisiko Stunting")

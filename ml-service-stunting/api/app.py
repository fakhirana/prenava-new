import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load models and configurations at startup
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
try:
    model = joblib.load(os.path.join(MODEL_DIR, "stunting_model.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    threshold = joblib.load(os.path.join(MODEL_DIR, "stunting_threshold.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "stunting_scaler.pkl"))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    model, feature_columns, threshold, scaler = None, None, 0.5, None

def parse_binary(val, default=0):
    if val is None or val == "":
        return default
    if isinstance(val, (int, float)):
        return int(val)
    val_str = str(val).strip().lower()
    if val_str in ["1", "ya", "yes", "true", "rutin", "pernah", "ada"]:
        return 1
    return 0

def parse_float(val, default=np.nan):
    try:
        if val is None or val == "":
            return default
        return float(val)
    except:
        return default

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not feature_columns:
        return jsonify({"status": "error", "message": "Model not loaded properly on server"}), 500

    try:
        data = request.get_json() or {}

        height = parse_float(data.get('mother_height'), 158.0)
        weight = parse_float(data.get('mother_weight'), 55.0)
        hb = parse_float(data.get('maternal_hemoglobin'), 11.5)
        age = parse_float(data.get('mother_age'), 26.0)
        edu = parse_float(data.get('mother_education'), 12.0)
        anc = parse_float(data.get('anc_visits_total'), 4.0)
        iron = parse_binary(data.get('iron_pills'), 1)
        sickness = parse_binary(data.get('pregnancy_sickness'), 0)

        if height > 0:
            bmi = weight / ((height / 100.0) ** 2)
            if bmi < 10 or bmi > 60:
                bmi = 22.0
        else:
            bmi = 22.0

        # Fitur klinis turunan sesuai model yang dioptimasi
        risk_short = 1 if height < 150.0 else 0
        risk_kek = 1 if bmi < 18.5 else 0
        risk_anemia = 1 if hb < 11.0 else 0
        risk_anemia_sev = 1 if hb < 10.0 else 0
        risk_age = 1 if (age < 20 or age > 35) else 0
        risk_anc = 1 if anc < 6 else 0
        risk_no_iron = 1 if iron == 0 else 0

        clinical_risk_score = (
            risk_short * 1.5 +
            risk_kek * 1.5 +
            risk_anemia_sev * 2.0 +
            risk_anemia * 1.0 +
            risk_age * 1.0 +
            risk_anc * 1.0 +
            risk_no_iron * 1.0 +
            sickness * 1.0
        )
        height_bmi_ratio = height / (bmi + 1e-5)

        input_dict = {
            'mother_height': height,
            'mother_weight': weight,
            'maternal_bmi': bmi,
            'maternal_hemoglobin': hb,
            'mother_age': age,
            'mother_education': edu,
            'anc_visits_total': anc,
            'iron_pills': iron,
            'pregnancy_sickness': sickness,
            'risk_short_stature': risk_short,
            'risk_kek': risk_kek,
            'risk_anemia': risk_anemia,
            'risk_anemia_severe': risk_anemia_sev,
            'risk_age_extreme': risk_age,
            'risk_anc_inadequate': risk_anc,
            'risk_no_iron': risk_no_iron,
            'total_clinical_risk_score': clinical_risk_score,
            'height_bmi_ratio': height_bmi_ratio
        }

        df = pd.DataFrame([input_dict])
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]

        df_scaled = scaler.transform(df)
        prob = float(model.predict_proba(df_scaled)[0, 1])

        # Analisis Faktor Risiko Klinis Standar Kemenkes RI
        risk_factors = []
        recommendations = []

        if height < 150.0:
            risk_factors.append(f"Tinggi badan Ibu ({height:.0f} cm) di bawah 150 cm (faktor risiko antropometri maternal terhadap pertumbuhan janin)")
            recommendations.append("Optimalkan asupan protein hewani setiap kali makan untuk mendukung pertumbuhan panjang janin di dalam kandungan.")

        if bmi < 18.5:
            risk_factors.append(f"Indeks Massa Tubuh/IMT Ibu ({bmi:.1f} kg/m²) tergolong kurang/KEK (< 18.5)")
            recommendations.append("Tambah porsi makan utama dan selingan padat gizi (telur, ikan, susu, kacang-kacangan) untuk mengejar kenaikan berat badan kehamilan.")
        elif bmi >= 25.0:
            risk_factors.append(f"IMT Ibu ({bmi:.1f} kg/m²) tergolong berlebih ({bmi:.1f} >= 25.0)")
            recommendations.append("Kurangi makanan tinggi gula/tepung berlebih dan utamakan protein serta serat dari sayur dan buah segar.")

        if hb < 11.0:
            severity = "berat" if hb < 8.0 else ("sedang" if hb < 10.0 else "ringan")
            risk_factors.append(f"Kadar Hemoglobin ({hb:.1f} g/dL) di bawah batas normal ibu hamil (< 11.0 g/dL - indikasi anemia {severity})")
            recommendations.append("Wajib konsumsi Tablet Tambah Darah (TTD) setiap malam bersama air putih/jeruk (hindari teh/kopi) serta perbanyak lauk kaya zat besi heme (hati ayam, daging merah, ikan).")

        if age < 20 or age > 35:
            risk_factors.append(f"Usia kehamilan Ibu ({age:.0f} tahun) berada pada rentang usia berisiko (< 20 atau > 35 tahun)")
            recommendations.append("Lakukan pemantauan tekanan darah dan perkembangan janin (USG) secara lebih ketat bersama dokter/bidan.")

        if anc < 6:
            risk_factors.append(f"Frekuensi pemeriksaan kehamilan/ANC ({anc:.0f} kali) belum mencapai standar minimal Kemenkes RI (minimal 6 kali selama kehamilan)")
            recommendations.append("Rutin jadwalkan kunjungan ANC ke fasilitas kesehatan (minimal 1x di Trimester 1, 2x di Trimester 2, dan 3x di Trimester 3).")

        if iron == 0:
            risk_factors.append("Ibu belum rutin mengonsumsi Pil Besi / Tablet Tambah Darah (TTD)")
            recommendations.append("Segera mulai konsumsi rutin minimal 90 tablet tambah darah selama kehamilan untuk mencegah anemia maternal dan bayi berat lahir rendah (BBLR).")

        if sickness == 1:
            risk_factors.append("Terdapat riwayat sakit/infeksi selama masa kehamilan")
            recommendations.append("Konsultasikan riwayat keluhan sakit ke bidan/dokter agar infeksi tertangani dan penyerapan nutrisi ke janin tidak terhambat.")

        # Penentuan Kategori Risiko (Rendah / Sedang / Tinggi) yang selaras antara Model ML & Klinis
        is_above_threshold = prob >= float(threshold)
        if prob >= 0.50 or len(risk_factors) >= 3 or hb < 10.0:
            risk_label = "Tinggi"
            pred_class = 1
        elif len(risk_factors) == 0 and prob < 0.40:
            risk_label = "Rendah"
            pred_class = 0
        elif is_above_threshold or len(risk_factors) >= 1:
            risk_label = "Sedang"
            pred_class = 1
        else:
            risk_label = "Rendah"
            pred_class = 0

        if risk_factors:
            explanation_text = (
                f"Hasil analisis menunjukkan risiko stunting kategori {risk_label} (Skor ML: {prob*100:.1f}%). "
                f"Ditemukan {len(risk_factors)} faktor risiko pada data kesehatan Ibu: {'; '.join(risk_factors)}. "
                f"Rekomendasi Medis: {' '.join(recommendations)}"
            )
        else:
            explanation_text = (
                f"Kondisi kehamilan Ibu sangat optimal dengan risiko stunting {risk_label} (Skor ML: {prob*100:.1f}%). "
                f"IMT Ibu ({bmi:.1f} kg/m²), kadar Hemoglobin ({hb:.1f} g/dL), serta kepatuhan konsumsi TTD dan pemeriksaan ANC sudah sesuai standar Kemenkes RI. "
                f"Pertahankan pola makan gizi seimbang 'Isi Piringku' setiap hari."
            )

        return jsonify({
            "status": "success",
            "prediction_class": pred_class,
            "stunting_risk": risk_label,
            "probability": round(prob, 4),
            "probability_percentage": round(prob * 100, 1),
            "threshold_used": round(float(threshold), 4),
            "bmi": round(bmi, 2),
            "risk_factors": risk_factors,
            "recommendations": recommendations,
            "message": explanation_text,
            "explanation": explanation_text
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

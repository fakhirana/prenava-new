def collect_warnings(model_input: dict) -> list[str]:
    warnings = []

    if model_input["body_temp"] < 35.0 or model_input["body_temp"] > 39.0:
        warnings.append("Nilai body_temp berada di luar rentang umum. Mohon pastikan data yang dimasukkan sudah benar.")

    if model_input["heart_rate"] > 120:
        warnings.append("Nilai heart_rate cukup tinggi. Hasil rekomendasi sebaiknya ditinjau dengan hati-hati.")

    if model_input["bmi"] >= 35:
        warnings.append("Nilai BMI cukup tinggi. Beberapa aktivitas mungkin perlu perhatian tambahan.")

    return warnings
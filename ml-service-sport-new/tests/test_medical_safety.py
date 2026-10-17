from app.services.recommendation_service import recommendation_service


def test_hard_medical_safety_hypertension():
    """Menguji bahwa tensi >= 140/90 langsung memicu Zero False-Safety (Dilarang Olahraga)."""
    result = recommendation_service.recommend(
        risk_label="high",
        model_input={
            "systolic_bp": 145,
            "diastolic_bp": 95,
            "body_temp": 36.5,
            "heart_rate": 80,
            "bmi": 24,
        },
        context={
            "gestational_age_weeks": 20,
        },
    )

    assert result["is_contraindicated"] is True
    assert result["safety_status"] == "contraindicated"
    assert "KONTRAINDIKASI MEDIS MUTLAK" in result["safety_alert"]
    assert len(result["highly_recommended"]) == 0
    assert len(result["allowed_with_caution"]) == 0
    assert len(result["avoid"]) > 0


def test_hard_medical_safety_vaginal_bleeding():
    """Menguji bahwa keluhan pendarahan pervaginam langsung memblokir seluruh olahraga fisik."""
    result = recommendation_service.recommend(
        risk_label="low",
        model_input={
            "systolic_bp": 110,
            "diastolic_bp": 70,
            "body_temp": 36.5,
            "heart_rate": 75,
            "bmi": 22,
        },
        context={
            "gestational_age_weeks": 14,
            "vaginal_bleeding": True,
        },
    )

    assert result["is_contraindicated"] is True
    assert result["safety_status"] == "contraindicated"
    assert len(result["highly_recommended"]) == 0


def test_trimester_1_gentle_focus():
    """Menguji bahwa Trimester 1 (misal 8 minggu) memprioritaskan aktivitas gentle dan membatasi guncangan."""
    result = recommendation_service.recommend(
        risk_label="low",
        model_input={
            "systolic_bp": 110,
            "diastolic_bp": 70,
            "body_temp": 36.5,
            "heart_rate": 72,
            "bmi": 21,
        },
        context={
            "gestational_age_weeks": 8,
        },
    )

    assert result["is_contraindicated"] is False
    assert result["trimester"] == 1
    highly_codes = {item["code"] for item in result["highly_recommended"]}
    assert "morning_walk" in highly_codes or "evening_walk" in highly_codes
    assert "breathing_exercises" in highly_codes
    assert "chair_yoga" in highly_codes


def test_trimester_3_pelvic_and_birth_prep():
    """Menguji bahwa Trimester 3 (misal 34 minggu) mempromosikan latihan panggul dan persiapan persalinan."""
    result = recommendation_service.recommend(
        risk_label="low",
        model_input={
            "systolic_bp": 115,
            "diastolic_bp": 75,
            "body_temp": 36.5,
            "heart_rate": 78,
            "bmi": 23,
        },
        context={
            "gestational_age_weeks": 34,
        },
    )

    assert result["is_contraindicated"] is False
    assert result["trimester"] == 3
    highly_codes = {item["code"] for item in result["highly_recommended"]}
    assert "pelvic_tilt" in highly_codes
    assert "cat_cow_modified" in highly_codes
    assert "kegel" in highly_codes

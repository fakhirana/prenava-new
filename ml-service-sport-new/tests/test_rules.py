from app.services.recommendation_service import recommendation_service


def test_rules_return_all_sections():
    result = recommendation_service.recommend(
        risk_label="low",
        model_input={
            "bmi": 24,
            "previous_complications": False,
        },
        context={
            "gestational_age_weeks": 15,
            "placenta_position_restriction": False,
            "low_impact_preference": False,
            "water_access": True,
            "back_pain": False,
        },
    )

    assert "highly_recommended" in result
    assert "allowed_with_caution" in result
    assert "avoid" in result
    assert "all" in result
from app.utils.validation import collect_warnings


def test_collect_warnings_body_temp():
    warnings = collect_warnings({
        "body_temp": 34.5,
        "heart_rate": 70,
        "bmi": 24
    })
    assert len(warnings) >= 1
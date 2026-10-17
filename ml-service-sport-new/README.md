# prenava-ml-service

Production-oriented ML service for maternal risk prediction and exercise recommendation.

## Features
- Random Forest risk prediction (`low` / `high`)
- Rule-based exercise recommendation from `rules.yml`
- FastAPI inference service
- Dockerized deployment
- Structured JSON response

## Request Contract
POST `/predict`

```json
{
  "model_input": {
    "age": 25,
    "systolic_bp": 100,
    "diastolic_bp": 80,
    "blood_sugar": 80,
    "body_temp": 36.5,
    "bmi": 30.1,
    "previous_complications": false,
    "preexisting_diabetes": false,
    "gestational_diabetes": false,
    "mental_health_issue": false,
    "heart_rate": 70
  },
  "recommendation_context": {
    "gestational_age_weeks": 15,
    "placenta_position_restriction": false,
    "low_impact_preference": false,
    "water_access": true,
    "back_pain": true
  }
}
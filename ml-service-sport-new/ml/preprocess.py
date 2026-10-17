import pandas as pd


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "Age": "age",
        "Systolic BP": "systolic_bp",
        "Diastolic": "diastolic_bp",
        "BS": "blood_sugar",
        "Body Temp": "body_temp",
        "BMI": "bmi",
        "Previous Complications": "previous_complications",
        "Preexisting Diabetes": "preexisting_diabetes",
        "Gestational Diabetes": "gestational_diabetes",
        "Mental Health": "mental_health_issue",
        "Heart Rate": "heart_rate",
        "Risk Level": "risk_level",
    }
    return df.rename(columns=colmap)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = rename_columns(df)
    df = df.dropna(subset=["risk_level"])
    if "bmi" in df.columns:
        df.loc[df["bmi"] == 0, "bmi"] = pd.NA

    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    df["risk_level"] = df["risk_level"].astype(str).str.strip()
    df = df[df["risk_level"].isin(["Low", "High"])].copy()
    df["risk_level"] = df["risk_level"].map({"Low": 0, "High": 1})

    bool_like_cols = [
        "previous_complications",
        "preexisting_diabetes",
        "gestational_diabetes",
        "mental_health_issue",
    ]
    for col in bool_like_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df
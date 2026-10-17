import joblib

m = joblib.load('api/models/stunting_model.pkl')
booster = m.booster_
print("handle:", getattr(booster, 'handle', None))
print("booster __dict__ keys:", list(booster.__dict__.keys()))
if 'model_str' in booster.__dict__:
    print("model_str len:", len(booster.model_str))
if 'model_from_string' in booster.__dict__:
    print("model_from_string:", booster.model_from_string)

import joblib

model = joblib.load('api/models/stunting_model.pkl')
print("Model attributes:")
for attr in dir(model):
    if not attr.startswith('__'):
        print(attr, getattr(model, attr, 'N/A') if not callable(getattr(model, attr)) else 'method')

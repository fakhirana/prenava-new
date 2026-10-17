import requests
from PIL import Image
import numpy as np

# Create a sample red image
img_array = np.full((224, 224, 3), [200, 50, 50], dtype=np.uint8)
img = Image.fromarray(img_array)
img.save("test_dummy.jpg")

# Test the endpoint
url = "http://localhost:9001/predict"
with open("test_dummy.jpg", "rb") as f:
    files = {"file": ("test_dummy.jpg", f, "image/jpeg")}
    try:
        response = requests.post(url, files=files)
        print("Status Code:", response.status_code)
        print("Response JSON:\n", response.json())
    except Exception as e:
        print("Error connecting to API:", str(e))

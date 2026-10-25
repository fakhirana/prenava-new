import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)

def create_dummy_image_bytes(width=300, height=300, color="green") -> io.BytesIO:
    """Helper membuat citra virtual untuk pengujian API."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["port"] == 9002

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert data["port"] == 9002

def test_predict_food_invalid_file_type():
    file_payload = {"file": ("test.txt", io.BytesIO(b"Hello text file"), "text/plain")}
    response = client.post("/predict-food", files=file_payload)
    assert response.status_code == 400
    assert "tidak didukung" in response.json()["detail"]

def test_predict_food_success():
    img_buf = create_dummy_image_bytes()
    file_payload = {"file": ("plate.jpg", img_buf, "image/jpeg")}
    response = client.post("/predict-food", files=file_payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "success"
    assert "data" in json_data
    assert "detected_items" in json_data["data"]
    assert "isi_piringku_analysis" in json_data["data"]
    assert "score" in json_data["data"]["isi_piringku_analysis"]
    assert "breakdown" in json_data["data"]["isi_piringku_analysis"]

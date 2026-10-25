import pytest
from app.services.food_analyzer import FoodAnalyzer
from app.schemas import DetectedItem

def test_relative_areas_calculation():
    """Menguji kebenaran perhitungan luas relatif bounding box."""
    raw_detections = [
        {"class": "karbohidrat", "name": "Nasi", "box": [0, 0, 100, 100]},     # Luas = 10,000
        {"class": "sayuran", "name": "Bayam", "box": [0, 0, 100, 100]},        # Luas = 10,000
        {"class": "protein_hewani", "name": "Ikan", "box": [0, 0, 50, 100]},   # Luas = 5,000
        {"class": "buah", "name": "Pepaya", "box": [0, 0, 50, 100]}           # Luas = 5,000
    ]
    # Total area = 30,000. Karbo = 33.3%, Sayur = 33.3%, Hewani = 16.7%, Buah = 16.7%
    items = FoodAnalyzer.calculate_relative_areas(raw_detections)
    assert len(items) == 4
    total_pct = sum(item.relative_area_percentage for item in items)
    assert abs(total_pct - 100.0) < 1.0  # Toleransi pembulatan

def test_balanced_plate_evaluation():
    """Menguji piring ideal seimbang sesuai standar Kemenkes RI."""
    items = [
        DetectedItem(class_name="karbohidrat", name="Nasi Putih", confidence=0.95, box=[10, 10, 110, 110], relative_area_percentage=33.5),
        DetectedItem(class_name="sayuran", name="Sayur Bayam", confidence=0.92, box=[10, 120, 110, 220], relative_area_percentage=33.0),
        DetectedItem(class_name="protein_hewani", name="Ikan Kembung", confidence=0.90, box=[120, 10, 170, 110], relative_area_percentage=17.0),
        DetectedItem(class_name="buah", name="Potongan Pepaya", confidence=0.91, box=[120, 120, 170, 220], relative_area_percentage=16.5)
    ]
    analysis = FoodAnalyzer.evaluate_isi_piringku(items)
    assert analysis.has_iron_protein is True
    assert analysis.status == "SEIMBANG"
    assert analysis.score >= 85
    assert "Kaya Zat Besi" in analysis.breakdown.lauk_protein

def test_iron_deficiency_warning():
    """
    Menguji proteksi klinis saat piring tidak mengandung protein hewani.
    Harus memunculkan peringatan risiko anemia dan penalti skor.
    """
    items = [
        DetectedItem(class_name="karbohidrat", name="Nasi Putih", confidence=0.95, box=[10, 10, 110, 110], relative_area_percentage=40.0),
        DetectedItem(class_name="sayuran", name="Sayur Kangkung", confidence=0.92, box=[10, 120, 110, 220], relative_area_percentage=35.0),
        DetectedItem(class_name="protein_nabati", name="Tempe Goreng", confidence=0.88, box=[120, 10, 170, 110], relative_area_percentage=15.0),
        DetectedItem(class_name="buah", name="Pisang", confidence=0.91, box=[120, 120, 170, 220], relative_area_percentage=10.0)
    ]
    analysis = FoodAnalyzer.evaluate_isi_piringku(items)
    assert analysis.has_iron_protein is False
    assert analysis.score <= 70  # Dibatasi karena defisiensi zat besi hewani
    assert "PERINGATAN GIZI" in analysis.clinical_feedback
    assert "Kurang Zat Besi Hewani" in analysis.breakdown.lauk_protein

def test_empty_plate():
    """Menguji penanganan ketika tidak ada objek terdeteksi."""
    analysis = FoodAnalyzer.evaluate_isi_piringku([])
    assert analysis.score == 0
    assert analysis.status == "TIDAK TERDETEKSI"
    assert analysis.has_iron_protein is False

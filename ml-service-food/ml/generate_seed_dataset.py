import os
import random
from pathlib import Path
from PIL import Image, ImageDraw

def create_synthetic_plate(image_path: Path, label_path: Path, include_animal_iron: bool = True):
    """
    Membuat citra sintetis piring makanan nusantara (640x640) beserta anotasi label YOLOv8
    untuk seed dataset dan pengujian pipeline training awal.
    """
    img_size = 640
    # Latar belakang meja makan
    table_colors = [(220, 210, 195), (235, 230, 220), (210, 190, 170)]
    img = Image.new("RGB", (img_size, img_size), color=random.choice(table_colors))
    draw = ImageDraw.Draw(img)

    # Gambar piring bulat (tampak atas)
    margin = 40
    plate_box = [margin, margin, img_size - margin, img_size - margin]
    draw.ellipse(plate_box, fill=(250, 250, 252), outline=(190, 190, 195), width=4)
    # Lingkaran dalam piring
    inner_margin = 70
    draw.ellipse([inner_margin, inner_margin, img_size - inner_margin, img_size - inner_margin], 
                 fill=(245, 245, 248), outline=(225, 225, 230), width=2)

    boxes_yolo = []  # format: [class_id, x_center, y_center, width, height]

    # 1. Karbohidrat (Nasi Putih / Nasi Merah) - Kuadran Kiri Atas
    x1, y1, x2, y2 = 120, 130, 310, 320
    draw.ellipse([x1, y1, x2, y2], fill=(240, 240, 245), outline=(210, 210, 215), width=2)
    # Tekstur butir nasi
    for _ in range(35):
        rx, ry = random.randint(x1 + 15, x2 - 15), random.randint(y1 + 15, y2 - 15)
        draw.ellipse([rx, ry, rx + 4, ry + 2], fill=(215, 215, 220))
    xc, yc = (x1 + x2) / (2 * img_size), (y1 + y2) / (2 * img_size)
    w, h = (x2 - x1) / img_size, (y2 - y1) / img_size
    boxes_yolo.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    # 2. Sayuran (Sayur Bayam / Kangkung) - Kuadran Kiri Bawah
    x1, y1, x2, y2 = 120, 340, 310, 520
    draw.ellipse([x1, y1, x2, y2], fill=(45, 115, 45), outline=(30, 85, 30), width=2)
    for _ in range(40):
        rx, ry = random.randint(x1 + 10, x2 - 10), random.randint(y1 + 10, y2 - 10)
        draw.ellipse([rx, ry, rx + 8, ry + 5], fill=(35, 140, 35))
    xc, yc = (x1 + x2) / (2 * img_size), (y1 + y2) / (2 * img_size)
    w, h = (x2 - x1) / img_size, (y2 - y1) / img_size
    boxes_yolo.append(f"3 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    # 3. Lauk Protein: Hewani vs Nabati - Kuadran Kanan Atas
    if include_animal_iron:
        # Kelas 1: Protein Hewani (Ikan / Ayam Goreng - Kaya Zat Besi)
        x1, y1, x2, y2 = 340, 130, 520, 310
        draw.rectangle([x1, y1, x2, y2], fill=(160, 90, 40), outline=(120, 65, 25), width=2)
        draw.ellipse([x1 + 15, y1 + 15, x2 - 15, y2 - 15], fill=(185, 110, 50))
        xc, yc = (x1 + x2) / (2 * img_size), (y1 + y2) / (2 * img_size)
        w, h = (x2 - x1) / img_size, (y2 - y1) / img_size
        boxes_yolo.append(f"1 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    else:
        # Kelas 2: Protein Nabati (Tempe / Tahu Goreng)
        x1, y1, x2, y2 = 340, 130, 520, 310
        draw.rectangle([x1, y1, x2, y2], fill=(205, 160, 60), outline=(170, 125, 40), width=2)
        for _ in range(5):
            rx = random.randint(x1 + 5, x2 - 30)
            ry = random.randint(y1 + 5, y2 - 30)
            draw.rectangle([rx, ry, rx + 25, ry + 25], fill=(225, 180, 80), outline=(150, 110, 30))
        xc, yc = (x1 + x2) / (2 * img_size), (y1 + y2) / (2 * img_size)
        w, h = (x2 - x1) / img_size, (y2 - y1) / img_size
        boxes_yolo.append(f"2 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    # 4. Buah (Pepaya / Jeruk / Pisang) - Kuadran Kanan Bawah
    x1, y1, x2, y2 = 350, 340, 510, 500
    draw.ellipse([x1, y1, x2, y2], fill=(245, 120, 30), outline=(200, 90, 20), width=2)
    # Irisan buah
    draw.ellipse([x1 + 20, y1 + 20, x2 - 20, y2 - 20], fill=(255, 140, 45))
    xc, yc = (x1 + x2) / (2 * img_size), (y1 + y2) / (2 * img_size)
    w, h = (x2 - x1) / img_size, (y2 - y1) / img_size
    boxes_yolo.append(f"4 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    # Simpan Gambar dan Label
    img.save(image_path, "JPEG", quality=90)
    with open(label_path, "w") as f:
        f.write("\n".join(boxes_yolo) + "\n")

def generate_seed_dataset():
    """Membuat seed dataset valid (Train: 14, Val: 4, Test: 2) untuk verifikasi pipeline."""
    base_dir = Path(__file__).resolve().parent.parent / "dataset_isi_piringku"
    splits = {
        "train": 14,
        "valid": 4,
        "test": 2
    }

    print("=== Membuat Seed Dataset YOLOv8 'Isi Piringku' ===")
    total_generated = 0

    for split, count in splits.items():
        img_dir = base_dir / split / "images"
        lbl_dir = base_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, count + 1):
            filename = f"plate_seed_{split}_{i:03d}"
            img_path = img_dir / f"{filename}.jpg"
            lbl_path = lbl_dir / f"{filename}.txt"

            # 75% piring memiliki zat besi hewani, 25% tanpa hewani untuk variasi uji defisiensi
            has_animal = (i % 4 != 0)
            create_synthetic_plate(img_path, lbl_path, include_animal_iron=has_animal)
            total_generated += 1

        print(f"[{split.upper()}] Dibuat {count} citra di {img_dir}")

    print(f"\n[SUKSES] Total {total_generated} citra dan anotasi seed berhasil dibuat!")

if __name__ == "__main__":
    generate_seed_dataset()

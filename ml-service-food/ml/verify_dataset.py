import os
import sys
import yaml
from pathlib import Path
from collections import defaultdict

# Enforce UTF-8 for Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def verify_dataset(yaml_path: str = "dataset_isi_piringku/data.yaml"):
    """
    Memvalidasi integritas dataset YOLOv8 'Isi Piringku'
    dan menghasilkan statistik distribusi kelas untuk laporan Bab 3 / Bab 4 Skripsi.
    """
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        print(f"[ERROR] File konfigurasi {yaml_path} tidak ditemukan!")
        return False

    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    base_dir = yaml_file.parent
    classes = config.get("names", {})
    if isinstance(classes, list):
        class_map = {i: name for i, name in enumerate(classes)}
    else:
        class_map = classes

    print(f"============================================================")
    print(f"  VALIDASI INTEGRITAS DATASET PRENAVA 'ISI PIRINGKU'        ")
    print(f"============================================================")
    print(f"Target Kelas ({len(class_map)}): {class_map}\n")

    splits = ["train", "valid", "test"]
    overall_class_counts = defaultdict(int)
    split_stats = {}
    is_valid = True

    for split in splits:
        img_dir = base_dir / split / "images"
        lbl_dir = base_dir / split / "labels"

        if not img_dir.exists():
            print(f"[WARNING] Direktori {img_dir} tidak ditemukan.")
            continue

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
        labels = list(lbl_dir.glob("*.txt"))

        # Verifikasi pasangan gambar dan label
        missing_labels = []
        for img in images:
            expected_lbl = lbl_dir / f"{img.stem}.txt"
            if not expected_lbl.exists():
                missing_labels.append(img.name)

        if missing_labels:
            print(f"[ERROR] Ditemukan {len(missing_labels)} gambar tanpa label di split {split}: {missing_labels[:5]}")
            is_valid = False

        # Hitung distribusi bounding box per kelas
        class_counts = defaultdict(int)
        iron_plates_count = 0

        for lbl_path in labels:
            with open(lbl_path, "r") as f:
                lines = f.readlines()

            plate_has_iron = False
            for line_no, line in enumerate(lines, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    print(f"[ERROR] Format tidak valid di {lbl_path.name} baris {line_no}: '{line.strip()}'")
                    is_valid = False
                    continue

                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])

                if cls_id not in class_map:
                    print(f"[ERROR] Class ID {cls_id} tidak terdaftar di {lbl_path.name}")
                    is_valid = False
                else:
                    class_counts[cls_id] += 1
                    overall_class_counts[cls_id] += 1
                    if cls_id == 1:  # 1 = protein_hewani (Heme-Iron)
                        plate_has_iron = True

                # Validasi batas normalisasi 0.0 s/d 1.0
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    print(f"[ERROR] Koordinat out-of-bounds di {lbl_path.name} baris {line_no}")
                    is_valid = False

            if plate_has_iron:
                iron_plates_count += 1

        split_stats[split] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": sum(class_counts.values()),
            "iron_plates": iron_plates_count,
            "class_counts": class_counts
        }

    # Tampilkan Ringkasan Statistik
    total_images = sum(s["images"] for s in split_stats.values())
    total_boxes = sum(s["boxes"] for s in split_stats.values())

    print(f"📊 STATISTIK PEMBAGIAN SPLIT DATASET:")
    print(f"| Split | Jumlah Gambar | Proporsi | Total Bounding Box | Piring Kaya Zat Besi |")
    print(f"| :--- | :---: | :---: | :---: | :---: |")
    for split, data in split_stats.items():
        pct = (data["images"] / total_images * 100) if total_images > 0 else 0
        iron_pct = (data["iron_plates"] / data["images"] * 100) if data["images"] > 0 else 0
        print(f"| **{split.capitalize()}** | {data['images']} citra | {pct:.1f}% | {data['boxes']} box | {data['iron_plates']} ({iron_pct:.1f}%) |")
    print(f"| **TOTAL** | **{total_images} citra** | **100%** | **{total_boxes} box** | - |\n")

    print(f"🥗 DISTRIBUSI KELAS MAKANAN (UNTUK BAB 3 & 4 SKRIPSI):")
    print(f"| ID | Nama Kelas | Train Box | Valid Box | Test Box | Total Box | Persentase |")
    print(f"| :-: | :--- | :-: | :-: | :-: | :-: | :-: |")
    for cls_id in sorted(class_map.keys()):
        c_name = class_map[cls_id]
        tr = split_stats.get("train", {}).get("class_counts", {}).get(cls_id, 0)
        va = split_stats.get("valid", {}).get("class_counts", {}).get(cls_id, 0)
        te = split_stats.get("test", {}).get("class_counts", {}).get(cls_id, 0)
        tot = overall_class_counts[cls_id]
        pct = (tot / total_boxes * 100) if total_boxes > 0 else 0
        print(f"| {cls_id} | `{c_name}` | {tr} | {va} | {te} | **{tot}** | {pct:.1f}% |")

    if is_valid and total_images > 0:
        print(f"\n✅ [STATUS: VALID] Dataset memenuhi seluruh spesifikasi format YOLOv8!")
        print(f"Siap digunakan untuk training transfer learning (`python ml/train.py`).")
        return True
    else:
        print(f"\n❌ [STATUS: TIDAK VALID] Terdapat anomali format data di atas yang perlu diperbaiki.")
        return False

if __name__ == "__main__":
    verify_dataset()

import os
import shutil
import argparse
from pathlib import Path
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Pelatihan Model YOLOv8-Nano untuk Deteksi Objek Makanan 'Isi Piringku'")
    parser.add_argument("--data", type=str, default="dataset_isi_piringku/data.yaml", help="Path ke file konfigurasi data.yaml Roboflow")
    parser.add_argument("--epochs", type=int, default=75, help="Jumlah epoch pelatihan (rekomendasi: 75)")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolusi gambar (default: 640x640)")
    parser.add_argument("--batch", type=int, default=16, help="Ukuran batch")
    parser.add_argument("--device", type=str, default=None, help="Device ID ('0' untuk GPU, 'cpu' untuk CPU, default: otomatis)")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Pretrained weights awal (default: yolov8n.pt)")
    return parser.parse_args()

def main():
    args = parse_args()
    from ultralytics import YOLO

    # 1. Deteksi device otomatis
    if args.device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"=== Memulai Pelatihan YOLOv8-Nano: Prenava Food Scanner ===")
    print(f"Dataset      : {args.data}")
    print(f"Epochs       : {args.epochs}")
    print(f"Image Size   : {args.imgsz}")
    print(f"Batch Size   : {args.batch}")
    print(f"Device       : {device} ({'GPU CUDA' if device != 'cpu' else 'CPU'})")
    print(f"Base Weights : {args.weights}")

    # 2. Inisialisasi Pretrained Model YOLOv8-Nano
    model = YOLO(args.weights)

    # 3. Jalankan Pelatihan
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name="prenava_food_yolov8n",
        device=device,
        save=True,
        plots=True
    )

    # 4. Salin bobot terbaik ke direktori produksi ml-service-food/models/best.pt
    run_dir = Path(results.save_dir)
    source_best = run_dir / "weights" / "best.pt"
    dest_models_dir = Path(__file__).resolve().parent.parent / "models"
    dest_models_dir.mkdir(parents=True, exist_ok=True)
    dest_best = dest_models_dir / "best.pt"

    if source_best.exists():
        shutil.copy2(source_best, dest_best)
        print(f"\n[SUKSES] Bobot terbaik model berhasil diekspor ke: {dest_best}")
    else:
        print(f"\n[PERINGATAN] File {source_best} tidak ditemukan. Periksa direktori run: {run_dir}")

if __name__ == "__main__":
    main()

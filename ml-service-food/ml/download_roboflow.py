import os
import sys
import shutil
import zipfile
import argparse
from pathlib import Path

# Enforce UTF-8 for Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="Download & Setup Dataset Isi Piringku dari Roboflow atau Zip")
    parser.add_argument("--api-key", type=str, default=None, help="Roboflow API Key")
    parser.add_argument("--workspace", type=str, default=None, help="Roboflow Workspace name")
    parser.add_argument("--project", type=str, default=None, help="Roboflow Project name")
    parser.add_argument("--version", type=int, default=1, help="Roboflow Project Version")
    parser.add_argument("--zip-file", type=str, default=None, help="Path ke file zip dataset hasil ekspor Roboflow/Kaggle")
    return parser.parse_args()

def extract_and_organize(source_dir: Path, target_dir: Path):
    """Menyelaraskan struktur dataset hasil unduhan ke format dataset_isi_piringku."""
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Cari folder train, valid, test dan data.yaml
    for split in ["train", "valid", "val", "test"]:
        sub = source_dir / split
        if sub.exists() and sub.is_dir():
            dest_split = "valid" if split == "val" else split
            dest = target_dir / dest_split
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(sub, dest)
            print(f"[OK] Split '{dest_split}' berhasil disalin ke {dest}")

    # Copy data.yaml
    found_yaml = list(source_dir.glob("*.yaml")) + list(source_dir.glob("*.yml"))
    if found_yaml:
        shutil.copy2(found_yaml[0], target_dir / "data.yaml")
        print(f"[OK] data.yaml berhasil diperbarui dari {found_yaml[0].name}")

def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent.parent
    target_dataset_dir = base_dir / "dataset_isi_piringku"

    print("=== Tool Akuisisi Dataset 'Isi Piringku' Prenava ===")

    # 1. Unduh via file ZIP lokal
    if args.zip_file:
        zip_path = Path(args.zip_file)
        if not zip_path.exists():
            print(f"[ERROR] File zip {zip_path} tidak ditemukan!")
            return

        temp_extract = base_dir / "temp_dataset_extracted"
        print(f"Mengekstrak {zip_path} ke {temp_extract}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract)

        extract_and_organize(temp_extract, target_dataset_dir)
        shutil.rmtree(temp_extract, ignore_errors=True)
        print(f"\n[SUKSES] Dataset berhasil dipasang di {target_dataset_dir}")
        return

    # 2. Unduh via Roboflow API
    if args.api_key and args.workspace and args.project:
        try:
            from roboflow import Roboflow
            print(f"Menghubungkan ke Roboflow Universe / Workspace: {args.workspace}...")
            rf = Roboflow(api_key=args.api_key)
            project = rf.workspace(args.workspace).project(args.project)
            version = project.version(args.version)
            download_path = version.download("yolov8")
            
            source_dir = Path(download_path.location)
            extract_and_organize(source_dir, target_dataset_dir)
            print(f"\n[SUKSES] Dataset Roboflow versi {args.version} berhasil diunduh dan dipasang!")
        except ImportError:
            print("[INFO] Paket 'roboflow' belum terpasang. Jalankan: pip install roboflow")
        except Exception as e:
            print(f"[ERROR] Gagal mengunduh dari Roboflow: {e}")
        return

    print("Penggunaan:")
    print("1. Jika memiliki file zip hasil download dari Roboflow / Kaggle:")
    print("   python ml/download_roboflow.py --zip-file path/to/dataset.zip")
    print("2. Jika menggunakan Roboflow API:")
    print("   python ml/download_roboflow.py --api-key <KEY> --workspace <WS> --project <PRJ> --version 1")

if __name__ == "__main__":
    main()

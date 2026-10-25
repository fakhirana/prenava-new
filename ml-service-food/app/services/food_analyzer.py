from typing import List, Dict, Any, Tuple
from app.schemas import DetectedItem, IsiPiringkuAnalysis, IsiPiringkuBreakdown

class FoodAnalyzer:
    """
    Kalkulator dan Evaluator Proporsi Gizi 'Isi Piringku' Standar Kemenkes RI
    Khusus Ibu Hamil untuk Pencegahan Anemia Maternal dan Stunting Bayi.
    """

    TARGET_KARBO: float = 33.33   # 1/3 piring
    TARGET_SAYUR: float = 33.33   # 1/3 piring
    TARGET_LAUK: float = 16.67    # 1/6 piring
    TARGET_BUAH: float = 16.67    # 1/6 piring

    @classmethod
    def calculate_relative_areas(cls, raw_detections: List[Dict[str, Any]]) -> List[DetectedItem]:
        """
        Menghitung luas relatif bounding box setiap objek terhadap total luas area makanan.
        Area_i = (x2 - x1) * (y2 - y1)
        """
        if not raw_detections:
            return []

        # Hitung luas absolut tiap bounding box
        areas = []
        for det in raw_detections:
            box = det["box"]
            width = max(0, box[2] - box[0])
            height = max(0, box[3] - box[1])
            area = width * height
            areas.append(area)

        total_area = sum(areas)
        if total_area <= 0:
            total_area = 1.0  # Menghindari ZeroDivisionError

        detected_items: List[DetectedItem] = []
        for det, area in zip(raw_detections, areas):
            rel_pct = round((area / total_area) * 100.0, 1)
            detected_items.append(
                DetectedItem(
                    class_name=det["class"],
                    name=det.get("name", det["class"].replace("_", " ").title()),
                    confidence=round(float(det.get("confidence", 0.90)), 2),
                    box=det["box"],
                    relative_area_percentage=rel_pct
                )
            )

        return detected_items

    @classmethod
    def evaluate_isi_piringku(cls, detected_items: List[DetectedItem]) -> IsiPiringkuAnalysis:
        """
        Mengevaluasi kesesuaian proporsi terhadap standar Kemenkes 'Isi Piringku Ibu Hamil'
        dan memverifikasi kecukupan zat besi (Heme-Iron).
        """
        if not detected_items:
            return IsiPiringkuAnalysis(
                score=0,
                status="TIDAK TERDETEKSI",
                has_iron_protein=False,
                breakdown=IsiPiringkuBreakdown(
                    karbohidrat="0.0% (Standar: 33%) - Tidak Ada",
                    sayuran="0.0% (Standar: 33%) - Tidak Ada",
                    lauk_protein="0.0% (Standar: 17%) - Tidak Ada",
                    buah="0.0% (Standar: 17%) - Tidak Ada"
                ),
                clinical_feedback="Tidak ditemukan objek makanan pada piring. Pastikan pencahayaan cukup dan foto diambil tegak lurus dari atas piring."
            )

        # 1. Agregasi persentase luas per kelompok makanan
        cat_pct: Dict[str, float] = {
            "karbohidrat": 0.0,
            "sayuran": 0.0,
            "protein_hewani": 0.0,
            "protein_nabati": 0.0,
            "buah": 0.0
        }

        specific_names: Dict[str, List[str]] = {k: [] for k in cat_pct.keys()}

        for item in detected_items:
            cls_name = item.class_name.lower().strip()
            if cls_name in cat_pct:
                cat_pct[cls_name] += item.relative_area_percentage
                specific_names[cls_name].append(item.name)

        karbo_pct = round(cat_pct["karbohidrat"], 1)
        sayur_pct = round(cat_pct["sayuran"], 1)
        hewani_pct = round(cat_pct["protein_hewani"], 1)
        nabati_pct = round(cat_pct["protein_nabati"], 1)
        buah_pct = round(cat_pct["buah"], 1)
        lauk_pct = round(hewani_pct + nabati_pct, 1)

        # 2. Hard Medical Safety: Cek keberadaan zat besi (Heme-Iron dari protein hewani)
        has_iron_protein = (hewani_pct > 0.0)

        # 3. Penilaian deviasi terhadap standar Kemenkes
        def get_adherence_status(actual: float, target: float, category_type: str = "normal") -> Tuple[str, float]:
            diff = abs(actual - target)
            penalty = diff * 2.2
            sub_score = max(0.0, 100.0 - penalty)

            if diff <= 6.0:
                status_txt = "Sesuai"
            elif actual > target:
                status_txt = "Berlebih"
            elif actual > 0.0:
                status_txt = "Cukup"
            else:
                status_txt = "Kurang / Tidak Ada"

            return status_txt, sub_score

        status_karbo, score_karbo = get_adherence_status(karbo_pct, cls.TARGET_KARBO)
        status_sayur, score_sayur = get_adherence_status(sayur_pct, cls.TARGET_SAYUR)
        status_lauk, score_lauk = get_adherence_status(lauk_pct, cls.TARGET_LAUK)
        status_buah, score_buah = get_adherence_status(buah_pct, cls.TARGET_BUAH)

        # Base Score (rata-rata kepatuhan 4 kuadran)
        raw_score = (score_karbo + score_sayur + score_lauk + score_buah) / 4.0

        # Adjust score berdasarkan defisiensi zat besi
        if not has_iron_protein:
            # Penalti defisiensi zat besi hewani untuk pencegahan anemia maternal
            raw_score = min(raw_score, 68.0)
            status_lauk_desc = f"{lauk_pct}% (Standar: 17%) - {status_lauk} (Kurang Zat Besi Hewani)"
        else:
            status_lauk_desc = f"{lauk_pct}% (Standar: 17%) - {status_lauk} (Kaya Zat Besi)"

        final_score = int(max(10, min(100, round(raw_score))))

        # Status Keseluruhan
        if final_score >= 80 and has_iron_protein:
            overall_status = "SEIMBANG"
        elif final_score >= 60:
            overall_status = "KURANG SEIMBANG"
        else:
            overall_status = "PERLU PERBAIKAN"

        # 4. Breakdown Kemenkes
        breakdown = IsiPiringkuBreakdown(
            karbohidrat=f"{karbo_pct}% (Standar: 33%) - {status_karbo}",
            sayuran=f"{sayur_pct}% (Standar: 33%) - {status_sayur}",
            lauk_protein=status_lauk_desc,
            buah=f"{buah_pct}% (Standar: 17%) - {status_buah}"
        )

        # 5. Tailored Clinical Feedback
        feedback_parts = []
        if has_iron_protein:
            hewani_items = ", ".join(specific_names["protein_hewani"]) if specific_names["protein_hewani"] else "lauk hewani"
            feedback_parts.append(
                f"Menu makanan sangat baik! Terdeteksi {hewani_items} sebagai sumber protein hewani kaya zat besi (heme-iron) yang efektif mencegah anemia maternal."
            )
        else:
            feedback_parts.append(
                "⚠️ PERINGATAN GIZI: Piring Anda kekurangan Protein Hewani (Zat Besi Heme). Ibu hamil sangat membutuhkan asupan telur, ikan, daging sapi, ayam, atau hati ayam untuk mencegah risiko anemia dan mendukung pencegahan stunting pada janin."
            )

        if sayur_pct < 20.0:
            feedback_parts.append("Porsi sayuran hijau masih kurang dari rekomendasi 1/3 piring. Tambahkan bayam/daun kelor/brokoli untuk menambah asam folat alami.")
        elif karbo_pct > 45.0:
            feedback_parts.append("Porsi makanan pokok (karbohidrat) cukup dominan. Seimbangkan dengan memperbanyak porsi sayur dan buah.")

        if buah_pct < 10.0:
            feedback_parts.append("Tambahkan buah segar kaya Vitamin C (seperti jeruk/pepaya) untuk mengoptimalkan penyerapan zat besi.")

        clinical_feedback = " ".join(feedback_parts)

        return IsiPiringkuAnalysis(
            score=final_score,
            status=overall_status,
            has_iron_protein=has_iron_protein,
            breakdown=breakdown,
            clinical_feedback=clinical_feedback
        )

import logging
import shutil
import tempfile
from pathlib import Path

import httpx
import yaml

from app.config import settings
from app.exceptions import RuleEngineError

logger = logging.getLogger(__name__)

RISK_LEVELS = ("highly_recommended", "allowed_with_caution", "avoid")


class RecommendationService:
    def __init__(self) -> None:
        if not self._sync_from_backend():
            self._load_from_local_file()

    def _load_from_local_file(self) -> None:
        with open(settings.rules_path, "r", encoding="utf-8") as f:
            self.rulebook = yaml.safe_load(f)
        self._rebuild_lookup()
        logger.warning(
            "Memuat rulebook dari file lokal (sync ke backend gagal/tidak tersedia)."
        )

    def _rebuild_lookup(self) -> None:
        self.activity_lookup = {
            item["code"]: item for item in self.rulebook["activities"]
        }

    def _sync_from_backend(self) -> bool:
        if not settings.backend_sync_url:
            logger.warning("BACKEND_SYNC_URL belum diset, skip sync ke backend.")
            return False

        try:
            response = httpx.get(
                settings.backend_sync_url,
                timeout=settings.backend_sync_timeout_seconds,
            )
            response.raise_for_status()
            rows = response.json()["data"]
        except Exception as e:
            logger.error(f"Gagal sync rules dari backend: {e}")
            return False

        new_rulebook = self._build_rulebook_from_rows(rows)
        self._persist_to_disk(new_rulebook)

        self.rulebook = new_rulebook
        self._rebuild_lookup()
        logger.info(f"Rulebook berhasil di-sync dari backend ({len(rows)} aktivitas).")
        return True

    @staticmethod
    def _build_rulebook_from_rows(rows: list[dict]) -> dict:
        activities = []
        rules = {
            "high_risk": {level: [] for level in RISK_LEVELS},
            "low_risk": {level: [] for level in RISK_LEVELS},
        }

        for row in rows:
            code = row["code"]

            activities.append({
                "code": code,
                "name": row["name"],
                "category": row.get("category") or "uncategorized",
            })

            risk_high = row.get("risk_level_high")
            if risk_high in RISK_LEVELS:
                rules["high_risk"][risk_high].append(code)

            risk_low = row.get("risk_level_low")
            if risk_low in RISK_LEVELS:
                rules["low_risk"][risk_low].append(code)

        return {"activities": activities, "rules": rules}

    @staticmethod
    def _persist_to_disk(rulebook: dict) -> None:
        try:
            target = Path(settings.rules_path)
            target.parent.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=target.parent, delete=False, suffix=".tmp"
            ) as tmp:
                yaml.safe_dump(rulebook, tmp, allow_unicode=True, sort_keys=False)
                tmp_path = tmp.name

            shutil.move(tmp_path, target)
        except Exception as e:
            logger.error(f"Gagal menulis cache rules.yml ke disk: {e}")

    def sync_now(self) -> bool:
        return self._sync_from_backend()

    def _build_items(self, codes: list[str], level: str) -> list[dict]:
        items = []
        for code in codes:
            activity = self.activity_lookup.get(code)
            if not activity:
                continue
            items.append(
                {
                    "code": activity["code"],
                    "name": activity["name"],
                    "category": activity["category"],
                    "recommendation_level": level,
                }
            )
        return items

    @staticmethod
    def _check_hard_medical_contraindications(model_input: dict, context: dict) -> tuple[bool, list[str]]:
        reasons = []

        # 1. Hipertensi Gestasional / Preeklampsia (ACOG: SBP >= 140 atau DBP >= 90)
        sbp = float(model_input.get("systolic_bp", 0))
        dbp = float(model_input.get("diastolic_bp", 0))
        if sbp >= 140 or dbp >= 90:
            reasons.append(f"Tekanan darah Ibu ({sbp:.0f}/{dbp:.0f} mmHg) melebihi batas aman (>= 140/90 mmHg - risiko preeklampsia/hipertensi kehamilan)")

        # 2. Pendarahan Pervaginam (Vaginal Bleeding)
        if context.get("vaginal_bleeding", False):
            reasons.append("Terdapat keluhan pendarahan / flek jalan lahir aktif (pervaginam)")

        # 3. Kontraksi Prematur (Premature Labor / Contractions)
        if context.get("premature_contractions", False):
            reasons.append("Terdapat kontraksi rahim prematur sebelum waktu persalinan")

        # 4. Ketuban Pecah Dini (Amniotic Fluid Leak)
        if context.get("amniotic_fluid_leak", False):
            reasons.append("Terdapat rembesan / kebocoran cairan ketuban")

        # 5. Gejala Kardiorespirasi Berat
        if context.get("chest_pain_or_dyspnea", False):
            reasons.append("Terdapat keluhan nyeri dada atau sesak napas berat saat istirahat")

        # 6. Pusing Hebat / Pingsan
        if context.get("severe_dizziness", False):
            reasons.append("Terdapat keluhan pusing berputar hebat atau riwayat hilang kesadaran")

        # 7. Demam Tinggi Maternal (Risiko Hipertermia Janin)
        temp = float(model_input.get("body_temp", 36.5))
        is_fever = (temp >= 101.3) if temp > 50 else (temp >= 38.5)
        if is_fever:
            reasons.append(f"Suhu tubuh Ibu ({temp}) tergolong demam tinggi (risiko hipertermia janin)")

        # 8. Takikardia Berat saat Istirahat
        hr = float(model_input.get("heart_rate", 75))
        if hr >= 130:
            reasons.append(f"Denyut nadi Ibu ({hr:.0f} bpm) sangat tinggi (takikardia berat)")

        is_danger = len(reasons) > 0
        return is_danger, reasons

    def recommend(self, risk_label: str, model_input: dict, context: dict) -> dict:
        try:
            gestational_age = int(context.get("gestational_age_weeks", 12))

            # Penentuan Trimester Kehamilan
            if gestational_age <= 12:
                trimester = 1
            elif gestational_age <= 27:
                trimester = 2
            else:
                trimester = 3

            # ==============================================================
            # LAYER 1: HARD MEDICAL SAFETY FILTER (ZERO FALSE-SAFETY POLICY)
            # ==============================================================
            is_danger, contraindications = self._check_hard_medical_contraindications(model_input, context)

            if is_danger:
                # Semua aktivitas fisik dilarang mutlak demi keselamatan ibu & janin!
                all_codes = list(self.activity_lookup.keys())
                avoid_items = self._build_items(sorted(all_codes), "avoid")
                all_items = sorted(avoid_items, key=lambda x: x["name"])

                safety_alert = (
                    "KONTRAINDIKASI MEDIS MUTLAK (DILARANG OLAHRAGA): "
                    f"Ditemukan tanda bahaya kehamilan: {'; '.join(contraindications)}. "
                    "Demi keselamatan Ibu dan janin, SELURUH AKTIVITAS OLAHRAGA FISIK DILARANG SEMENTARA. "
                    "Ibu sangat disarankan untuk segera istirahat (bed rest) dan hubungi bidan atau dokter kandungan (SpOG)."
                )

                return {
                    "highly_recommended": [],
                    "allowed_with_caution": [],
                    "avoid": avoid_items,
                    "all": all_items,
                    "is_contraindicated": True,
                    "safety_status": "contraindicated",
                    "safety_alert": safety_alert,
                    "trimester": trimester,
                }

            # ==============================================================
            # LAYER 2: DYNAMIC TRIMESTER & CLINICAL CONTEXT ENGINE
            # ==============================================================
            base = self.rulebook["rules"]["high_risk" if risk_label == "high" else "low_risk"]

            highly = set(base["highly_recommended"])
            caution = set(base["allowed_with_caution"])
            avoid = set(base["avoid"])

            # --------------------------------------------------------------
            # FILTER TRIMESTER 1 (Minggu 1 - 12): Rawan Keguguran Dini
            # --------------------------------------------------------------
            if trimester == 1:
                t1_downgrade = {
                    "casual_cycling", "stationary_cycling", "treadmill_walking",
                    "slow_hiking", "light_dance", "low_impact_zumba", "light_stair_climbing"
                }
                highly -= t1_downgrade
                caution |= (t1_downgrade & (set(base["highly_recommended"]) | set(base["allowed_with_caution"])))

                # Prioritaskan olahraga paling lembut dan minim risiko di Trimester 1
                t1_promote = {"walking", "morning_walk", "evening_walk", "breathing_exercises", "stretching", "chair_yoga"}
                highly |= (t1_promote - avoid)
                caution -= t1_promote

            # --------------------------------------------------------------
            # FILTER TRIMESTER 2 (Minggu 13 - 27): Golden Period
            # --------------------------------------------------------------
            elif trimester == 2:
                # Gerakan berbaring telentang (supine) datar dibatasi jika usia kehamilan > 16 minggu
                if gestational_age > 16:
                    supine_risk = {"futsal", "basketball", "running"}
                    avoid |= supine_risk

            # --------------------------------------------------------------
            # FILTER TRIMESTER 3 (Minggu 28 - 42): Hindari Supine & Kurangi Risiko Sendi
            # --------------------------------------------------------------
            elif trimester == 3:
                # 1. Turunkan aktivitas berisiko guncangan, sendi, & kehilangan keseimbangan
                t3_reduce = {
                    "brisk_walking", "stair_climbing_light", "light_stair_climbing",
                    "light_dance", "casual_cycling", "slow_hiking", "stationary_cycling",
                    "low_impact_zumba"
                }
                highly -= t3_reduce
                caution |= (t3_reduce - avoid)

                # 2. Promosikan latihan persiapan persalinan, pernapasan, dan pelemasan panggul
                t3_promote = {
                    "pelvic_tilt", "cat_cow_modified", "bird_dog_modified",
                    "kegel", "chair_yoga", "breathing_exercises",
                    "back_stretch_routine", "sit_to_stand_slow"
                }
                highly |= (t3_promote - avoid)
                caution -= t3_promote

            # --------------------------------------------------------------
            # KONDISI KLINIS SPESIFIK LAINNYA
            # --------------------------------------------------------------
            if context.get("placenta_position_restriction", False):
                move_to_avoid = {
                    "brisk_walking", "stationary_cycling", "swimming", "water_aerobics",
                    "aqua_walking", "slow_hiking", "elliptical_light", "light_dance"
                }
                highly -= move_to_avoid
                caution -= move_to_avoid
                avoid |= move_to_avoid

            if context.get("back_pain", False):
                promote = {"pelvic_tilt", "chair_yoga", "stretching", "cat_cow_modified", "breathing_exercises"}
                avoid_back = {"jump_rope", "running", "jogging", "basketball", "futsal"}
                highly |= (promote - avoid)
                caution -= promote
                highly -= avoid_back
                caution -= avoid_back
                avoid |= avoid_back

            if context.get("low_impact_preference", False):
                move_to_avoid = {"jogging", "running", "jump_rope", "basketball", "futsal"}
                highly -= move_to_avoid
                caution -= move_to_avoid
                avoid |= move_to_avoid

            if not context.get("water_access", False):
                water_codes = {"swimming", "water_aerobics", "aqua_walking"}
                highly -= water_codes
                caution -= water_codes
                avoid |= water_codes

            if model_input.get("bmi", 0) >= 30:
                highly.discard("brisk_walking")
                caution.add("brisk_walking")

            if model_input.get("previous_complications", False):
                downgrade = {"stationary_cycling", "light_dance", "slow_hiking", "elliptical_light"}
                highly -= downgrade
                caution |= (downgrade - avoid)

            highly_items = self._build_items(sorted(highly), "highly_recommended")
            caution_items = self._build_items(sorted(caution), "allowed_with_caution")
            avoid_items = self._build_items(sorted(avoid), "avoid")

            all_items = highly_items + caution_items + avoid_items
            all_items = sorted(all_items, key=lambda x: x["name"])

            safety_status = "caution" if risk_label == "high" else "safe"
            safety_alert = None
            if risk_label == "high":
                safety_alert = (
                    "Profil kesehatan menunjukkan kategori risiko sedang/tinggi. "
                    "Utamakan olahraga berintensitas ringan dan batasi durasi latihan maksimal 15-20 menit per sesi."
                )

            return {
                "highly_recommended": highly_items,
                "allowed_with_caution": caution_items,
                "avoid": avoid_items,
                "all": all_items,
                "is_contraindicated": False,
                "safety_status": safety_status,
                "safety_alert": safety_alert,
                "trimester": trimester,
            }
        except Exception as e:
            raise RuleEngineError(f"Gagal membuat rekomendasi olahraga: {str(e)}") from e


recommendation_service = RecommendationService()
#!/usr/bin/env python3
"""
DICOM → VerSe-compatible NIfTI 변환 파이프라인

PACS에서 가져온 DICOM을 3d-spine 파이프라인 호환 NIfTI로 변환합니다.
DICOM 헤더(PatientID + StudyDate)를 읽어 환자·날짜별로 자동 분리합니다.
progress.csv에 변환 진행 상황과 DICOM 정보를 기록합니다.

- 증분 처리: 이미 완료된 건은 자동 스킵 (DICOM 추가 후 재실행하면 새 건만 처리)
- Spot instance 대응: 매 건 완료 시 즉시 progress.csv에 기록

사용법:
    python convert.py                                    # ./input → ./output
    python convert.py --input-dir /dicoms --output-dir /out
    python convert.py --ct-only                          # 분할 없이 CT만 변환

입력 구조 (PACS 폴더 — 깊이 무관, 재귀 탐색):
    input/
    ├── 2026-02-09/00457744/CT/*.dcm
    ├── 2026-02-19/00449886/CT/*.dcm
    └── ...  (어떤 깊이든 .dcm 파일만 있으면 자동 감지)

출력 구조 (VerSe-style, 3d-spine 호환):
    output/
    ├── sub-00457744_20260209/
    │   ├── sub-00457744_20260209_ct.nii.gz
    │   └── sub-00457744_20260209_seg-vert_msk.nii.gz
    └── sub-00449886_20260219/
        └── ...
    progress.csv
"""

import argparse
import csv
import logging
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# TotalSegmentator 척추 라벨 → VerSe 라벨 매핑
TOTALSEG_TO_VERSE = {
    "vertebrae_T1": 8, "vertebrae_T2": 9, "vertebrae_T3": 10,
    "vertebrae_T4": 11, "vertebrae_T5": 12, "vertebrae_T6": 13,
    "vertebrae_T7": 14, "vertebrae_T8": 15, "vertebrae_T9": 16,
    "vertebrae_T10": 17, "vertebrae_T11": 18, "vertebrae_T12": 19,
    "vertebrae_L1": 20, "vertebrae_L2": 21, "vertebrae_L3": 22,
    "vertebrae_L4": 23, "vertebrae_L5": 24,
}

PROGRESS_FIELDS = [
    "case_id", "patient_name", "patient_age", "patient_sex",
    "study_date", "modality", "num_slices", "dicom_source",
    "status", "ct_file", "mask_file", "timestamp",
]


# ============================================================
# DICOM 탐색
# ============================================================

def find_dicom_series(input_dir: Path) -> list[tuple[Path, str, dict]]:
    """DICOM 헤더를 읽어 환자·날짜별 시리즈를 자동 감지합니다.

    Returns:
        [(dicom_dir, case_id, dicom_info), ...]
    """
    # 재귀적으로 .dcm 파일이 있는 디렉토리 수집
    dcm_dirs: set[Path] = set()
    for pattern in ("**/*.dcm", "**/*.DCM", "**/*.ima", "**/*.IMA"):
        for f in input_dir.glob(pattern):
            dcm_dirs.add(f.parent)

    if not dcm_dirs:
        for d in input_dir.rglob("*"):
            if d.is_dir():
                has_extensionless = any(
                    f.is_file() and not f.suffix for f in d.iterdir()
                )
                if has_extensionless:
                    dcm_dirs.add(d)

    # 각 디렉토리에서 DICOM 헤더 읽어 case_id 및 환자 정보 추출
    series: dict[str, tuple[Path, dict]] = {}

    for dcm_dir in sorted(dcm_dirs):
        dcm_file = _pick_one_dcm(dcm_dir)
        if dcm_file is None:
            continue

        info = _read_dicom_info(dcm_file)
        num_dcm = _count_dcm_files(dcm_dir)
        info["num_slices"] = str(num_dcm)

        patient_id = _sanitize(info.get("patient_id", "unknown")) or "unknown"
        study_date = _sanitize(info.get("study_date", ""))

        if study_date:
            case_id = f"sub-{patient_id}_{study_date}"
        else:
            case_id = f"sub-{patient_id}"

        # 중복 case_id 처리
        base_id = case_id
        counter = 2
        while case_id in series and series[case_id][0] != dcm_dir:
            case_id = f"{base_id}_{counter}"
            counter += 1

        series[case_id] = (dcm_dir, info)

    return [(path, cid, info) for cid, (path, info) in sorted(series.items())]


def _read_dicom_info(dcm_file: Path) -> dict:
    """DICOM 파일에서 환자 정보를 읽습니다."""
    info = {
        "patient_id": "unknown",
        "patient_name": "",
        "patient_age": "",
        "patient_sex": "",
        "study_date": "",
        "modality": "",
    }
    try:
        ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
        info["patient_id"] = str(getattr(ds, "PatientID", "unknown")).strip()
        info["patient_name"] = str(getattr(ds, "PatientName", "")).strip()
        info["patient_age"] = str(getattr(ds, "PatientAge", "")).strip()
        info["patient_sex"] = str(getattr(ds, "PatientSex", "")).strip()
        info["study_date"] = str(getattr(ds, "StudyDate", "")).strip()
        info["modality"] = str(getattr(ds, "Modality", "")).strip()
    except Exception as e:
        logger.warning(f"  Cannot read DICOM header {dcm_file}: {e}")
    return info


def _pick_one_dcm(dcm_dir: Path) -> Path | None:
    """디렉토리에서 DICOM 파일 하나를 골라 반환합니다."""
    for pattern in ("*.dcm", "*.DCM", "*.ima", "*.IMA"):
        files = list(dcm_dir.glob(pattern))
        if files:
            return files[0]
    for f in dcm_dir.iterdir():
        if f.is_file() and not f.suffix:
            return f
    return None


def _count_dcm_files(dcm_dir: Path) -> int:
    """디렉토리의 DICOM 파일 수를 셉니다."""
    count = 0
    for pattern in ("*.dcm", "*.DCM", "*.ima", "*.IMA"):
        count += len(list(dcm_dir.glob(pattern)))
    if count == 0:
        count = sum(1 for f in dcm_dir.iterdir() if f.is_file() and not f.suffix)
    return count


def _sanitize(name: str) -> str:
    """파일명에 사용할 수 없는 문자를 제거합니다."""
    return "".join(c for c in name if c.isalnum() or c in "-_")


# ============================================================
# Progress CSV
# ============================================================

def load_progress(progress_path: Path) -> set[str]:
    """progress.csv에서 완료된 case_id 목록을 불러옵니다."""
    done = set()
    if not progress_path.exists():
        return done
    try:
        with open(progress_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "done":
                    done.add(row["case_id"])
    except Exception as e:
        logger.warning(f"Could not read progress file: {e}")
    return done


def append_progress(progress_path: Path, row: dict):
    """progress.csv에 한 줄 추가합니다 (즉시 flush)."""
    write_header = not progress_path.exists() or progress_path.stat().st_size == 0
    with open(progress_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


# ============================================================
# 변환 함수
# ============================================================

def convert_dicom_to_nifti(dicom_dir: Path, case_dir: Path, case_id: str) -> Path | None:
    """dcm2niix로 DICOM → NIfTI 변환."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        cmd = [
            "dcm2niix",
            "-z", "y",
            "-f", case_id,
            "-o", str(tmp),
            "-b", "n",
            str(dicom_dir),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"  dcm2niix failed for {case_id}: {result.stderr}")
            return None

        nii_files = list(tmp.glob("*.nii.gz")) + list(tmp.glob("*.nii"))
        if not nii_files:
            logger.error(f"  No NIfTI output for {case_id}")
            return None

        src = max(nii_files, key=lambda f: f.stat().st_size)
        dst = case_dir / f"{case_id}_ct.nii.gz"
        shutil.copy2(src, dst)

        return dst


def run_totalsegmentator(ct_path: Path, case_dir: Path, case_id: str, fast: bool = False) -> Path | None:
    """TotalSegmentator로 척추 분할 마스크 생성."""
    with tempfile.TemporaryDirectory() as tmpdir:
        seg_dir = Path(tmpdir) / "seg"

        cmd = [
            "TotalSegmentator",
            "-i", str(ct_path),
            "-o", str(seg_dir),
            "--task", "vertebrae_body",
        ]
        if fast:
            cmd.append("--fast")

        logger.info("  Running TotalSegmentator (this may take a few minutes)...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.warning("  vertebrae_body task failed, trying default task...")
            cmd = [
                "TotalSegmentator",
                "-i", str(ct_path),
                "-o", str(seg_dir),
            ]
            if fast:
                cmd.append("--fast")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"  TotalSegmentator failed: {result.stderr[:500]}")
                return None

        return merge_vertebra_masks(seg_dir, ct_path, case_dir, case_id)


def merge_vertebra_masks(seg_dir: Path, ct_path: Path, case_dir: Path, case_id: str) -> Path | None:
    """TotalSegmentator 개별 척추 마스크를 VerSe 라벨 규칙으로 합칩니다."""
    import nibabel as nib

    ref_img = nib.load(ct_path)
    combined_mask = np.zeros(ref_img.shape, dtype=np.uint8)
    found_count = 0

    for ts_name, verse_label in TOTALSEG_TO_VERSE.items():
        mask_path = seg_dir / f"{ts_name}.nii.gz"
        if not mask_path.exists():
            continue

        mask_data = nib.load(mask_path).get_fdata()
        combined_mask[mask_data > 0.5] = verse_label
        found_count += 1

    if found_count == 0:
        logger.error(f"  No vertebra masks found in {seg_dir}")
        return None

    logger.info(f"  Found {found_count}/17 vertebrae (T1-L5)")

    mask_img = nib.Nifti1Image(combined_mask, ref_img.affine, ref_img.header)
    mask_path = case_dir / f"{case_id}_seg-vert_msk.nii.gz"
    nib.save(mask_img, mask_path)

    return mask_path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DICOM → VerSe-compatible NIfTI 변환 (CT + 척추 분할 마스크)",
    )
    parser.add_argument("--input-dir", type=str, default="input",
                        help="DICOM 폴더 경로 (default: input)")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="출력 디렉토리 (default: output)")
    parser.add_argument("--ct-only", action="store_true",
                        help="CT 변환만 수행 (TotalSegmentator 분할 건너뛰기)")
    parser.add_argument("--fast", action="store_true",
                        help="TotalSegmentator fast 모드 (품질↓ 속도↑)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    progress_path = Path("progress.csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    if shutil.which("dcm2niix") is None:
        logger.error("dcm2niix not found. Install: conda install -c conda-forge dcm2niix")
        sys.exit(1)

    if not args.ct_only and shutil.which("TotalSegmentator") is None:
        logger.error("TotalSegmentator not found. Install: pip install TotalSegmentator")
        sys.exit(1)

    # 기존 진행 상황 로드
    done_cases = load_progress(progress_path)
    if done_cases:
        logger.info(f"Loaded progress: {len(done_cases)} cases already completed")

    # DICOM 스캔
    logger.info(f"Scanning {input_dir} for DICOM series...")
    dicom_series = find_dicom_series(input_dir)
    if not dicom_series:
        logger.error(f"No DICOM series found in {input_dir}")
        sys.exit(1)

    # 새로 처리할 건 필터
    todo = [(p, cid, info) for p, cid, info in dicom_series if cid not in done_cases]

    logger.info(f"Found {len(dicom_series)} total, {len(done_cases)} done, {len(todo)} to process")
    if args.ct_only:
        logger.info("Mode: CT only (no segmentation)")

    if not todo:
        logger.info("Nothing new to process. Done!")
        return

    success, fail = 0, 0

    for dicom_path, case_id, info in todo:
        logger.info(f"\n[{case_id}] Processing... ({success + fail + 1}/{len(todo)})")
        logger.info(f"  Patient: {info['patient_name']} | "
                     f"Age: {info['patient_age']} | Sex: {info['patient_sex']} | "
                     f"Date: {info['study_date']} | Mod: {info['modality']} | "
                     f"Slices: {info['num_slices']}")
        logger.info(f"  Source: {dicom_path}")

        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: DICOM → NIfTI
        logger.info("  Step 1: DICOM → NIfTI")
        ct_path = convert_dicom_to_nifti(dicom_path, case_dir, case_id)
        if ct_path is None:
            fail += 1
            append_progress(progress_path, {
                "case_id": case_id,
                "patient_name": info["patient_name"],
                "patient_age": info["patient_age"],
                "patient_sex": info["patient_sex"],
                "study_date": info["study_date"],
                "modality": info["modality"],
                "num_slices": info["num_slices"],
                "dicom_source": str(dicom_path),
                "status": "failed_ct",
                "ct_file": "",
                "mask_file": "",
                "timestamp": datetime.now().isoformat(),
            })
            continue
        logger.info(f"  → {ct_path.name}")

        mask_file = ""
        if not args.ct_only:
            # Step 2: 척추 분할
            logger.info("  Step 2: Vertebra segmentation")
            mask_path = run_totalsegmentator(ct_path, case_dir, case_id, fast=args.fast)
            if mask_path is None:
                fail += 1
                append_progress(progress_path, {
                    "case_id": case_id,
                    "patient_name": info["patient_name"],
                    "patient_age": info["patient_age"],
                    "patient_sex": info["patient_sex"],
                    "study_date": info["study_date"],
                    "modality": info["modality"],
                    "num_slices": info["num_slices"],
                    "dicom_source": str(dicom_path),
                    "status": "failed_seg",
                    "ct_file": str(ct_path.relative_to(Path.cwd())) if ct_path else "",
                    "mask_file": "",
                    "timestamp": datetime.now().isoformat(),
                })
                continue
            logger.info(f"  → {mask_path.name}")
            mask_file = str(mask_path.relative_to(Path.cwd()))

        # 성공 → progress.csv에 즉시 기록
        success += 1
        append_progress(progress_path, {
            "case_id": case_id,
            "patient_name": info["patient_name"],
            "patient_age": info["patient_age"],
            "patient_sex": info["patient_sex"],
            "study_date": info["study_date"],
            "modality": info["modality"],
            "num_slices": info["num_slices"],
            "dicom_source": str(dicom_path),
            "status": "done",
            "ct_file": str(ct_path.relative_to(Path.cwd())),
            "mask_file": mask_file,
            "timestamp": datetime.now().isoformat(),
        })

    logger.info(f"\nDone! {success} success, {fail} failed out of {len(todo)}")
    logger.info(f"Progress: {progress_path}")
    logger.info(f"Output: {output_dir}")

    if not args.ct_only:
        logger.info("\n3d-spine 파이프라인에서 사용:")
        logger.info(f"  python phase1_ssm/scripts/preprocess_meshes.py --ct-dir {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
DICOM → VerSe-compatible NIfTI 변환 파이프라인

입력: DICOM 폴더들 (환자별 하위 폴더)
출력: *_ct.nii.gz + *_seg-vert_msk.nii.gz (3d-spine 파이프라인 호환)

사용법:
    python convert.py --input-dir /path/to/dicoms --output-dir /path/to/output

입력 구조 (아래 중 하나):
    input_dir/
    ├── patient001/          # 환자별 폴더 (DICOM 시리즈)
    │   ├── IM-0001.dcm
    │   └── ...
    ├── patient002/
    └── ...

    또는:
    input_dir/               # 단일 환자 DICOM 시리즈
    ├── IM-0001.dcm
    └── ...

출력 구조 (flat, preprocess_meshes.py 호환):
    output_dir/
    ├── patient001_ct.nii.gz
    ├── patient001_seg-vert_msk.nii.gz
    ├── patient002_ct.nii.gz
    └── ...
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# TotalSegmentator 척추 라벨 → VerSe 라벨 매핑
# VerSe: T1=8, T2=9, ..., T12=19, L1=20, L2=21, L3=22, L4=23, L5=24
TOTALSEG_TO_VERSE = {
    "vertebrae_T1": 8,
    "vertebrae_T2": 9,
    "vertebrae_T3": 10,
    "vertebrae_T4": 11,
    "vertebrae_T5": 12,
    "vertebrae_T6": 13,
    "vertebrae_T7": 14,
    "vertebrae_T8": 15,
    "vertebrae_T9": 16,
    "vertebrae_T10": 17,
    "vertebrae_T11": 18,
    "vertebrae_T12": 19,
    "vertebrae_L1": 20,
    "vertebrae_L2": 21,
    "vertebrae_L3": 22,
    "vertebrae_L4": 23,
    "vertebrae_L5": 24,
}


def find_dicom_dirs(input_dir: Path) -> list[tuple[Path, str]]:
    """DICOM 시리즈가 있는 디렉토리들을 탐색합니다.

    Returns:
        [(dicom_dir, subject_id), ...]
    """
    dicom_dirs = []

    # 하위 폴더에 .dcm 파일이 있는지 확인
    subdirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    if subdirs:
        for subdir in subdirs:
            dcm_files = (
                list(subdir.glob("*.dcm"))
                + list(subdir.glob("*.DCM"))
                + list(subdir.glob("*.ima"))
                + list(subdir.glob("*.IMA"))
            )
            # .dcm 확장자 없는 DICOM도 탐색 (숫자 파일명 등)
            if not dcm_files:
                dcm_files = [
                    f for f in subdir.iterdir()
                    if f.is_file() and not f.suffix
                ]
            if dcm_files:
                dicom_dirs.append((subdir, subdir.name))

    # 하위 폴더가 없거나 DICOM이 없으면 input_dir 자체를 확인
    if not dicom_dirs:
        dcm_files = (
            list(input_dir.glob("*.dcm"))
            + list(input_dir.glob("*.DCM"))
            + list(input_dir.glob("*.ima"))
            + list(input_dir.glob("*.IMA"))
        )
        if dcm_files:
            dicom_dirs.append((input_dir, input_dir.name))

    return dicom_dirs


def convert_dicom_to_nifti(dicom_dir: Path, output_dir: Path, subject_id: str) -> Path | None:
    """dcm2niix로 DICOM → NIfTI 변환.

    Returns:
        변환된 NIfTI 파일 경로 또는 None
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        cmd = [
            "dcm2niix",
            "-z", "y",        # gzip 압축
            "-f", subject_id,  # 출력 파일명
            "-o", str(tmp),
            "-b", "n",        # BIDS sidecar 생략
            str(dicom_dir),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"  dcm2niix failed for {subject_id}: {result.stderr}")
            return None

        # 생성된 NIfTI 파일 찾기
        nii_files = list(tmp.glob("*.nii.gz")) + list(tmp.glob("*.nii"))
        if not nii_files:
            logger.error(f"  No NIfTI output for {subject_id}")
            return None

        # 여러 시리즈가 있으면 가장 큰 파일 선택 (CT 볼륨일 가능성 높음)
        src = max(nii_files, key=lambda f: f.stat().st_size)
        dst = output_dir / f"{subject_id}_ct.nii.gz"
        shutil.copy2(src, dst)

        return dst


def run_totalsegmentator(ct_path: Path, output_dir: Path, subject_id: str, fast: bool = False) -> Path | None:
    """TotalSegmentator로 척추 분할 마스크 생성.

    Returns:
        VerSe 포맷 마스크 파일 경로 또는 None
    """
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
            # vertebrae_body 태스크 실패 시 기본 태스크로 재시도
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

        # 개별 척추 마스크 → 단일 VerSe 포맷 마스크로 합침
        return merge_vertebra_masks(seg_dir, ct_path, output_dir, subject_id)


def merge_vertebra_masks(seg_dir: Path, ct_path: Path, output_dir: Path, subject_id: str) -> Path | None:
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
    mask_path = output_dir / f"{subject_id}_seg-vert_msk.nii.gz"
    nib.save(mask_img, mask_path)

    return mask_path


def main():
    parser = argparse.ArgumentParser(
        description="DICOM → VerSe-compatible NIfTI 변환 (CT + 척추 분할 마스크)",
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="DICOM 폴더 경로 (환자별 하위 폴더 또는 단일 시리즈)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="출력 디렉토리 (default: output)",
    )
    parser.add_argument(
        "--ct-only", action="store_true",
        help="CT 변환만 수행 (TotalSegmentator 분할 건너뛰기)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="TotalSegmentator fast 모드 (품질↓ 속도↑)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="이미 처리된 환자 건너뛰기",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # dcm2niix 설치 확인
    if shutil.which("dcm2niix") is None:
        logger.error("dcm2niix not found. Install: conda install -c conda-forge dcm2niix")
        sys.exit(1)

    # TotalSegmentator 설치 확인 (--ct-only가 아닐 때만)
    if not args.ct_only and shutil.which("TotalSegmentator") is None:
        logger.error("TotalSegmentator not found. Install: pip install TotalSegmentator")
        sys.exit(1)

    dicom_dirs = find_dicom_dirs(input_dir)
    if not dicom_dirs:
        logger.error(f"No DICOM series found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(dicom_dirs)} DICOM series")
    logger.info(f"Output: {output_dir}")
    if args.ct_only:
        logger.info("Mode: CT only (no segmentation)")

    success, fail = 0, 0

    for dicom_path, subject_id in dicom_dirs:
        logger.info(f"\n[{subject_id}] Processing...")

        # 이미 처리된 경우 건너뛰기
        ct_out = output_dir / f"{subject_id}_ct.nii.gz"
        mask_out = output_dir / f"{subject_id}_seg-vert_msk.nii.gz"
        if args.skip_existing:
            if args.ct_only and ct_out.exists():
                logger.info("  Skipping (already processed)")
                success += 1
                continue
            if not args.ct_only and ct_out.exists() and mask_out.exists():
                logger.info("  Skipping (already processed)")
                success += 1
                continue

        # Step 1: DICOM → NIfTI
        logger.info("  Step 1: DICOM → NIfTI")
        ct_path = convert_dicom_to_nifti(dicom_path, output_dir, subject_id)
        if ct_path is None:
            fail += 1
            continue

        logger.info(f"  → {ct_path.name}")

        if args.ct_only:
            success += 1
            continue

        # Step 2: 척추 분할 마스크 생성
        logger.info("  Step 2: Vertebra segmentation")
        mask_path = run_totalsegmentator(ct_path, output_dir, subject_id, fast=args.fast)
        if mask_path is None:
            fail += 1
            continue

        logger.info(f"  → {mask_path.name}")
        success += 1

    logger.info(f"\nDone! {success} success, {fail} failed")
    logger.info(f"Output: {output_dir}")

    if not args.ct_only:
        logger.info("\n3d-spine 파이프라인에서 사용:")
        logger.info(f"  python phase1_ssm/scripts/preprocess_meshes.py --ct-dir {output_dir}")


if __name__ == "__main__":
    main()

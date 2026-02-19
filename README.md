# dicom2nii

DICOM CT 영상을 [3d-spine](https://github.com/your-repo/3d-spine) 파이프라인에서 사용할 수 있는 VerSe-compatible NIfTI 형식으로 변환합니다.

## 변환 과정

1. **DICOM → NIfTI** : `dcm2niix`로 DICOM 시리즈를 `*_ct.nii.gz`로 변환
2. **척추 분할** : `TotalSegmentator`로 T1-L5 척추체를 자동 분할하여 `*_seg-vert_msk.nii.gz` 생성
3. **VerSe 라벨 매핑** : TotalSegmentator 출력을 VerSe 라벨 규칙(T1=8, ..., T12=19, L1=20, ..., L5=24)으로 변환

## 설치

### 사전 요구사항

- Python 3.10+
- CUDA 지원 GPU (TotalSegmentator용, 권장)

### 패키지 설치

```bash
pip install -r requirements.txt
```

`dcm2niix`가 PATH에 없으면 별도 설치:

```bash
# conda
conda install -c conda-forge dcm2niix

# Ubuntu/Debian
sudo apt install dcm2niix

# Windows (scoop)
scoop install dcm2niix
```

## 사용법

### 기본 사용 (CT 변환 + 척추 분할)

```bash
python convert.py --input-dir /path/to/dicoms --output-dir ./output
```

### CT 변환만 (분할 없이)

```bash
python convert.py --input-dir /path/to/dicoms --output-dir ./output --ct-only
```

### 빠른 분할 모드

```bash
python convert.py --input-dir /path/to/dicoms --output-dir ./output --fast
```

### 중단 후 이어서 처리

```bash
python convert.py --input-dir /path/to/dicoms --output-dir ./output --skip-existing
```

## 입력 구조

아래 두 가지 구조 모두 지원합니다:

```
# 환자별 하위 폴더
input_dir/
├── patient001/
│   ├── IM-0001.dcm
│   ├── IM-0002.dcm
│   └── ...
├── patient002/
│   └── ...
└── ...

# 단일 환자 (DICOM 파일이 직접 위치)
input_dir/
├── IM-0001.dcm
├── IM-0002.dcm
└── ...
```

## 출력 구조

```
output_dir/
├── patient001_ct.nii.gz              # CT 볼륨
├── patient001_seg-vert_msk.nii.gz    # 척추 분할 마스크 (VerSe 라벨)
├── patient002_ct.nii.gz
├── patient002_seg-vert_msk.nii.gz
└── ...
```

## 3d-spine 연동

변환 결과를 3d-spine 파이프라인에 바로 전달할 수 있습니다:

```bash
# 1. DICOM 변환
python convert.py --input-dir /path/to/dicoms --output-dir ../3d-spine/data/raw/custom

# 2. 3d-spine에서 메쉬 추출
cd ../3d-spine
python phase1_ssm/scripts/preprocess_meshes.py --ct-dir data/raw/custom
```

또는 전체 파이프라인 실행:

```bash
cd ../3d-spine
bash scripts/prepare_data.sh data/raw/custom
```

## 옵션 정리

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--input-dir` | DICOM 폴더 경로 (필수) | - |
| `--output-dir` | 출력 디렉토리 | `output` |
| `--ct-only` | CT 변환만 수행 (분할 건너뛰기) | off |
| `--fast` | TotalSegmentator 빠른 모드 | off |
| `--skip-existing` | 이미 처리된 환자 건너뛰기 | off |

## VerSe 라벨 매핑

| 척추 | VerSe 라벨 | | 척추 | VerSe 라벨 |
|------|-----------|---|------|-----------|
| T1   | 8         | | T10  | 17        |
| T2   | 9         | | T11  | 18        |
| T3   | 10        | | T12  | 19        |
| T4   | 11        | | L1   | 20        |
| T5   | 12        | | L2   | 21        |
| T6   | 13        | | L3   | 22        |
| T7   | 14        | | L4   | 23        |
| T8   | 15        | | L5   | 24        |
| T9   | 16        | |      |           |

# dicom2nii

PACS에서 가져온 DICOM CT 영상을 [3d-spine](https://github.com/xoincare/3d-spine) 파이프라인에서 사용할 수 있는 VerSe-compatible NIfTI 형식으로 변환합니다.

## 특징

- **자동 감지** : 입력 폴더를 재귀 탐색하여 DICOM 시리즈 자동 발견 (PACS 구조 무관)
- **증분 처리** : 이미 완료된 건은 자동 스킵 → DICOM 추가 후 재실행하면 새 건만 처리
- **Spot instance 대응** : 매 건 완료 시 `progress.csv`에 즉시 기록, 중단 후 재실행 안전
- **진행 기록** : `progress.csv`에 환자 정보(이름, 나이, 성별, 촬영일, 촬영법) 자동 정리

## 변환 과정

1. DICOM 헤더(PatientID + StudyDate) 읽어 환자·날짜별 고유 케이스 생성
2. `dcm2niix`로 DICOM → `*_ct.nii.gz` 변환
3. `TotalSegmentator`로 T1-L5 척추 자동 분할 → `*_seg-vert_msk.nii.gz` (VerSe 라벨)

## 설치

```bash
pip install -r requirements.txt
```

`dcm2niix`가 PATH에 없으면:

```bash
conda install -c conda-forge dcm2niix   # conda
sudo apt install dcm2niix               # Ubuntu
scoop install dcm2niix                  # Windows
```

## 사용법

### 기본 사용

```bash
# ./input 폴더의 DICOM → ./output 으로 변환
python convert.py
```

### DICOM 추가 후 재실행 (새 건만 처리)

```bash
# input/ 폴더에 새 DICOM 추가 후 그냥 다시 실행
python convert.py
```

### CT 변환만 (분할 없이)

```bash
python convert.py --ct-only
```

### 빠른 분할 모드

```bash
python convert.py --fast
```

### 경로 지정

```bash
python convert.py --input-dir /path/to/dicoms --output-dir /path/to/output
```

## 입력 구조

PACS에서 가져온 폴더를 그대로 `./input`에 넣으면 됩니다.
어떤 깊이든 `.dcm` 파일이 있으면 자동 감지합니다.

```
input/
├── 2026-02-09/
│   └── 00457744/
│       └── CT/
│           ├── CT1416487791.dcm
│           └── ...
├── 2026-02-19/
│   ├── 00449886/CT/*.dcm
│   └── 00460143/CT/*.dcm
└── ...
```

## 출력 구조

```
output/
├── sub-00457744_20260209/
│   ├── sub-00457744_20260209_ct.nii.gz
│   └── sub-00457744_20260209_seg-vert_msk.nii.gz
├── sub-00449886_20260219/
│   └── ...
└── sub-00460143_20260219/
    └── ...

progress.csv    ← 전체 진행 현황 + DICOM 정보
```

### progress.csv 예시

| case_id | patient_name | patient_age | patient_sex | study_date | modality | num_slices | status | timestamp |
|---------|-------------|-------------|-------------|------------|----------|------------|--------|-----------|
| sub-00457744_20260209 | Hong Gildong | 045Y | M | 20260209 | CT | 247 | done | 2026-02-19T... |
| sub-00449886_20260219 | Kim Cheolsu | 062Y | M | 20260219 | CT | 238 | done | 2026-02-19T... |

## 3d-spine 연동

```bash
# 1. DICOM 변환
python convert.py --output-dir ../3d-spine/data/raw/custom

# 2. 3d-spine 파이프라인
cd ../3d-spine
python phase1_ssm/scripts/preprocess_meshes.py --ct-dir data/raw/custom

# 또는 전체 파이프라인
bash scripts/prepare_data.sh data/raw/custom
```

## 옵션 정리

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--input-dir` | DICOM 폴더 경로 | `input` |
| `--output-dir` | 출력 디렉토리 | `output` |
| `--ct-only` | CT 변환만 (분할 건너뛰기) | off |
| `--fast` | TotalSegmentator 빠른 모드 | off |

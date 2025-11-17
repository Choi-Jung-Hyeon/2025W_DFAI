# 파일명: dataset_scripts/build_dataset_folders.py
# (기능: 원본 데이터를 훈련/검증 폴더로 복사 및 분리)

import os
from pathlib import Path
import glob
from tqdm import tqdm
import random
import shutil
from sklearn.model_selection import train_test_split

# --- (1) 팀원들이 수집할 원본 데이터 경로 ---
CLEAN_REAL_DIR = Path("../data/real_clean")
CLEAN_FAKE_DIR = Path("../data/fake_clean")

# --- (2) 최종 학습에 사용할 훈련/검증 폴더 ---
# (공지사항의 "train" 문자열 포함 금지 규칙 준수)
TRAIN_DIR = Path("../data/training_data")
VAL_DIR = Path("../data/validation_data")

VAL_SPLIT_RATIO = 0.1 # 10%를 검증용 데이터로 사용

def split_and_copy_files(input_dirs, label_name):
    """
    입력 폴더의 모든 원본 파일을 train/val 폴더로 복사합니다.
    (label_name은 'real' 또는 'fake'가 됩니다)
    """
    
    all_files = []
    for dir_path in input_dirs:
        if not dir_path.exists():
            print(f"경고: 원본 폴더를 찾을 수 없습니다: {dir_path}")
            print("팀원들이 데이터를 수집했는지 확인하세요.")
            continue
        print(f"원본 데이터 로드 중: {dir_path}")
        files = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png")) + list(dir_path.glob("*.jpeg"))
        all_files.extend(files)
        
    if not all_files:
        print(f"오류: {input_dirs}에서 이미지를 찾을 수 없습니다.")
        return

    # 훈련/검증 파일 리스트 분리
    train_files, val_files = train_test_split(
        all_files, test_size=VAL_SPLIT_RATIO, random_state=42, shuffle=True
    )

    print(f"[{label_name}] 총 {len(all_files)}개 이미지 발견. (Train: {len(train_files)}, Val: {len(val_files)})")

    # --- 훈련(Train) 데이터 복사 ---
    train_out_dir = TRAIN_DIR / label_name
    train_out_dir.mkdir(parents=True, exist_ok=True)
    for f in tqdm(train_files, desc=f"Train 데이터 복사 중 ({label_name})"):
        shutil.copy(f, train_out_dir / f.name)

    # --- 검증(Validation) 데이터 복사 ---
    val_out_dir = VAL_DIR / label_name
    val_out_dir.mkdir(parents=True, exist_ok=True)
    for f in tqdm(val_files, desc=f"Val 데이터 복사 중 ({label_name})"):
        shutil.copy(f, val_out_dir / f.name)

# --- (중요!) multiprocessing 오류 방지를 위해 __main__ 가드 사용 ---
if __name__ == "__main__":
    # 기존 폴더가 있으면 삭제 (중복 생성을 막기 위해)
    if TRAIN_DIR.exists():
        print(f"기존 폴더 삭제: {TRAIN_DIR}")
        shutil.rmtree(TRAIN_DIR)
    if VAL_DIR.exists():
        print(f"기존 폴더 삭제: {VAL_DIR}")
        shutil.rmtree(VAL_DIR)
    
    # 1. Real 데이터 처리
    split_and_copy_files([CLEAN_REAL_DIR], "real") 
    
    # 2. Fake 데이터 처리
    split_and_copy_files([CLEAN_FAKE_DIR], "fake")
    
    print("\n--- 모든 데이터셋 폴더 분리 완료 ---")
    print(f"Train 데이터 경로: {TRAIN_DIR.resolve()}")
    print(f"Val 데이터 경로: {VAL_DIR.resolve()}")
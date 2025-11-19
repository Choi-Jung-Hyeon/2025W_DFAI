import os
from pathlib import Path
import glob
from tqdm import tqdm
import random
import shutil
from sklearn.model_selection import train_test_split


CLEAN_REAL_DIR = Path("../data/real_clean")
CLEAN_FAKE_DIR = Path("../data/fake_clean")


TRAIN_DIR = Path("../data/training_data")
VAL_DIR = Path("../data/validation_data")

VAL_SPLIT_RATIO = 0.1

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

    train_files, val_files = train_test_split(
        all_files, test_size=VAL_SPLIT_RATIO, random_state=42, shuffle=True
    )

    print(f"[{label_name}] 총 {len(all_files)}개 이미지 발견. (Train: {len(train_files)}, Val: {len(val_files)})")

    train_out_dir = TRAIN_DIR / label_name
    train_out_dir.mkdir(parents=True, exist_ok=True)
    for f in tqdm(train_files, desc=f"Train 데이터 복사 중 ({label_name})"):
        shutil.copy(f, train_out_dir / f.name)

    val_out_dir = VAL_DIR / label_name
    val_out_dir.mkdir(parents=True, exist_ok=True)
    for f in tqdm(val_files, desc=f"Val 데이터 복사 중 ({label_name})"):
        shutil.copy(f, val_out_dir / f.name)

if __name__ == "__main__":
    if TRAIN_DIR.exists():
        print(f"기존 폴더 삭제: {TRAIN_DIR}")
        shutil.rmtree(TRAIN_DIR)
    if VAL_DIR.exists():
        print(f"기존 폴더 삭제: {VAL_DIR}")
        shutil.rmtree(VAL_DIR)
    
    split_and_copy_files([CLEAN_REAL_DIR], "real") 
    
    split_and_copy_files([CLEAN_FAKE_DIR], "fake")
    
    print("\n--- 모든 데이터셋 폴더 분리 완료 ---")
    print(f"Train 데이터 경로: {TRAIN_DIR.resolve()}")
    print(f"Val 데이터 경로: {VAL_DIR.resolve()}")
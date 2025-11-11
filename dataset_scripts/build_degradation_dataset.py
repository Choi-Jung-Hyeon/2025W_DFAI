# 파일명: dataset_scripts/build_degradation_dataset.py

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random
import shutil

# --- (1) 팀원들이 수집할 원본 데이터 경로 ---
# (이 폴더들은 .gitignore로 GitHub에 올라가지 않습니다)
CLEAN_REAL_DIR = Path("../data/real_clean")
CLEAN_FAKE_DIR = Path("../data/fake_clean")

# --- (2) 우리가 생성할 훼손 데이터 저장 경로 ---
DEGRADED_REAL_DIR = Path("../data/real_degraded")
DEGRADED_FAKE_DIR = Path("../data/fake_degraded")

# --- (3) 최종 학습에 사용할 훈련/검증 폴더 ---
# (이 폴더 구조는 STEP 3의 train_dino.py가 사용할 구조와 일치합니다)
TRAIN_DIR = Path("../data/train")
VAL_DIR = Path("../data/val")

# --- 훼손 강도 설정 (arXiv:2509.09172v1 논문 기반) ---
JPEG_QUALITY_RANGE = (70, 90) # '인터넷 전송' 시뮬레이션
BLUR_KERNEL_RANGE = (3, 7) # '재 디지털화' (초점)
GAUSSIAN_NOISE_STD_RANGE = (10, 30) # '재 디지털화' (센서 노이즈)
VAL_SPLIT_RATIO = 0.1 # 10%를 검증용 데이터로 사용

def apply_jpeg_compression(img_pil):
    """
    '인터넷 전송' 훼손 시뮬레이션: JPEG 압축
    [arXiv:2509.09172v1, Table 2 'Transmission']
    """
    quality = random.randint(*JPEG_QUALITY_RANGE)
    # Pillow 이미지를 메모리 상에서 JPEG로 압축 후 다시 로드
    import io
    buffer = io.BytesIO()
    img_pil.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

def apply_reshoot_simulation(img_pil):
    """
    '재 디지털화' 훼손 시뮬레이션: 블러 + 노이즈 (재촬영 근사)
    [arXiv:2509.09172v1, Table 2 'Re-digitization']
    """
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # 1. 블러 (초점 흐림)
    kernel_size = random.randrange(*BLUR_KERNEL_RANGE, 2) # 3, 5, 7 중 하나
    img_cv = cv2.GaussianBlur(img_cv, (kernel_size, kernel_size), 0)
    
    # 2. 노이즈 (센서 노이즈)
    std_dev = random.uniform(*GAUSSIAN_NOISE_STD_RANGE)
    noise = np.random.normal(0, std_dev, img_cv.shape)
    img_cv = np.clip(img_cv + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def process_and_split_data(input_dirs, output_base_dir):
    """
    입력 폴더의 모든 이미지를 3가지 버전(Clean, JPEG, Reshoot)으로
    train/val 폴더에 나누어 저장합니다.
    """
    
    all_files = []
    for dir_path in input_dirs:
        if not dir_path.exists():
            print(f"경고: 원본 폴더를 찾을 수 없습니다: {dir_path}")
            continue
        print(f"원본 데이터 로드 중: {dir_path}")
        files = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png")) + list(dir_path.glob("*.jpeg"))
        all_files.extend(files)
        
    if not all_files:
        print(f"오류: {input_dirs}에서 이미지를 찾을 수 없습니다. 팀원들이 데이터를 추가했는지 확인하세요.")
        return

    random.shuffle(all_files)
    split_idx = int(len(all_files) * (1 - VAL_SPLIT_RATIO))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"총 {len(all_files)}개 이미지 발견. (Train: {len(train_files)}, Val: {len(val_files)})")

    # --- 훈련(Train) 데이터 처리 ---
    train_out_dir = TRAIN_DIR / output_base_dir.name
    train_out_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in tqdm(train_files, desc=f"Train 데이터 생성 중 ({output_base_dir.name})"):
        try:
            img_pil = Image.open(img_path).convert("RGB")
            
            # 1. 원본(Clean) 저장
            img_pil.save(train_out_dir / f"{img_path.stem}_clean.jpg", "JPEG", quality=95)
            
            # 2. JPEG 훼손 버전 저장
            img_jpeg = apply_jpeg_compression(img_pil)
            img_jpeg.save(train_out_dir / f"{img_path.stem}_jpeg.jpg", "JPEG", quality=95)

            # 3. 재촬영 훼손 버전 저장
            img_reshoot = apply_reshoot_simulation(img_pil)
            img_reshoot.save(train_out_dir / f"{img_path.stem}_reshoot.jpg", "JPEG", quality=95)

        except Exception as e:
            print(f"파일 처리 오류 {img_path.name}: {e}")

    # --- 검증(Validation) 데이터 처리 ---
    # (주의: 검증 데이터는 훼손하지 않고 '원본'만 사용해야 
    #  모델이 훼손된 이미지에만 과적합되는 것을 방지할 수 있습니다.)
    val_out_dir = VAL_DIR / output_base_dir.name
    val_out_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in tqdm(val_files, desc=f"Val 데이터 생성 중 ({output_base_dir.name})"):
        try:
            img_pil = Image.open(img_path).convert("RGB")
            # 검증 데이터는 원본(Clean)만 저장
            img_pil.save(val_out_dir / f"{img_path.stem}_clean.jpg", "JPEG", quality=95)
        except Exception as e:
            print(f"파일 처리 오류 {img_path.name}: {e}")


if __name__ == "__main__":
    # 기존 폴더가 있으면 삭제 (중복 생성을 막기 위해)
    if TRAIN_DIR.exists(): shutil.rmtree(TRAIN_DIR)
    if VAL_DIR.exists(): shutil.rmtree(VAL_DIR)
    
    # 1. Real 데이터 처리 (real_clean -> train/real, val/real)
    # (출력 폴더 이름은 'real'로 지정)
    process_and_split_data([CLEAN_REAL_DIR], Path("real")) 
    
    # 2. Fake 데이터 처리 (fake_clean -> train/fake, val/fake)
    # (출력 폴더 이름은 'fake'로 지정)
    process_and_split_data([CLEAN_FAKE_DIR], Path("fake"))
    
    print("\n--- 모든 데이터셋 구축 완료 ---")
    print(f"Train 데이터 경로: {TRAIN_DIR.resolve()}")
    print(f"Val 데이터 경로: {VAL_DIR.resolve()}")
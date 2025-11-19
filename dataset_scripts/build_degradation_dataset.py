import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random
import shutil


CLEAN_REAL_DIR = Path("../data/real_clean")
CLEAN_FAKE_DIR = Path("../data/fake_clean")

DEGRADED_REAL_DIR = Path("../data/real_degraded")
DEGRADED_FAKE_DIR = Path("../data/fake_degraded")


TRAIN_DIR = Path("../data/train")
VAL_DIR = Path("../data/val")


JPEG_QUALITY_RANGE = (70, 90) 
BLUR_KERNEL_RANGE = (3, 7) 
GAUSSIAN_NOISE_STD_RANGE = (10, 30) 
VAL_SPLIT_RATIO = 0.1

def apply_jpeg_compression(img_pil):
    """
    '인터넷 전송' 훼손 시뮬레이션: JPEG 압축
    [arXiv:2509.09172v1, Table 2 'Transmission']
    """
    quality = random.randint(*JPEG_QUALITY_RANGE)
    import io
    buffer = io.BytesIO()
    img_pil.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

def apply_reshoot_simulation(img_pil):

    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    

    kernel_size = random.randrange(*BLUR_KERNEL_RANGE, 2) 
    img_cv = cv2.GaussianBlur(img_cv, (kernel_size, kernel_size), 0)
    

    std_dev = random.uniform(*GAUSSIAN_NOISE_STD_RANGE)
    noise = np.random.normal(0, std_dev, img_cv.shape)
    img_cv = np.clip(img_cv + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def process_and_split_data(input_dirs, output_base_dir):

    
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


    train_out_dir = TRAIN_DIR / output_base_dir.name
    train_out_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in tqdm(train_files, desc=f"Train 데이터 생성 중 ({output_base_dir.name})"):
        try:
            img_pil = Image.open(img_path).convert("RGB")
            

            img_pil.save(train_out_dir / f"{img_path.stem}_clean.jpg", "JPEG", quality=95)
            

            img_jpeg = apply_jpeg_compression(img_pil)
            img_jpeg.save(train_out_dir / f"{img_path.stem}_jpeg.jpg", "JPEG", quality=95)


            img_reshoot = apply_reshoot_simulation(img_pil)
            img_reshoot.save(train_out_dir / f"{img_path.stem}_reshoot.jpg", "JPEG", quality=95)

        except Exception as e:
            print(f"파일 처리 오류 {img_path.name}: {e}")


    val_out_dir = VAL_DIR / output_base_dir.name
    val_out_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in tqdm(val_files, desc=f"Val 데이터 생성 중 ({output_base_dir.name})"):
        try:
            img_pil = Image.open(img_path).convert("RGB")

            img_pil.save(val_out_dir / f"{img_path.stem}_clean.jpg", "JPEG", quality=95)
        except Exception as e:
            print(f"파일 처리 오류 {img_path.name}: {e}")


if __name__ == "__main__":

    if TRAIN_DIR.exists(): shutil.rmtree(TRAIN_DIR)
    if VAL_DIR.exists(): shutil.rmtree(VAL_DIR)
    

    process_and_split_data([CLEAN_REAL_DIR], Path("real")) 
    

    process_and_split_data([CLEAN_FAKE_DIR], Path("fake"))
    
    print("\n--- 모든 데이터셋 구축 완료 ---")
    print(f"Train 데이터 경로: {TRAIN_DIR.resolve()}")
    print(f"Val 데이터 경로: {VAL_DIR.resolve()}")
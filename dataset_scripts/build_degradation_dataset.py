import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random

# --- (1) 팀원들이 수집한 원본 데이터 경로 ---
CLEAN_REAL_DIR = Path("../data/real")
CLEAN_FAKE_DIR = Path("../data/fake_clean")

# --- (2) 우리가 생성할 훼손 데이터 저장 경로 ---
DEGRADED_REAL_DIR = Path("../data/real_degraded")
DEGRADED_FAKE_DIR = Path("../data/fake_degraded")

# 훼손 강도 설정
JPEG_QUALITY_RANGE = (70, 90)
BLUR_KERNEL_RANGE = (3, 7) # 홀수만
GAUSSIAN_NOISE_STD_RANGE = (10, 30)

def apply_jpeg_compression(img_pil, output_path):
    """
    '인터넷 전송' 훼손 시뮬레이션: JPEG 압축
    [arXiv:2509.09172v1]
    """
    quality = random.randint(*JPEG_QUALITY_RANGE)
    img_pil.save(output_path, "JPEG", quality=quality)

def apply_reshoot_simulation(img_pil, output_path):
    """
    '재 디지털화' 훼손 시뮬레이션: 블러 + 노이즈 (재촬영 근사)
    [arXiv:2509.09172v1]
    """
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # 1. 블러 (초점 흐림)
    kernel_size = random.randrange(*BLUR_KERNEL_RANGE, 2) # 3, 5, 7 중 하나
    img_cv = cv2.GaussianBlur(img_cv, (kernel_size, kernel_size), 0)
    
    # 2. 노이즈 (센서 노이즈)
    std_dev = random.uniform(*GAUSSIAN_NOISE_STD_RANGE)
    noise = np.random.normal(0, std_dev, img_cv.shape).astype(np.uint8)
    img_cv = cv2.add(img_cv, noise)
    
    cv2.imwrite(str(output_path), img_cv)

def process_directory(input_dir, output_dir):
    """입력 폴더의 모든 이미지를 훼손시켜 출력 폴더에 저장"""
    if not input_dir.exists():
        print(f"경고: 원본 폴더를 찾을 수 없습니다: {input_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpeg"))
    
    for img_path in tqdm(image_files, desc=f"훼손 데이터 생성 중: {input_dir.name}"):
        try:
            img_pil = Image.open(img_path).convert("RGB")
            
            # 1. JPEG 훼손 버전 저장
            jpeg_name = f"{img_path.stem}_jpeg.jpg"
            apply_jpeg_compression(img_pil, output_dir / jpeg_name)
            
            # 2. 재촬영 훼손 버전 저장
            reshoot_name = f"{img_path.stem}_reshoot.jpg"
            apply_reshoot_simulation(img_pil, output_dir / reshoot_name)
            
        except Exception as e:
            print(f"파일 처리 오류 {img_path.name}: {e}")

if __name__ == "__main__":
    print("--- 1. 'Real' 데이터 훼손 시작 ---")
    process_directory(CLEAN_REAL_DIR, DEGRADED_REAL_DIR)
    
    print("\n--- 2. 'Fake' 데이터 훼손 시작 ---")
    process_directory(CLEAN_FAKE_DIR, DEGRADED_FAKE_DIR)
    
    print("\n모든 훼손 데이터 생성이 완료되었습니다.")
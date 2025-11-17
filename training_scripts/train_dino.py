# 파일명: training_scripts/train_dino.py
# (수정: 4채널 입력 및 실시간 훼손/디포커스 맵 생성, __main__ 가드, NameError 수정)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import DinoV2ForImageClassification, AutoImageProcessor
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm
import random
import io
import os

# --- 1. 하이퍼파라미터 및 경로 설정 ---
BATCH_SIZE = 16 
LEARNING_RATE = 1e-5
EPOCHS = 30 # (1위 탈환을 위해 30 epoch 이상 권장)
MODEL_NAME = "facebook/dinov2-base"
SAVE_DIR = Path("../models_trained/dino_v2_4channel") # 1위 모델 저장 폴더

# (공지사항 준수: 'train', 'val' 폴더명 사용 X)
TRAIN_DIR = Path("../data/training_data")
VAL_DIR = Path("../data/validation_data")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [arXiv:2509.09172v1] 기반 훼손(Degradation) 파라미터
JPEG_QUALITY_RANGE = (70, 90)
BLUR_KERNEL_RANGE = (3, 7)
GAUSSIAN_NOISE_STD_RANGE = (10, 30)

# --- 2. 훼손 및 디포커스 맵 함수 ---

def get_defocus_map(img_pil):
    """
    (핵심 2) [11주차 디포커스 맵 세미나] 기반
    [ICV_01_Image_Filtering]의 라플라시안 필터로 디포커스 맵 계산
    """
    img_gray_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    # 32-bit float로 라플라시안 계산 (경계선을 더 잘 잡기 위해)
    laplacian = cv2.Laplacian(img_gray_cv, cv2.CV_32F, ksize=3)
    # 0-255 범위로 정규화
    defocus_map = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # (H, W) -> (H, W, 1) -> PIL Image
    return Image.fromarray(defocus_map)

def apply_degradation(img_pil):
    """
    (핵심 1) [arXiv:2509.09172v1] & [FakeChain] 세미나 기반
    '훼손'을 실시간(on-the-fly)으로 적용합니다.
    """
    choice = random.choice(['jpeg', 'reshoot', 'clean']) # 3가지 중 1개 랜덤 적용

    if choice == 'jpeg':
        # '인터넷 전송' 훼손 (JPEG 압축)
        quality = random.randint(*JPEG_QUALITY_RANGE)
        buffer = io.BytesIO()
        img_pil.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    
    elif choice == 'reshoot':
        # '재 디지털화' 훼손 (블러 + 노이즈)
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        kernel_size = random.randrange(*BLUR_KERNEL_RANGE, 2) # 홀수
        img_cv = cv2.GaussianBlur(img_cv, (kernel_size, kernel_size), 0)
        std_dev = random.uniform(*GAUSSIAN_NOISE_STD_RANGE)
        noise = np.random.normal(0, std_dev, img_cv.shape)
        img_cv = np.clip(img_cv + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    else:
        # 'Clean' 원본
        return img_pil

# --- 3. 커스텀 데이터셋 (4채널) ---
class DeepfakeDataset4Channel(Dataset):
    def __init__(self, file_paths, labels, processor, apply_degradation_flag=False):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor # DINOv2 프로세서
        self.apply_degradation = apply_degradation_flag

        # DINOv2의 기본 변환 (정규화 제외)
        self.resize_crop = transforms.Compose([
            transforms.Resize(processor.size["shortest_edge"]),
            transforms.CenterCrop(processor.size["shortest_edge"]),
        ])
        
        self.random_flip = transforms.RandomHorizontalFlip()
        self.to_tensor = transforms.ToTensor()
        
        # 3채널(RGB) / 1채널(Defocus) 정규화
        self.normalize_rgb = transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
        self.normalize_defocus = transforms.Normalize(mean=[0.5], std=[0.5]) # 1채널 정규화

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # 1. 원본(Clean) 로드
            clean_image_pil = Image.open(img_path).convert("RGB")
            
            # 2. 디포커스 맵 계산 (Clean 이미지 기준)
            defocus_map_pil = get_defocus_map(clean_image_pil)

            # 3. (훈련 시에만) RGB 이미지 훼손 적용
            if self.apply_degradation:
                rgb_image_pil = apply_degradation(clean_image_pil)
            else:
                rgb_image_pil = clean_image_pil # 검증 시에는 원본 사용

            # 4. 동기화 변환 (RGB와 Defocus Map을 동일하게 리사이즈/크롭)
            rgb_image = self.resize_crop(rgb_image_pil)
            defocus_map = self.resize_crop(defocus_map_pil)

            # 5. (훈련 시에만) 동기화 뒤집기 (Augmentation)
            if self.apply_degradation:
                state = torch.get_rng_state() # 랜덤 상태 고정
                rgb_image = self.random_flip(rgb_image)
                torch.set_rng_state(state) # 동일한 상태로
                defocus_map = self.random_flip(defocus_map)

            # 6. 텐서로 변환 및 정규화
            rgb_tensor = self.normalize_rgb(self.to_tensor(rgb_image))
            defocus_tensor = self.normalize_defocus(self.to_tensor(defocus_map))
            
            # 7. (핵심) 4채널로 결합 [R, G, B, Defocus]
            combined_tensor = torch.cat((rgb_tensor, defocus_tensor), dim=0)
            
        except Exception as e:
            print(f"Warning: Skipping broken image {img_path}: {e}")
            combined_tensor = torch.zeros((4, 224, 224)) # 4채널
            label = 0 # 임의의 레이블
            
        return combined_tensor, label

# --- 4. 데이터 로드 ---
def load_data():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    
    train_files_real = glob.glob(str(TRAIN_DIR / "real" / "*.*[jpg|png|jpeg]"))
    train_files_fake = glob.glob(str(TRAIN_DIR / "fake" / "*.*[jpg|png|jpeg]"))
    val_files_real = glob.glob(str(VAL_DIR / "real" / "*.*[jpg|png|jpeg]"))
    val_files_fake = glob.glob(str(VAL_DIR / "fake" / "*.*[jpg|png|jpeg]"))

    train_files = train_files_real + train_files_fake
    train_labels = [0] * len(train_files_real) + [1] * len(train_files_fake)
    
    val_files = val_files_real + val_files_fake
    val_labels = [0] * len(val_files_real) + [1] * len(val_files_fake)

    if not train_files or not val_files:
        print("="*50)
        print(f"오류: 훈련 또는 검증 데이터가 없습니다.")
        # (수정!) NameError 오류 수정: TRAIN_DIR, VAL_DIR 변수 참조로 변경
        print(f"  1. '{TRAIN_DIR}'와 '{VAL_DIR}' 상위 폴더(data/)에")
        print(f"     'real_clean', 'fake_clean' 폴더가 채워졌는지 확인하세요.")
        print(f"  2. `dataset_scripts/build_dataset_folders.py`를 실행했는지 확인하세요.")
        print("="*50)
        return None, None, None

    print(f"Total Train images: {len(train_files)} (Real: {len(train_files_real)}, Fake: {len(train_files_fake)})")
    print(f"Total Val images: {len(val_files)} (Real: {len(val_files_real)}, Fake: {len(val_files_fake)})")

    train_dataset = DeepfakeDataset4Channel(train_files, train_labels, processor, apply_degradation_flag=True)
    val_dataset = DeepfakeDataset4Channel(val_files, val_labels, processor, apply_degradation_flag=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, processor

# --- 5. 모델 정의 (4채널 입력) ---
def build_model():
    print(f"Loading pre-trained DINOv2 model: {MODEL_NAME} and modifying for 4-channel input")
    
    model = DinoV2ForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "REAL", 1: "FAKE"},
        label2id={"REAL": 0, "FAKE": 1},
        ignore_mismatched_sizes=True # 2-label 분류 헤드로 새로 초기화
    )

    # (핵심) DINOv2의 첫 번째 레이어(Patch Embedding) 수정
    orig_layer = model.dinov2.patch_embeddings.projection
    
    new_layer = nn.Conv2d(
        in_channels=4, 
        out_channels=orig_layer.out_channels, 
        kernel_size=orig_layer.kernel_size, 
        stride=orig_layer.stride, 
        padding=orig_layer.padding, 
        bias=(orig_layer.bias is not None)
    )

    # (중요!) 원본 3채널(RGB)의 가중치를 그대로 복사
    with torch.no_grad():
        new_layer.weight[:, :3, :, :] = orig_layer.weight.clone()
        # 새 4번째 채널(Defocus) 가중치는 0으로 초기화
        new_layer.weight[:, 3, :, :].zero_() 
        
        if orig_layer.bias is not None:
            new_layer.bias = nn.Parameter(orig_layer.bias.clone())

    # DINOv2 모델의 첫 번째 레이어를 새 4채널 레이어로 교체
    model.dinov2.patch_embeddings.projection = new_layer

    print("Model modified for 4-channel (RGB + Defocus) input.")
    return model.to(DEVICE)

# --- 6. 학습 및 검증 로직 ---
def train_model(model, train_loader, val_loader, processor):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = SAVE_DIR / "best_model.pth" # (가중치만 저장)

    for epoch in range(EPOCHS):
        # --- 훈련 (Train) ---
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", unit="batch")
        
        for images, labels in train_loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
        
        # --- 검증 (Validation) ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", unit="batch")

        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = correct / total

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            
            # (중요!) 제출에 필요한 3종 세트(가중치, config, processor) 저장
            torch.save(model.state_dict(), best_model_path)
            model.config.save_pretrained(SAVE_DIR)
            processor.save_pretrained(SAVE_DIR)
            
            print(f"New best model saved with Val Acc: {best_val_acc:.4f} to {SAVE_DIR}")

    print(f"Training finished. Best model saved at {SAVE_DIR} with Val Acc: {best_val_acc:.4f}")

# --- 7. 메인 실행 ---
# (수정!) [multiprocessing 공지사항] 준수를 위해 모든 실행 코드를 __main__ 가드 안에 배치
if __name__ == "__main__":
    # (선택) PyTorch가 spawn/fork 중 올바른 방식을 사용하도록 명시
    # torch.multiprocessing.set_start_method('spawn') 
    
    print(f"Using device: {DEVICE}")
    train_loader, val_loader, processor = load_data()
    
    if train_loader and val_loader:
        model = build_model()
        train_model(model, train_loader, val_loader, processor)
    else:
        print("모델 학습을 시작할 수 없습니다. 데이터셋 경로를 확인하세요.")
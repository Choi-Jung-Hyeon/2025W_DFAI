# 파일명: training_scripts/train_dino.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import DinoV2ForImageClassification, AutoImageProcessor
from PIL import Image
from pathlib import Path
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

# --- 1. 하이퍼파라미터 및 경로 설정 ---
# (DINOv2는 ViT보다 크므로 배치 크기를 작게 시작합니다)
BATCH_SIZE = 16 
LEARNING_RATE = 1e-5
EPOCHS = 10 # (테스트용. 실제로는 30-50 Epoch 권장)
MODEL_NAME = "facebook/dinov2-base" # DINOv2 백본
SAVE_DIR = Path("../models_trained/dino_v1") # 학습된 모델 저장 폴더

# (중요!) STEP 2의 `build_degradation_dataset.py`가 생성할 데이터 폴더
TRAIN_DIR = Path("../data/train")
VAL_DIR = Path("../data/val")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 2. 커스텀 데이터셋 정의 ---
class DeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # (중요) PIL로 열어야 DINOv2 프로세서와 호환됩니다.
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            # 깨진 이미지 로드 실패 시, 임의의 검은색 이미지 반환
            print(f"Warning: Skipping broken image {img_path}: {e}")
            image = torch.zeros((3, 224, 224)) # (C, H, W)
            
        return image, label

# --- 3. 데이터 로드 및 전처리 ---
def load_data():
    print("Loading data...")
    # 0 = Real, 1 = Fake
    
    # STEP 2에서 이미 train/val로 분리된 데이터를 읽어옵니다.
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
        print(f"팀원들이 'data/real_clean', 'data/fake_clean' 폴더를 채웠는지,")
        print(f"`dataset_scripts/build_degradation_dataset.py`를 실행했는지 확인하세요.")
        print("="*50)
        return None, None, None

    print(f"Total Train images: {len(train_files)} (Real: {len(train_files_real)}, Fake: {len(train_files_fake)})")
    print(f"Total Val images: {len(val_files)} (Real: {len(val_files_real)}, Fake: {len(val_files_fake)})")

    # DINOv2에 맞는 전처리기(ImageProcessor) 로드
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    
    # 훈련용 변환 (Augmentation): 리사이즈, 정규화 + 랜덤 뒤집기
    train_transform = transforms.Compose([
        transforms.Resize(processor.size["shortest_edge"]),
        transforms.CenterCrop(processor.size["shortest_edge"]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    # 검증용 변환: 리사이즈, 정규화
    val_transform = transforms.Compose([
        transforms.Resize(processor.size["shortest_edge"]),
        transforms.CenterCrop(processor.size["shortest_edge"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    train_dataset = DeepfakeDataset(train_files, train_labels, train_transform)
    val_dataset = DeepfakeDataset(val_files, val_labels, val_transform)

    # DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, processor

# --- 4. 모델 정의 ---
def build_model():
    print(f"Loading pre-trained DINOv2 model: {MODEL_NAME}")
    
    # (중요!) 앙상블 금지 규칙을 준수하는 "단일 모델"입니다.
    model = DinoV2ForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2, # (Real: 0, Fake: 1)
        id2label={0: "REAL", 1: "FAKE"},
        label2id={"REAL": 0, "FAKE": 1},
        ignore_mismatched_sizes=True # (필수) 분류 헤드(classifier)를 2-label로 새로 초기화합니다.
    )
    return model.to(DEVICE)

# --- 5. 학습 및 검증 로직 ---
def train_model(model, train_loader, val_loader, processor):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = SAVE_DIR / "best_model.pth"

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

        # 최고 성능 모델 저장
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            
            # (중요!) 나중에 `task.ipynb`에서 불러올 수 있도록
            # 가중치(state_dict)와 설정 파일(config, preprocessor)을 모두 저장합니다.
            torch.save(model.state_dict(), best_model_path)
            model.config.save_pretrained(SAVE_DIR)
            processor.save_pretrained(SAVE_DIR)
            
            print(f"New best model saved with Val Acc: {best_val_acc:.4f} to {SAVE_DIR}")

    print(f"Training finished. Best model saved at {best_model_path} with Val Acc: {best_val_acc:.4f}")

# --- 6. 메인 실행 ---
if __name__ == "__main__":
    train_loader, val_loader, processor = load_data()
    
    # 데이터 로드에 성공한 경우에만 학습 시작
    if train_loader and val_loader:
        model = build_model()
        train_model(model, train_loader, val_loader, processor)
    else:
        print("모델 학습을 시작할 수 없습니다. 데이터셋 경로를 확인하세요.")
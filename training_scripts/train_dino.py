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

# --- 1. 하이퍼파라미터 및 경로 설정 ---
# (DINOv2는 ViT보다 크므로 배치 크기를 작게 시작합니다)
BATCH_SIZE = 16 
LEARNING_RATE = 1e-5
EPOCHS = 10
MODEL_NAME = "facebook/dinov2-base" # DINOv2 백본
SAVE_PATH = "../models_trained/dino_v1.pth" # 학습된 모델 저장 경로

# (중요!) 팀원들이 수집/가공한 데이터셋 경로
DATA_DIR = Path("../data/")
REAL_DIRS = [DATA_DIR / "real_clean", DATA_DIR / "real_degraded"]
FAKE_DIRS = [DATA_DIR / "fake_clean", DATA_DIR / "fake_degraded"]

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
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            # 깨진 이미지 로드 실패 시, 임의의 검은색 이미지 반환
            print(f"Warning: Skipping broken image {img_path}: {e}")
            image = torch.zeros((3, 224, 224))
            
        return image, label

# --- 3. 데이터 로드 및 전처리 ---
def load_data():
    print("Loading data...")
    # 0 = Real, 1 = Fake
    all_files = []
    all_labels = []

    for dir_path in REAL_DIRS:
        print(f"Loading REAL from: {dir_path}")
        files = glob.glob(str(dir_path / "*.*[jpg|png|jpeg]"))
        all_files.extend(files)
        all_labels.extend([0] * len(files))

    for dir_path in FAKE_DIRS:
        print(f"Loading FAKE from: {dir_path}")
        files = glob.glob(str(dir_path / "*.*[jpg|png|jpeg]"))
        all_files.extend(files)
        all_labels.extend([1] * len(files))
        
    if not all_files:
        print("Error: No data found. Please check DATA_DIR paths.")
        return None, None, None

    print(f"Total images found: {len(all_files)}")

    # 훈련/검증 데이터 분리 (90% 훈련, 10% 검증)
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels, test_size=0.1, random_state=42, stratify=all_labels
    )

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

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
        ignore_mismatched_sizes=True # 분류기 헤드를 새로 초기화
    )
    return model.to(DEVICE)

# --- 5. 학습 및 검증 로직 ---
def train_model(model, train_loader, val_loader, processor):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    
    Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        # --- 훈련 (Train) ---
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for images, labels in train_loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_loop.set_postfix(loss=loss.item())
        
        # --- 검증 (Validation) ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")

        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

        # 최고 성능 모델 저장
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            # (중요!) 제출용 config 파일도 함께 저장
            model.config.save_pretrained(Path(SAVE_PATH).parent)
            processor.save_pretrained(Path(SAVE_PATH).parent)
            print(f"New best model saved with Val Acc: {best_val_acc:.4f} to {SAVE_PATH}")

    print("Training finished.")

# --- 6. 메인 실행 ---
if __name__ == "__main__":
    train_loader, val_loader, processor = load_data()
    if train_loader:
        model = build_model()
        train_model(model, train_loader, val_loader, processor)
import cv2
import os
import shutil
from PIL import Image

# 경로 설정
input_dir = "./data/images/real"   # Kaggle 원본 FairFace 이미지 경로
output_dir = "../data/images/real_clean"          # 통과한 고해상도 이미지 저장
reject_dir = "../data/images/reject"              # 조건 불충족 이미지 저장

os.makedirs(output_dir, exist_ok=True)
os.makedirs(reject_dir, exist_ok=True)

# 얼굴 검출기 초기화
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# -------------------------------
# 1️⃣ 해상도 검사 (Full HD 기준)
# -------------------------------
def is_high_res(image):
    h, w = image.shape[:2]
    return w >= 1920 and h >= 1080   # ✅ Full HD 이상만 통과

# -------------------------------
# 2️⃣ 블러 검사 (Laplacian variance)
# -------------------------------
def is_not_blurry(image, threshold=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance > threshold

# -------------------------------
# 3️⃣ 얼굴 검출 (MTCNN)
# -------------------------------
def has_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0

# -------------------------------
# 4️⃣ 고품질 JPG/PNG 확인
# -------------------------------
# def is_high_quality_jpg(img_path):
#     ext = img_path.lower().split('.')[-1]
#     if ext == "png":
#         return True
#     if ext not in ["jpg", "jpeg"]:
#         return False
#     try:
#         img = Image.open(img_path)
#         # EXIF, DPI, JFIF 등 메타데이터 존재 시 원본 확률 높음
#         if "jfif_density" in img.info or "dpi" in img.info:
#             return True
#     except:
#         pass
#     return True  # 기본적으로 .jpg 허용

# -------------------------------
# 5️⃣ 필터링 루프
# -------------------------------
def filter_images():
    total, passed = 0, 0

    for fname in os.listdir(input_dir):
        path = os.path.join(input_dir, fname)
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        total += 1
        img = cv2.imread(path)
        if img is None:
            shutil.copy(path, reject_dir)
            continue

        # 해상도 확인
        if not is_high_res(img):
            shutil.copy(path, reject_dir)
            continue

        # 얼굴 검출
        if not has_face(img):
            shutil.copy(path, reject_dir)
            continue

        # 흐림 검사
        if not is_not_blurry(img):
            shutil.copy(path, reject_dir)
            continue

        # # 포맷 검사
        # if not is_high_quality_jpg(path):
        #     shutil.copy(path, reject_dir)
        #     continue

        # 조건 통과 → real_clean 폴더로 복사
        shutil.copy(path, output_dir)
        passed += 1

        if passed % 50 == 0:
            print(f"✅ {passed} images passed / {total} checked")

    print(f"\n총 {total}개 중 {passed}개가 통과했습니다.")

# -------------------------------
# 6️⃣ 실행
# -------------------------------
if __name__ == "__main__":
    filter_images()

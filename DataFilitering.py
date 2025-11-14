import os
import cv2
import shutil
from pathlib import Path

# ========= 경로 =========
# 경로 알아서 바꿔서 지정하면 될 것 같아요
input_dir  = "./data/images/celeba_hq/train/train_real"     # 1024x1024 급 폴더(예: FFHQ 1024)
output_dir = "./data/images/celeba_hq/train/train_real_clean"   # 통과본
reject_dir = "./data/images/celeba_hq/train/train_reject"       # 리젝트

os.makedirs(output_dir, exist_ok=True)
os.makedirs(reject_dir, exist_ok=True)

# ========= 기준 =========
# Threshold값 조정해가면서 돌려보면 될 듯 (80으로 해보니까 잘 동작하는 것 같음)
MIN_SIDE       = 1024        # ✅ min(w,h) ≥ 1024
BLUR_THRESHOLD = 80.0        # Laplacian variance 임계값(필요시 60~100 사이 조정)
SCALE_FACTOR   = 1.1
MIN_NEIGHBORS  = 5
MAX_FILES      = 1000        # ✅ 앞 1000장만

# ========= 얼굴 검출기 =========
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ========= 판정 함수 =========
def is_high_res(img):
    h, w = img.shape[:2]
    return min(w, h) >= MIN_SIDE

def has_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS
    )
    return len(faces) > 0

def is_not_blurry(img, threshold=BLUR_THRESHOLD):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var  = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var > threshold, var

def classify_and_copy(p: Path):
    img = cv2.imread(str(p))
    if img is None:
        shutil.copy2(str(p), os.path.join(reject_dir, p.name))
        return f"[{p.name}] READ_FAIL → reject"

    h, w = img.shape[:2]

    # 1) 해상도 체크
    if not is_high_res(img):
        shutil.copy2(str(p), os.path.join(reject_dir, p.name))
        return f"[{p.name}] LOW_RES ({w}x{h}, need min-side≥{MIN_SIDE}) → reject"

    # 2) 얼굴 유무
    if not has_face(img):
        shutil.copy2(str(p), os.path.join(reject_dir, p.name))
        return f"[{p.name}] NO_FACE ({w}x{h}) → reject"

    # 3) 선명도
    sharp, var = is_not_blurry(img, BLUR_THRESHOLD)
    if not sharp:
        shutil.copy2(str(p), os.path.join(reject_dir, p.name))
        return f"[{p.name}] BLUR ({w}x{h}, var={var:.1f} ≤ {BLUR_THRESHOLD}) → reject"

    # 통과
    shutil.copy2(str(p), os.path.join(output_dir, p.name))
    return f"[{p.name}] PASS ({w}x{h}, var={var:.1f}) → real_clean"

def filter_images():
    files = [p for p in Path(input_dir).glob("*.jpg")]
    files.sort()
    # files = files[:MAX_FILES]

    total = len(files)
    passed = 0
    print(f"MIN_SIDE={MIN_SIDE}, BLUR_THRESHOLD={BLUR_THRESHOLD}, MAX_FILES={MAX_FILES}")
    print(f"Processing {total} files from: {input_dir}\n")

    for idx, p in enumerate(files, 1):
        log = classify_and_copy(p)
        print(f"{idx:04d}/{total}  {log}")
        if "PASS" in log:
            passed += 1

    print("\n=== SUMMARY ===")
    print(f"Checked: {total}, PASS: {passed}, FAIL: {total - passed}")
    print(f"Output:  {output_dir}")
    print(f"Reject:  {reject_dir}")

if __name__ == "__main__":
    filter_images()

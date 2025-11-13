import os, time, math, csv, cv2, numpy as np, requests, pandas as pd
from pathlib import Path

# ================== 설정 ==================
API_KEY    = "YOUR_FLICKR_API_KEY"
API_SECRET = "YOUR_FLICKR_SECRET"

SAVE_ROOT      = "../data"                         # 루트
CANDIDATE_DIR  = f"{SAVE_ROOT}/real_candidates"    # 원본 임시 저장(그대로)
CLEAN_DIR      = f"{SAVE_ROOT}/real_clean"         # 최종 통과(그대로 복사)
REJECT_DIR     = f"{SAVE_ROOT}/reject"             # 리젝트(그대로 복사)
META_CSV       = f"{SAVE_ROOT}/metadata_real_clean.csv"

os.makedirs(CANDIDATE_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(REJECT_DIR, exist_ok=True)

# 해상도 기준
MIN_SIDE      = 1024
FHD_W, FHD_H  = 1920, 1080

# 정제 기준
BLUR_THRESHOLD = 80.0      # Laplacian variance(선명도)
FACE_MIN_FRAC  = 0.20      # 얼굴박스가 이미지의 20% 이상
EYE_MAX_TILT   = 20.0      # 양눈 기울기 허용 각도(도)

# 탐지기 (YOLO 있으면 사용, 없으면 Haar)
USE_YOLO = False
try:
    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")      # 설치돼 있으면 자동 사용
    USE_YOLO = True
except Exception:
    pass

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# 다양성 확보용 검색어(원하는 만큼 추가/수정)
QUERIES = [
    "east asian woman portrait", "east asian man portrait",
    "black woman portrait", "black man portrait",
    "latino woman portrait", "latino man portrait",
    "indian woman portrait", "indian man portrait",
    "middle eastern woman portrait", "middle eastern man portrait",
    "white woman portrait", "white man portrait",
    "elderly woman portrait", "elderly man portrait",
    "teenager girl portrait", "teenager boy portrait"
]

FLICKR_API = "https://api.flickr.com/services/rest/"

def flickr_search(query, page=1, per_page=200):
    # CC 라이선스만: 4,5,6,9,10 (필요에 맞게 조정 가능)
    params = {
        "method": "flickr.photos.search",
        "api_key": API_KEY,
        "text": query,
        "license": "4,5,6,9,10",
        "content_type": 1,      # photos only
        "safe_search": 1,
        "media": "photos",
        "extras": "url_o,owner_name,license,o_dims",
        "per_page": per_page,
        "page": page,
        "format": "json",
        "nojsoncallback": 1
    }
    r = requests.get(FLICKR_API, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def pass_resolution_meta(p):
    try:
        w = int(p.get("width_o", 0)); h = int(p.get("height_o", 0))
    except:
        return False
    if min(w,h) >= MIN_SIDE: return True
    if (w >= FHD_W and h >= FHD_H): return True
    return False

def detect_face_yolo(img):
    res = yolo.predict(img, conf=0.35, classes=[0], verbose=False)  # class 0=person
    if len(res)==0 or len(res[0].boxes)==0:
        return None
    boxes = res[0].boxes.xyxy.cpu().numpy().astype(int)
    if boxes.shape[0]==0:
        return None
    areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    i = int(np.argmax(areas))
    x1,y1,x2,y2 = boxes[i]
    return (x1, y1, max(1, x2-x1), max(1, y2-y1))

def detect_face_haar(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces)==0: return None
    faces = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
    return faces[0]

def detect_face(img):
    return detect_face_yolo(img) if USE_YOLO else detect_face_haar(img)

def is_high_res(img):
    h, w = img.shape[:2]
    return (min(w, h) >= MIN_SIDE) or (w >= FHD_W and h >= FHD_H)

def is_not_blurry(img, thr=BLUR_THRESHOLD):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() > thr

def face_big_enough(face_box, img_shape):
    x,y,w,h = face_box
    H,W = img_shape[:2]
    return (w*h)/(W*H) >= FACE_MIN_FRAC

def eyes_frontal_and_level(img, face_box):
    x,y,w,h = face_box
    roi = img[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)
    if len(eyes) < 2: return False
    # 가장 멀리 떨어진 눈 두 개를 골라 기울기 확인
    eyes = sorted(eyes, key=lambda e:e[2]*e[3], reverse=True)[:3]
    centers = [(ex+ew/2, ey+eh/2) for (ex,ey,ew,eh) in eyes]
    best_d, pair = -1, None
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            (x1,y1),(x2,y2) = centers[i], centers[j]
            d = (x2-x1)**2 + (y2-y1)**2
            if d>best_d:
                best_d, pair = d, ((x1,y1),(x2,y2))
    if not pair: return False
    (x1,y1),(x2,y2) = pair
    ang = abs(np.degrees(np.arctan2((y2-y1), (x2-x1)+1e-6)))
    return ang <= EYE_MAX_TILT

def run():
    meta_rows = []
    for q in QUERIES:
        page = 1
        while True:
            data = flickr_search(q, page=page, per_page=200)
            photos = data.get("photos",{}).get("photo",[])
            if not photos: break

            for p in photos:
                url = p.get("url_o")
                if not url: 
                    continue
                if not pass_resolution_meta(p):
                    continue

                # 원본 다운로드(원본 그 자체로 저장)
                try:
                    resp = requests.get(url, timeout=30)
                    resp.raise_for_status()
                    fname = f"{q.replace(' ','_')}_{p['id']}.jpg"
                    path_candidate = os.path.join(CANDIDATE_DIR, fname)
                    with open(path_candidate, "wb") as f:
                        f.write(resp.content)
                except Exception:
                    continue

                # 판정용 로드
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                # 1) 실제 해상도 확인(보수적 재검)
                if not is_high_res(img):
                    shutil.copy2(path_candidate, os.path.join(REJECT_DIR, fname))
                    continue

                # 2) 얼굴 검출
                face = detect_face(img)
                if face is None:
                    shutil.copy2(path_candidate, os.path.join(REJECT_DIR, fname))
                    continue

                # 3) 얼굴 크기
                if not face_big_enough(face, img.shape):
                    shutil.copy2(path_candidate, os.path.join(REJECT_DIR, fname))
                    continue

                # 4) 정면/반측면 근사(양눈 + 기울기)
                if not eyes_frontal_and_level(img, face):
                    shutil.copy2(path_candidate, os.path.join(REJECT_DIR, fname))
                    continue

                # 5) 선명도
                if not is_not_blurry(img):
                    shutil.copy2(path_candidate, os.path.join(REJECT_DIR, fname))
                    continue

                # 최종 통과 → 원본 그대로 clean에 복사
                shutil.copy2(path_candidate, os.path.join(CLEAN_DIR, fname))

                # 메타 기록(출처/라이선스/크기)
                meta_rows.append({
                    "filename": fname,
                    "query": q,
                    "owner": p.get("ownername",""),
                    "license": p.get("license",""),
                    "source": "Flickr",
                    "width_o": p.get("width_o",""),
                    "height_o": p.get("height_o",""),
                    "original_url": url
                })

            pages = int(data.get("photos",{}).get("pages",1))
            if page >= pages: break
            page += 1
            time.sleep(0.4)  # 과한 호출 방지

    if meta_rows:
        pd.DataFrame(meta_rows).to_csv(META_CSV, index=False, encoding="utf-8-sig")
    print("완료. 실사, 고해상도, 정면/선명 조건 통과본만 저장됨.")
    print("CLEAN_DIR:", CLEAN_DIR)
    print("META_CSV:", META_CSV)

if __name__ == "__main__":
    run()

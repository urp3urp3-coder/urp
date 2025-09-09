# 9월 2주차: Ablation Study 실험.ver1 
# - 전처리, K-means 샘플링, DM 증강, 특징 증강 등 단계별 효과 분석
# - 회귀/분류 모두 대응
# - 데이터셋: metadata.csv + photo/*.jpg (NAIL_BOUNDING_BOXES 컬럼 사용)

import numpy as np, cv2
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, f1_score
import pandas as pd
import sklearn.model_selection
import sklearn.metrics
import ast

rng = np.random.default_rng(42)

# --------------------------
# 0) 전처리
# --------------------------
def gray_world_wb(img_bgr):
    img = img_bgr.astype(np.float32)
    means = img.reshape(-1,3).mean(0) + 1e-6
    gray = means.mean()
    gains = gray / means
    img *= gains
    return np.clip(img, 0, 255).astype(np.uint8)

def lab_clahe_L_only(img_bgr, clip=2.0, tiles=(8,8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def preprocess(img_bgr):
    x = gray_world_wb(img_bgr)
    x = lab_clahe_L_only(x)
    return x

# --------------------------
# 1) DM 약한 img2img (자리표시자)
#    실제 Diffusion으로 교체 시, 시그니처만 유지해 주세요.
# --------------------------
def dm_img2img_light(img_bgr, strength=0.25):
    img = img_bgr.astype(np.float32)
    # gamma
    g = rng.uniform(0.9, 1.1)
    img = 255.0 * ((img/255.0) ** g)
    # brightness/contrast (약하게)
    alpha = 1.0 + rng.uniform(-0.08, 0.08) * strength
    beta  = rng.uniform(-15, 15) * strength
    img = alpha*img + beta
    # noise & slight blur
    noise = rng.normal(0, 3.0*strength, img.shape)
    img = img + noise
    if rng.random() < 0.25*strength + 0.05:
        img = cv2.GaussianBlur(img, (3,3), 0)
    return np.clip(img, 0, 255).astype(np.uint8)

# --------------------------
# 2) 유틸: 색·좌표 특징 만들기
# --------------------------
def features_xy_color(img_bgr, ys, xs, alpha_xy=0.5):
    H, W = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    coords = np.stack([xs/W, ys/H], 1).astype(np.float32)
    ab = lab[ys, xs][:,1:3]
    hsvp = hsv[ys, xs]  # H,S,V
    feats = np.concatenate([alpha_xy*coords, ab, hsvp], 1)  # [x',y',a,b,H,S,V]
    return feats

# 글레어(하이라이트) 마스크 제거(선택)
def remove_glare(mask, img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    glare = ((hsv[...,1] < 20) & (hsv[...,2] > 240)).astype(np.uint8)  # 무채색+밝음
    cleaned = mask.copy().astype(np.uint8)
    cleaned[glare>0] = 0
    return cleaned

# --------------------------
# 3) k-means 샘플링
# --------------------------
def kmeans_sampling(img_bgr, mask, k=5, alpha_xy=0.5, extra_perc=0.0):
    m = mask.astype(bool)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return np.empty((0,7), np.float32), np.empty((0,2), np.int32)
    feats = features_xy_color(img_bgr, ys, xs, alpha_xy=alpha_xy)

    k = int(min(max(2, k), len(feats)))
    km = KMeans(n_clusters=k, n_init="auto", random_state=0)
    labels = km.fit_predict(feats)
    centers = km.cluster_centers_

    picked = []
    for c in range(k):
        idxs = np.where(labels==c)[0]
        if len(idxs)==0: continue
        dif = feats[idxs] - centers[c]
        # 대표 1
        j = idxs[np.argmin((dif*dif).sum(1))]
        picked.append(j)
        # 추가 r%
        if extra_perc > 0 and len(idxs) > 1:
            n_extra = max(1, int(len(idxs)*extra_perc))
            extra = rng.choice(idxs, size=min(n_extra, len(idxs)), replace=False)
            picked.extend(extra.tolist())

    picked = np.unique(np.array(picked))
    sel_feats = feats[picked]
    sel_xy = np.stack([ys[picked], xs[picked]], 1)
    return sel_feats.astype(np.float32), sel_xy.astype(np.int32)

# --------------------------
# 4) 특징 단계 증강 (소량)
# --------------------------
def feature_jitter(feats):
    f = feats.copy()
    # coords
    f[:,0:2] += rng.normal(0, 0.002, size=f[:,0:2].shape)                 # ±0.2%
    # a,b,H,S,V
    f[:,2]   += rng.normal(0, 1.0, size=f[:,2].shape)
    f[:,3]   += rng.normal(0, 1.0, size=f[:,3].shape)
    f[:,4]   += rng.normal(0, 0.8, size=f[:,4].shape)
    f[:,5:7] += rng.normal(0, 1.5, size=f[:,5:7].shape)
    return f

# --------------------------
# 5) 이미지 단위 요약 특징(샘플링 없을 때용)
# --------------------------
def roi_stats_features(img_bgr, mask, q=(0.25,0.5,0.75)):
    m = mask.astype(bool)
    if m.sum()==0:
        return np.zeros(30, np.float32)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    L = lab[...,0][m]; a = lab[...,1][m]; b = lab[...,2][m]
    Hc = hsv[...,0][m]; S = hsv[...,1][m]; V = hsv[...,2][m]
    def stats(v):
        qs = np.quantile(v, q)
        return np.array([v.mean(), v.std(), *qs], np.float32)
    # 각 채널 5통계 → 6*5=30
    return np.concatenate([stats(L), stats(a), stats(b), stats(Hc), stats(S), stats(V)], 0)

# --------------------------
# 6) 데이터셋 생성(한 config에 대해)
# task='reg'|'clf'
# --------------------------
def build_dataset(images, masks, y_per_image, group_ids,
                  use_dm=False, use_kmeans=False, extra_perc=0.0,
                  use_feat_jitter=False, k=5, alpha_xy=0.5, task='reg'):
    X_list, y_list, g_list = [], [], []
    for img, m, y, gid in zip(images, masks, y_per_image, group_ids):
        # 전처리
        x = preprocess(img)
        m2 = remove_glare(m, x)

        # DM 약한 img2img (선택)
        if use_dm:
            x = dm_img2img_light(x, strength=0.25)

        if use_kmeans:
            feats, _ = kmeans_sampling(x, m2, k=k, alpha_xy=alpha_xy, extra_perc=extra_perc)
            if feats.shape[0]==0:
                continue
            if use_feat_jitter:
                feats = feature_jitter(feats)
            # 포인트별 라벨/그룹
            X_list.append(feats)
            y_list.append(np.repeat(y, len(feats)))
            g_list.append(np.repeat(gid, len(feats)))
        else:
            # 이미지 요약 특징(고정 길이)
            feats = roi_stats_features(x, m2)
            X_list.append(feats[None, :])
            y_list.append(np.array([y]))
            g_list.append(np.array([gid]))
    if len(X_list)==0:
        return np.empty((0,7), np.float32), np.empty((0,), np.float32), np.empty((0,), int)
    X = np.vstack(X_list); y = np.hstack(y_list); grp = np.hstack(g_list)
    # 스케일러(선형모델 대비)
    if task=='reg':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        # 분류도 선형이면 스케일 권장
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X, y, grp

# --------------------------
# 7) 학습/평가 (GroupKFold)
# --------------------------
def evaluate_cv(X, y, grp, task='reg', n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for tr, te in gkf.split(X, y, groups=grp):
        Xtr, Xte = X[tr], X[te]; ytr, yte = y[tr], y[te]
        if task=='reg':
            model = Ridge(alpha=1.0, random_state=0)
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            scores.append(mean_absolute_error(yte, pred))
        else:
            # 이진 분류 가정
            model = LogisticRegression(max_iter=200)
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            scores.append(f1_score(yte, pred))
    return float(np.mean(scores)), float(np.std(scores))

# --------------------------
# 8) Ablation 실행
# --------------------------
def run_ablation(images, masks, y_per_image, group_ids, task='reg'):
    # 1. 전처리만 (통계 특징 사용)
    # 2. 전처리 + K-means 샘플링
    # 3. 전처리 + K-means + DM 증강
    # 4. 전처리 + K-means + DM 증강 + 추가 샘플
    # 5. 전처리 + K-means + DM 증강 + 추가 샘플 + 특징 증강
    configs = [
        dict(name='1_Preproc_Stats',          use_dm=False, use_kmeans=False, extra_perc=0.0, use_feat_jitter=False),
        dict(name='2_Preproc_DM',         use_dm=True, use_kmeans=False,  extra_perc=0.0, use_feat_jitter=False),
        dict(name='3_KMeans_DM',              use_dm=True,  use_kmeans=True,  extra_perc=0.0, use_feat_jitter=False),
        dict(name='4_KMeans_DM_Extra',        use_dm=True,  use_kmeans=True,  extra_perc=0.2, use_feat_jitter=False),
        dict(name='5_KMeans_DM_Extra_Jitter', use_dm=True,  use_kmeans=True,  extra_perc=0.2, use_feat_jitter=True),
    ]
    results = []
    print("--- Starting Ablation Study ---")
    for cfg in configs:
        print(f"Running config: {cfg['name']}...")
        X, y, grp = build_dataset(images, masks, y_per_image, group_ids,
                                  use_dm=cfg['use_dm'], use_kmeans=cfg['use_kmeans'],
                                  extra_perc=cfg['extra_perc'], use_feat_jitter=cfg['use_feat_jitter'],
                                  k=5, alpha_xy=0.5, task=task)
        if len(y)==0:
            print(f"Skipping {cfg['name']} due to empty dataset.")
            results.append((cfg['name'], np.nan, np.nan))
            continue
        
        metric_name = 'MAE' if task == 'reg' else 'F1'
        mean, std = evaluate_cv(X, y, grp, task=task, n_splits=5)
        results.append((cfg['name'], mean, std))
        print(f"  -> {cfg['name']:<28s} | {metric_name} = {mean:.4f} ± {std:.4f}\n")
    
    print("--- Ablation Study Finished ---")
    return results

# --------------------------
# 9) 데이터 로딩 및 메인 실행
# --------------------------
def main():
    try:
        metadata = pd.read_csv('data/metadata.csv')
        image_folder = 'data/photo/'
    except FileNotFoundError:
        print("Error: 'data/metadata.csv' or 'data/photo/' not found.")
        print("디렉토리 다시 설정1")
        return

    print(f"Loading data from {len(metadata)} records...")
    print("Generating masks from 'NAIL_BOUNDING_BOXES' column in the CSV.")
    
    images, masks, y_per_image, group_ids = [], [], [], []
    for idx, row in metadata.iterrows():
        patient_id = row['PATIENT_ID']
        img_path = f"{image_folder}{patient_id}.jpg"
        
        img = cv2.imread(img_path)
        if img is None:
            continue

        # bouding box에서 마스크 생성 
        try:
            bboxes = ast.literal_eval(row['NAIL_BOUNDING_BOXES'])
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(mask, (x1, y1), (x2, y2), (255), -1)
        except (ValueError, SyntaxError) as e:
            continue
        
        images.append(img)
        masks.append(mask)
        y_per_image.append(row['HB_LEVEL_GperL'])
        group_ids.append(patient_id)

    if not images:
        print("\nERROR: No data was loaded successfully.")
        print("디렉토리 다시 설정 2")
        return
        
    print(f"Successfully loaded {len(images)} image-mask pairs.")

    # Ablation study 실행 (회귀)
    run_ablation(images, masks, y_per_image, group_ids, task='reg')

if __name__ == '__main__':
    main()

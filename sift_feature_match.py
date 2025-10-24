import os, re, cv2, random, numpy as np
from glob import glob

DATASET_DIR = "Dataset"
OUTPUT_DIR = "Output"
LOWE_RATIO = 0.75
RANSAC_REPROJ_THRESH = 3.0
DRAW_MAX = 150
MAX_WIDTH = 900
SIFT_NFEATURES = 2000
BOTTOM_MARGIN = 70
IMAGE_EXTS = (".ppm", ".png", ".jpg", ".jpeg", ".tif")
TRANSFORM_MAP = {
    "bark": "Zoom + Rotation (scale & rotation)",
    "boat": "Zoom + Rotation (scale & rotation)",
    "graf": "Viewpoint change",
    "leuven": "Illumination change",
    "trees": "Blur change",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def try_create_sift():
    try:
        return cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
    except Exception:
        try:
            return cv2.xfeatures2d.SIFT_create(nfeatures=SIFT_NFEATURES)
        except Exception as e:
            raise RuntimeError(
                "SIFT not available. Install opencv-contrib-python."
            ) from e


def numeric_key(path):
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d+)$", name)
    if m:
        return int(m.group(1))
    return name.lower()


def find_images(folder):
    imgs = []
    for ext in IMAGE_EXTS:
        imgs += glob(os.path.join(folder, f"*{ext}"))
    imgs = sorted(set(imgs), key=numeric_key)
    return imgs


def resize_keep_aspect(img, max_width):
    h, w = img.shape[:2]
    if w <= max_width:
        return img, 1.0
    scale = max_width / w
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def draw_matches_single(
    img1, kp1, img2, kp2, matches, inlier_mask=None, draw_max=DRAW_MAX
):
    img1_c = (
        cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if len(img1.shape) == 2 else img1.copy()
    )
    img2_c = (
        cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if len(img2.shape) == 2 else img2.copy()
    )
    h1, w1 = img1_c.shape[:2]
    h2, w2 = img2_c.shape[:2]
    H = max(h1, h2)
    W = w1 + w2
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    canvas[:h1, :w1] = img1_c
    canvas[:h2, w1 : w1 + w2] = img2_c
    n_matches = len(matches)
    if n_matches == 0:
        return canvas
    indices = list(range(n_matches))
    if inlier_mask is not None and any(inlier_mask):
        inlier_indices = [i for i, v in enumerate(inlier_mask) if v]
        if len(inlier_indices) > draw_max:
            inlier_indices = random.sample(inlier_indices, draw_max)
        indices = sorted(inlier_indices)
    else:
        if n_matches > draw_max:
            indices = sorted(random.sample(indices, draw_max))
    random.seed(1)
    np.random.seed(1)

    def random_color():
        return tuple(int(x) for x in np.random.choice(np.arange(60, 256), size=3))

    palette = [random_color() for _ in range(len(indices))]
    for idx_idx, i in enumerate(indices):
        m = matches[i]
        color = palette[idx_idx]
        thickness = 2
        x1, y1 = map(int, kp1[m.queryIdx].pt)
        x2, y2 = map(int, kp2[m.trainIdx].pt)
        x2s = x2 + w1
        if inlier_mask is not None:
            if inlier_mask[i]:
                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2s, y2),
                    (255, 255, 255),
                    thickness + 2,
                    lineType=cv2.LINE_AA,
                )
                cv2.circle(canvas, (x1, y1), 5, (255, 255, 255), -1)
                cv2.circle(canvas, (x2s, y2), 5, (255, 255, 255), -1)
            else:
                color = tuple(int(c * 0.55) for c in color)
                thickness = 1
        cv2.line(canvas, (x1, y1), (x2s, y2), color, thickness, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 4, color, -1)
        cv2.circle(canvas, (x2s, y2), 4, color, -1)
    return canvas


def process_dataset(folder):
    ds_name = os.path.basename(folder)
    imgs = find_images(folder)
    out_file = os.path.join(OUTPUT_DIR, f"{ds_name}_matches.png")
    if len(imgs) < 2:
        blank = np.ones((400, 1000, 3), dtype=np.uint8) * 240
        text = f"{ds_name.upper()} - not enough images (need >=2)"
        cv2.putText(
            blank,
            text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(out_file, blank)
        print(f"[WARN] {ds_name}: not enough images. Saved placeholder -> {out_file}")
        return
    p1, p2 = imgs[0], imgs[1]
    orig1 = cv2.imread(p1, cv2.IMREAD_COLOR)
    orig2 = cv2.imread(p2, cv2.IMREAD_COLOR)
    if orig1 is None or orig2 is None:
        print(f"[ERROR] Couldn't read images for {ds_name}")
        return
    img1, s1 = resize_keep_aspect(orig1, MAX_WIDTH)
    img2, s2 = resize_keep_aspect(orig2, MAX_WIDTH)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = try_create_sift()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    kp1_count = len(kp1) if kp1 is not None else 0
    kp2_count = len(kp2) if kp2 is not None else 0
    good = []
    if des1 is not None and des2 is not None:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn = bf.knnMatch(des1, des2, k=2)
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < LOWE_RATIO * n.distance:
                good.append(m)
    inlier_mask = None
    ransac_inliers = 0
    if len(good) >= 4:
        pts1 = np.array([kp1[m.queryIdx].pt for m in good], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in good], dtype=np.float32)
        H_est, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, RANSAC_REPROJ_THRESH)
        if mask is not None:
            mask = mask.ravel().astype(bool).tolist()
            inlier_mask = mask
            ransac_inliers = int(sum(mask))
    draw_only_inliers = inlier_mask is not None and ransac_inliers >= 10
    used_mask = inlier_mask if draw_only_inliers else (None if len(good) > 0 else None)
    draw_count = min(DRAW_MAX, ransac_inliers if draw_only_inliers else len(good))
    canvas = draw_matches_single(
        img1, kp1, img2, kp2, good, inlier_mask=used_mask, draw_max=draw_count
    )
    ch, cw = canvas.shape[:2]
    new_h = ch + BOTTOM_MARGIN
    new_canvas = np.ones((new_h, cw, 3), dtype=np.uint8) * 255
    new_canvas[0:ch, 0:cw] = canvas
    bottom_text = " | ".join(
        [
            f"{ds_name.upper()} ({TRANSFORM_MAP.get(ds_name, 'Transformation: N/A')})",
            f"Keypoints: {kp1_count} vs {kp2_count}",
            f"Good Matches : {len(good)}",
        ]
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(bottom_text, font, font_scale, thickness)
    text_x = max(10, (cw - text_w) // 2)
    text_y = ch + (BOTTOM_MARGIN + text_h) // 2
    cv2.putText(
        new_canvas,
        bottom_text,
        (text_x + 1, text_y + 1),
        font,
        font_scale,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        new_canvas,
        bottom_text,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    cv2.imwrite(out_file, new_canvas)
    print(
        f"[OK] Saved: {out_file}  (kp: {kp1_count} vs {kp2_count}, good: {len(good)}, ransac_inliers: {ransac_inliers})"
    )


def main():
    if not os.path.exists(DATASET_DIR):
        print(
            f"Dataset folder '{DATASET_DIR}' not found. Place Dataset/ next to this script."
        )
        return
    subfolders = sorted(
        [
            os.path.join(DATASET_DIR, d)
            for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        ]
    )
    if not subfolders:
        print("No dataset subfolders found inside Dataset/.")
        return
    for folder in subfolders:
        process_dataset(folder)
    print("Done. Outputs are in the 'output' folder.")


if __name__ == "__main__":
    main()

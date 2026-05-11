import cv2
import numpy as np
import os
import glob
import re


# ============================================================
# KONFIGURASI
# ============================================================

DATASET_DIR = r"D:\VISION-BIMA\100GOPRO"

OUTPUT_DIR = r"D:\VISION-BIMA\output"
OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, "hasil_stitching_gopro.jpg")

USE_REDUNDANCY_FILTER = True

# Resize gambar sebelum stitching.
# Gunakan None jika tidak ingin resize.
# Contoh: 1200 artinya lebar gambar dibuat 1200 piksel.
RESIZE_WIDTH = 1200

# Mode stitcher:
# cv2.Stitcher_PANORAMA cocok untuk panorama kamera berputar.
# cv2.Stitcher_SCANS cocok untuk scan permukaan, drone, atau gerakan translasi.
STITCHER_MODE = cv2.Stitcher_SCANS


# ============================================================
# THRESHOLD REDUNDANCY FILTER
# ============================================================

HASH_HAMMING_THRESHOLD = 8
PIXEL_DIFF_THRESHOLD = 0.02
FEATURE_MATCH_THRESHOLD = 0.85


# ============================================================
# FUNGSI REDUNDANCY FILTER
# ============================================================

def _phash(image: np.ndarray, hash_size: int = 8) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size * 4, hash_size * 4))
    dct = cv2.dct(np.float32(small))
    dct_low = dct[:hash_size, :hash_size]
    median = np.median(dct_low)

    return (dct_low > median).flatten()


def _hamming(h1: np.ndarray, h2: np.ndarray) -> int:
    return int(np.count_nonzero(h1 != h2))


def _pixel_diff_ratio(img_a: np.ndarray, img_b: np.ndarray) -> float:
    size = (64, 64)

    small_a = cv2.resize(img_a, size)
    small_b = cv2.resize(img_b, size)

    diff = cv2.absdiff(small_a, small_b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    changed = np.count_nonzero(gray > 10)

    return changed / gray.size


def _feature_match_ratio(img_a: np.ndarray, img_b: np.ndarray) -> float:
    detector = cv2.ORB_create(200)

    def detect(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return detector.detectAndCompute(gray, None)

    kp_a, des_a = detect(img_a)
    kp_b, des_b = detect(img_b)

    if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return 1.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_a, des_b)

    good = [m for m in matches if m.distance < 50]

    return len(good) / max(len(kp_a), len(kp_b))


def redundancy_filter(image_a: np.ndarray, image_b: np.ndarray) -> bool:
    """
    Return:
        True  = image_b terlalu mirip, skip
        False = image_b berbeda, pakai untuk stitching
    """

    hash_a = _phash(image_a)
    hash_b = _phash(image_b)

    distance = _hamming(hash_a, hash_b)

    if distance <= HASH_HAMMING_THRESHOLD:
        print(f"[SKIP] pHash distance = {distance}")
        return True

    diff_ratio = _pixel_diff_ratio(image_a, image_b)

    if diff_ratio < PIXEL_DIFF_THRESHOLD:
        print(f"[SKIP] pixel diff = {diff_ratio:.3f}")
        return True

    match_ratio = _feature_match_ratio(image_a, image_b)

    if match_ratio >= FEATURE_MATCH_THRESHOLD:
        print(f"[SKIP] feature match = {match_ratio:.2f}")
        return True

    print(
        f"[PASS] hash = {distance}, "
        f"diff = {diff_ratio:.3f}, "
        f"feature = {match_ratio:.2f}"
    )

    return False


# ============================================================
# FUNGSI BANTUAN
# ============================================================

def natural_sort_key(path):
    """
    Mengurutkan nama file secara natural.
    Contoh:
        img2.jpg sebelum img10.jpg
    """
    filename = os.path.basename(path)

    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", filename)
    ]


def resize_image(image, width):
    if width is None:
        return image

    h, w = image.shape[:2]

    if w <= width:
        return image

    scale = width / w
    new_height = int(h * scale)

    resized = cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)

    return resized


def load_images_from_folder(folder_path):
    extensions = [
        "*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"
    ]

    image_paths = []

    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    image_paths = sorted(image_paths, key=natural_sort_key)

    images = []
    valid_paths = []

    for path in image_paths:
        image = cv2.imread(path)

        if image is None:
            print(f"[WARNING] Gagal membaca gambar: {path}")
            continue

        image = resize_image(image, RESIZE_WIDTH)

        images.append(image)
        valid_paths.append(path)

        print(f"[LOAD] {os.path.basename(path)}")

    return images, valid_paths


def filter_redundant_images(images, paths):
    if len(images) == 0:
        return [], []

    selected_images = [images[0]]
    selected_paths = [paths[0]]

    print("\nGambar pertama dipakai:")
    print(os.path.basename(paths[0]))

    for i in range(1, len(images)):
        previous_image = selected_images[-1]
        current_image = images[i]

        print(f"\nCek gambar: {os.path.basename(paths[i])}")

        is_redundant = redundancy_filter(previous_image, current_image)

        if not is_redundant:
            selected_images.append(current_image)
            selected_paths.append(paths[i])
            print(f"[SELECTED] {os.path.basename(paths[i])}")
        else:
            print(f"[REMOVED] {os.path.basename(paths[i])}")

    return selected_images, selected_paths


def stitch_images(images):
    if len(images) < 2:
        print("[ERROR] Minimal butuh 2 gambar untuk stitching.")
        return None, None

    stitcher = cv2.Stitcher_create(STITCHER_MODE)

    print("\nMulai proses stitching...")
    status, stitched = stitcher.stitch(images)

    return status, stitched


def explain_stitch_status(status):
    status_map = {
        cv2.Stitcher_OK: "Stitching berhasil.",
        cv2.Stitcher_ERR_NEED_MORE_IMGS: "Gagal. Gambar kurang atau overlap tidak cukup.",
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Gagal. Homography tidak bisa dihitung.",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Gagal. Parameter kamera tidak bisa disesuaikan."
    }

    return status_map.get(status, f"Gagal. Status tidak dikenal: {status}")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("======================================")
    print(" IMAGE STITCHING GOPRO + REDUNDANCY")
    print("======================================")
    print(f"Dataset folder : {DATASET_DIR}")
    print(f"Output file    : {OUTPUT_IMAGE}")

    images, paths = load_images_from_folder(DATASET_DIR)

    print(f"\nTotal gambar terbaca: {len(images)}")

    if len(images) < 2:
        print("[ERROR] Dataset harus berisi minimal 2 gambar.")
        return

    if USE_REDUNDANCY_FILTER:
        print("\nMenjalankan redundancy filter...")
        selected_images, selected_paths = filter_redundant_images(images, paths)
    else:
        selected_images = images
        selected_paths = paths

    print("\n======================================")
    print("RINGKASAN DATASET")
    print("======================================")
    print(f"Total gambar awal       : {len(images)}")
    print(f"Total gambar dipakai    : {len(selected_images)}")
    print(f"Total gambar dibuang    : {len(images) - len(selected_images)}")

    print("\nDaftar gambar yang dipakai:")
    for path in selected_paths:
        print("-", os.path.basename(path))

    status, stitched = stitch_images(selected_images)

    print("\n======================================")
    print("HASIL STITCHING")
    print("======================================")
    print(explain_stitch_status(status))

    if status == cv2.Stitcher_OK and stitched is not None:
        cv2.imwrite(OUTPUT_IMAGE, stitched)
        print(f"Hasil disimpan di: {OUTPUT_IMAGE}")

        cv2.imshow("Hasil Stitching", stitched)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
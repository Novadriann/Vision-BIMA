import cv2
import numpy as np
import argparse
from pathlib import Path


def load_images(folder_path, resize_width=None):
    folder = Path(folder_path)
    image_paths = sorted(
        list(folder.glob("*.jpg")) +
        list(folder.glob("*.jpeg")) +
        list(folder.glob("*.png"))
    )

    images = []

    for path in image_paths:
        img = cv2.imread(str(path))

        if img is None:
            print(f"Gagal membaca gambar: {path}")
            continue

        if resize_width is not None:
            h, w = img.shape[:2]
            scale = resize_width / w
            img = cv2.resize(img, (resize_width, int(h * scale)))

        images.append(img)
        print(f"Berhasil membaca: {path.name}")

    return images


def detect_features(image, method="orb"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method.lower() == "sift":
        detector = cv2.SIFT_create(5000)
    else:
        detector = cv2.ORB_create(nfeatures=5000)

    mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
    keypoints, descriptors = detector.detectAndCompute(gray, mask)

    return keypoints, descriptors


def match_features(desc1, desc2, method="orb", ratio=0.75):
    if desc1 is None or desc2 is None:
        return []

    if method.lower() == "sift":
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = matcher.knnMatch(desc2, desc1, k=2)

    good_matches = []

    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good_matches.append(m)

    return good_matches


def estimate_homography(kp1, kp2, matches):
    if len(matches) < 8:
        return None

    src_pts = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H


def warp_and_blend(base, new_img, H):
    h1, w1 = base.shape[:2]
    h2, w2 = new_img.shape[:2]

    corners_base = np.float32([
        [0, 0],
        [w1, 0],
        [w1, h1],
        [0, h1]
    ]).reshape(-1, 1, 2)

    corners_new = np.float32([
        [0, 0],
        [w2, 0],
        [w2, h2],
        [0, h2]
    ]).reshape(-1, 1, 2)

    warped_corners_new = cv2.perspectiveTransform(corners_new, H)
    all_corners = np.concatenate((corners_base, warped_corners_new), axis=0)

    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])

    output_width = x_max - x_min
    output_height = y_max - y_min

    warped_new = cv2.warpPerspective(
        new_img,
        translation @ H,
        (output_width, output_height)
    )

    canvas_base = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    canvas_base[-y_min:h1 - y_min, -x_min:w1 - x_min] = base

    mask_base = cv2.cvtColor(canvas_base, cv2.COLOR_BGR2GRAY) > 0
    mask_new = cv2.cvtColor(warped_new, cv2.COLOR_BGR2GRAY) > 0

    overlap = mask_base & mask_new
    only_base = mask_base & ~mask_new
    only_new = mask_new & ~mask_base

    result = np.zeros_like(canvas_base)
    result[only_base] = canvas_base[only_base]
    result[only_new] = warped_new[only_new]

    if np.any(overlap):
        blended = (
            0.5 * canvas_base[overlap].astype(np.float32) +
            0.5 * warped_new[overlap].astype(np.float32)
        )
        result[overlap] = blended.astype(np.uint8)

    return result


def crop_black_area(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image

    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    cropped = image[y:y + h, x:x + w]

    return cropped


def stitch_pair(base, new_img, method="orb"):
    kp1, desc1 = detect_features(base, method)
    kp2, desc2 = detect_features(new_img, method)

    matches = match_features(desc1, desc2, method)

    print(f"Jumlah match bagus: {len(matches)}")

    H = estimate_homography(kp1, kp2, matches)

    if H is None:
        print("Homography gagal. Gambar dilewati.")
        return base

    stitched = warp_and_blend(base, new_img, H)
    stitched = crop_black_area(stitched)

    return stitched


def stitch_dataset(images, method="orb"):
    if len(images) < 2:
        raise ValueError("Minimal diperlukan dua gambar untuk stitching.")

    mosaic = images[0]

    for i in range(1, len(images)):
        print(f"\nMenggabungkan gambar ke-{i + 1}")
        mosaic = stitch_pair(mosaic, images[i], method)

    return mosaic


def main():
    parser = argparse.ArgumentParser(
        description="Program sederhana untuk membuat orthomosaic dari dataset gambar UAV."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Folder berisi dataset gambar UAV."
    )

    parser.add_argument(
        "--output",
        default="hasil_orthomosaic.jpg",
        help="Nama file output."
    )

    parser.add_argument(
        "--method",
        default="orb",
        choices=["orb", "sift"],
        help="Metode ekstraksi fitur. Gunakan orb untuk cepat, sift untuk lebih akurat."
    )

    parser.add_argument(
    "--resize_width",
    default=None,
    help="Lebar resize gambar. Contoh: 1200. Tulis None atau kosongkan jika tidak ingin resize."
    )

    args = parser.parse_args()
    if args.resize_width is None or str(args.resize_width).lower() == "none":
        args.resize_width = None
    else:
        args.resize_width = int(args.resize_width)
    images = load_images(args.input, args.resize_width)

    if len(images) < 2:
        print("Dataset harus berisi minimal dua gambar.")
        return

    result = stitch_dataset(images, args.method)

    cv2.imwrite(args.output, result)
    print(f"\nSelesai. Hasil disimpan sebagai: {args.output}")


if __name__ == "__main__":
    main()
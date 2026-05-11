"""
stitch_uav.py
=============
Pipeline stitching foto UAV dari folder 100GOPRO menggunakan:
  1. Ekstraksi GPS dari EXIF metadata
  2. Sortir gambar berdasarkan koordinat GPS (path flight order)
  3. Redundancy filter (pHash + pixel diff + ORB feature match)
  4. cv2.Stitcher untuk menggabungkan gambar

Kebutuhan:
    pip install opencv-python pillow numpy piexif
"""

import cv2
import numpy as np
import os
import sys
import json
import math
import time
from pathlib import Path
from PIL import Image
import piexif

# ── Konfigurasi path ──────────────────────────────────────────────────────────
FOLDER_INPUT   = r"D:\VISION-BIMA\100GOPRO"   # <-- sesuaikan jika perlu
FOLDER_OUTPUT  = r"D:\VISION-BIMA\output"
RESULT_NAME    = "orthomosaic_result.jpg"

# ── Konfigurasi redundancy filter ────────────────────────────────────────────
HASH_HAMMING_THRESHOLD  = 8      # 0–64; makin kecil makin ketat
PIXEL_DIFF_THRESHOLD    = 0.02   # fraksi pixel yang berubah
FEATURE_MATCH_THRESHOLD = 0.85   # rasio feature match; lebih tinggi = lebih mirip

# ── Konfigurasi stitching ────────────────────────────────────────────────────
MAX_DIM        = 1200   # resize gambar agar tidak OOM (piksel, sisi terpanjang)
BATCH_SIZE     = 15     # jumlah gambar per batch; turunkan jika masih OOM
JPEG_QUALITY   = 95     # kualitas output JPEG


# ═══════════════════════════════════════════════════════════════════════════════
#  BAGIAN 1 — UTILITAS GPS
# ═══════════════════════════════════════════════════════════════════════════════

def _dms_to_decimal(dms_tuple, ref: str) -> float:
    """Konversi DMS (Degrees, Minutes, Seconds) ke desimal."""
    deg  = dms_tuple[0][0] / dms_tuple[0][1]
    mins = dms_tuple[1][0] / dms_tuple[1][1]
    secs = dms_tuple[2][0] / dms_tuple[2][1]
    decimal = deg + mins / 60 + secs / 3600
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def extract_gps(filepath: str) -> dict | None:
    """
    Baca koordinat GPS dari EXIF menggunakan piexif.
    Mengembalikan {'lat': float, 'lon': float, 'alt': float} atau None.
    """
    try:
        exif_dict = piexif.load(filepath)
        gps_data  = exif_dict.get("GPS", {})

        if not gps_data:
            return None

        lat_tag  = gps_data.get(piexif.GPSIFD.GPSLatitude)
        lat_ref  = gps_data.get(piexif.GPSIFD.GPSLatitudeRef)
        lon_tag  = gps_data.get(piexif.GPSIFD.GPSLongitude)
        lon_ref  = gps_data.get(piexif.GPSIFD.GPSLongitudeRef)
        alt_tag  = gps_data.get(piexif.GPSIFD.GPSAltitude)

        if not (lat_tag and lat_ref and lon_tag and lon_ref):
            return None

        lat = _dms_to_decimal(lat_tag, lat_ref.decode() if isinstance(lat_ref, bytes) else lat_ref)
        lon = _dms_to_decimal(lon_tag, lon_ref.decode() if isinstance(lon_ref, bytes) else lon_ref)
        alt = (alt_tag[0] / alt_tag[1]) if alt_tag else 0.0

        return {"lat": lat, "lon": lon, "alt": alt}

    except Exception as e:
        return None


def haversine_distance(coord_a: dict, coord_b: dict) -> float:
    """Jarak (meter) antara dua koordinat GPS menggunakan formula Haversine."""
    R = 6_371_000  # radius bumi dalam meter
    phi1, phi2 = math.radians(coord_a["lat"]), math.radians(coord_b["lat"])
    dphi       = math.radians(coord_b["lat"] - coord_a["lat"])
    dlambda    = math.radians(coord_b["lon"] - coord_a["lon"])
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def sort_by_gps(image_info_list: list) -> list:
    """
    Urutkan gambar menggunakan nearest-neighbor traversal dari titik tengah GPS.
    Strategi: mulai dari gambar paling barat-laut, lalu cari tetangga terdekat.
    Fallback: urutkan berdasarkan nama file jika GPS tidak tersedia.
    """
    gps_available = [info for info in image_info_list if info["gps"] is not None]
    no_gps        = [info for info in image_info_list if info["gps"] is None]

    if not gps_available:
        print("[GPS] Tidak ada data GPS – diurutkan berdasarkan nama file.")
        return sorted(image_info_list, key=lambda x: x["path"])

    # Mulai dari titik paling barat-laut (lat terbesar + lon terkecil)
    start = min(gps_available, key=lambda x: (-x["gps"]["lat"], x["gps"]["lon"]))
    ordered = [start]
    remaining = [i for i in gps_available if i is not start]

    while remaining:
        last = ordered[-1]
        nearest = min(remaining, key=lambda x: haversine_distance(last["gps"], x["gps"]))
        ordered.append(nearest)
        remaining.remove(nearest)

    return ordered + sorted(no_gps, key=lambda x: x["path"])


# ═══════════════════════════════════════════════════════════════════════════════
#  BAGIAN 2 — REDUNDANCY FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def _phash(image: np.ndarray, hash_size: int = 8) -> np.ndarray:
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small   = cv2.resize(gray, (hash_size * 4, hash_size * 4))
    dct     = cv2.dct(np.float32(small))
    dct_low = dct[:hash_size, :hash_size]
    median  = np.median(dct_low)
    return (dct_low > median).flatten()


def _hamming(h1: np.ndarray, h2: np.ndarray) -> int:
    return int(np.count_nonzero(h1 != h2))


def _pixel_diff_ratio(img_a: np.ndarray, img_b: np.ndarray) -> float:
    SIZE    = (64, 64)
    small_a = cv2.resize(img_a, SIZE)
    small_b = cv2.resize(img_b, SIZE)
    diff    = cv2.absdiff(small_a, small_b)
    gray    = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
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
    good    = [m for m in matches if m.distance < 50]
    return len(good) / max(len(kp_a), len(kp_b))


def redundancy_filter(image_a: np.ndarray, image_b: np.ndarray) -> bool:
    """True = SKIP (redundan), False = PASS (gambar baru)."""
    hash_a   = _phash(image_a)
    hash_b   = _phash(image_b)
    distance = _hamming(hash_a, hash_b)

    if distance <= HASH_HAMMING_THRESHOLD:
        print(f"  [SKIP] Stage 1 – pHash distance={distance}")
        return True

    diff_ratio = _pixel_diff_ratio(image_a, image_b)
    if diff_ratio < PIXEL_DIFF_THRESHOLD:
        print(f"  [SKIP] Stage 2 – pixel diff={diff_ratio:.3f}")
        return True

    match_ratio = _feature_match_ratio(image_a, image_b)
    if match_ratio >= FEATURE_MATCH_THRESHOLD:
        print(f"  [SKIP] Stage 3 – feature match={match_ratio:.2f}")
        return True

    print(f"  [PASS] hash={distance}  diff={diff_ratio:.3f}  feat={match_ratio:.2f}")
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  BAGIAN 3 — UTILITAS GAMBAR
# ═══════════════════════════════════════════════════════════════════════════════

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def load_image(filepath: str, max_dim: int = MAX_DIM) -> np.ndarray | None:
    """Baca gambar dan resize proporsional agar sisi terpanjang ≤ max_dim."""
    img = cv2.imread(filepath)
    if img is None:
        return None

    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img   = cv2.resize(img, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return img


def scan_folder(folder: str) -> list:
    """
    Scan folder, ekstrak GPS dari setiap file gambar.
    Mengembalikan list dict: {'path', 'name', 'gps'}.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"[ERROR] Folder tidak ditemukan: {folder}")
        sys.exit(1)

    files = sorted([
        f for f in folder_path.iterdir()
        if f.suffix.lower() in SUPPORTED_EXT
    ])

    if not files:
        print(f"[ERROR] Tidak ada gambar di folder: {folder}")
        sys.exit(1)

    print(f"\n[INFO] Ditemukan {len(files)} gambar di {folder}")

    result = []
    for f in files:
        gps = extract_gps(str(f))
        result.append({"path": str(f), "name": f.name, "gps": gps})

    gps_count = sum(1 for r in result if r["gps"] is not None)
    print(f"[INFO] GPS tersedia: {gps_count}/{len(result)} gambar")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  BAGIAN 4 — HIERARCHICAL BATCH STITCHING
# ═══════════════════════════════════════════════════════════════════════════════

STITCH_ERROR_MAP = {
    cv2.Stitcher_ERR_NEED_MORE_IMGS           : "Terlalu sedikit gambar / fitur tidak cukup",
    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL      : "Estimasi homografi gagal",
    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Kalibrasi kamera gagal",
}


def stitch_batch(images: list, names: list, label: str) -> np.ndarray | None:
    """
    Coba stitch satu batch gambar.
    Urutan: SCANS → PANORAMA → split-half fallback.
    Mengembalikan hasil pano (ndarray) atau None jika semua gagal.
    """
    if len(images) == 1:
        return images[0]

    for mode_name, mode in [("SCANS", cv2.Stitcher_SCANS),
                             ("PANORAMA", cv2.Stitcher_PANORAMA)]:
        stitcher = cv2.Stitcher.create(mode)
        try:
            status, pano = stitcher.stitch(images)
            if status == cv2.Stitcher_OK:
                print(f"  [OK] {label} — mode {mode_name} berhasil  "
                      f"({pano.shape[1]}x{pano.shape[0]})")
                return pano
            else:
                reason = STITCH_ERROR_MAP.get(status, f"status={status}")
                print(f"  [WARN] {label} — mode {mode_name} gagal: {reason}")
        except cv2.error as e:
            print(f"  [WARN] {label} — mode {mode_name} exception: {e}")

    # Fallback: bagi batch menjadi dua, stitch masing-masing, lalu gabungkan
    if len(images) >= 4:
        mid = len(images) // 2
        print(f"  [RETRY] {label} — split menjadi 2 sub-batch ({mid} + {len(images)-mid})")
        left  = stitch_batch(images[:mid],  names[:mid],  label + "-L")
        right = stitch_batch(images[mid:],  names[mid:],  label + "-R")
        if left is not None and right is not None:
            combined = stitch_batch([left, right], [label+"-L", label+"-R"], label+"-merge")
            return combined
        return left if left is not None else right

    return None


def hierarchical_stitch(images: list, names: list, out_folder: str) -> tuple:
    """
    Proses stitching secara hierarkis:
      Level 0 : stitch per BATCH_SIZE gambar  → simpan tile sementara
      Level 1+ : stitch tile-tile sebelumnya   → sampai 1 gambar tersisa

    Mengembalikan (path_hasil_final, elapsed_detik).
    """
    os.makedirs(out_folder, exist_ok=True)
    t0 = time.time()

    # ── Level 0: batch pertama ────────────────────────────────────────────────
    current_tiles   = []
    current_names   = []
    n_batches = math.ceil(len(images) / BATCH_SIZE)

    print(f"\n[LEVEL 0] {len(images)} gambar → {n_batches} batch × {BATCH_SIZE}")

    for b in range(n_batches):
        start = b * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(images))
        batch_imgs  = images[start:end]
        batch_names = names[start:end]
        label       = f"Batch-{b+1:03d}/{n_batches}"

        print(f"\n  {label}  [{start+1}–{end}]  {batch_names[0]} … {batch_names[-1]}")

        result = stitch_batch(batch_imgs, batch_names, label)

        if result is not None:
            tile_path = os.path.join(out_folder, f"_tile_L0_{b:04d}.jpg")
            cv2.imwrite(tile_path, result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            current_tiles.append(tile_path)
            current_names.append(f"tile_L0_{b:04d}")
        else:
            print(f"  [SKIP] {label} gagal total – dilewati.")

    if not current_tiles:
        return None, time.time() - t0

    # ── Level 1+: gabungkan tile hasil batch ──────────────────────────────────
    level = 1
    while len(current_tiles) > 1:
        print(f"\n[LEVEL {level}] Menggabungkan {len(current_tiles)} tile...")

        # Baca tile dari disk (hemat RAM di level sebelumnya)
        tile_images = []
        for tp in current_tiles:
            img = cv2.imread(tp)
            if img is not None:
                tile_images.append(img)

        next_tiles = []
        n_batches_l = math.ceil(len(tile_images) / BATCH_SIZE)

        for b in range(n_batches_l):
            start = b * BATCH_SIZE
            end   = min(start + BATCH_SIZE, len(tile_images))
            batch = tile_images[start:end]
            label = f"L{level}-Batch-{b+1:03d}"

            print(f"\n  {label}  [{start+1}–{end} dari {len(tile_images)} tile]")
            result = stitch_batch(batch, [f"t{i}" for i in range(start, end)], label)

            if result is not None:
                tile_path = os.path.join(out_folder, f"_tile_L{level}_{b:04d}.jpg")
                cv2.imwrite(tile_path, result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                next_tiles.append(tile_path)
            else:
                print(f"  [SKIP] {label} gagal – dilewati.")

        # Hapus tile level sebelumnya dari disk untuk hemat ruang
        for tp in current_tiles:
            try:
                os.remove(tp)
            except Exception:
                pass

        if not next_tiles:
            print("[ERROR] Semua tile level ini gagal digabungkan.")
            break

        current_tiles = next_tiles
        level += 1

    # ── Tile terakhir = hasil final ───────────────────────────────────────────
    final_tile   = current_tiles[0]
    final_output = os.path.join(out_folder, RESULT_NAME)

    import shutil
    shutil.move(final_tile, final_output)

    return final_output, time.time() - t0


# ═══════════════════════════════════════════════════════════════════════════════
#  BAGIAN 5 — PIPELINE UTAMA
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline():  # noqa: C901
    t_start = time.time()
    os.makedirs(FOLDER_OUTPUT, exist_ok=True)

    # ── Langkah 1: Scan folder dan ambil info GPS ──────────────────────────────
    print("\n" + "="*60)
    print("  STEP 1 — Scan Folder & Ekstraksi GPS")
    print("="*60)
    image_info = scan_folder(FOLDER_INPUT)

    # ── Langkah 2: Sortir berdasarkan GPS ─────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 2 — Sorting Berdasarkan GPS")
    print("="*60)
    image_info = sort_by_gps(image_info)
    print(f"[INFO] Urutan gambar setelah sorting GPS:")
    for i, info in enumerate(image_info):
        gps_str = (f"lat={info['gps']['lat']:.6f}, lon={info['gps']['lon']:.6f}"
                   if info["gps"] else "NO GPS")
        print(f"  [{i+1:03d}] {info['name']}  ({gps_str})")

    # Simpan urutan ke JSON untuk referensi
    order_path = os.path.join(FOLDER_OUTPUT, "gps_order.json")
    with open(order_path, "w") as f:
        json.dump([{"name": i["name"], "gps": i["gps"]} for i in image_info], f, indent=2)
    print(f"[INFO] Urutan GPS disimpan ke: {order_path}")

    # ── Langkah 3: Load gambar + Redundancy Filter ────────────────────────────
    print("\n" + "="*60)
    print("  STEP 3 — Load Gambar & Redundancy Filter")
    print("="*60)
    selected_images  = []
    selected_names   = []
    reference_image  = None

    for i, info in enumerate(image_info):
        print(f"\n[{i+1}/{len(image_info)}] Memproses: {info['name']}")
        img = load_image(info["path"])

        if img is None:
            print(f"  [WARN] Gagal membaca gambar, dilewati.")
            continue

        if reference_image is None:
            # Gambar pertama selalu dipakai sebagai referensi
            reference_image = img
            selected_images.append(img)
            selected_names.append(info["name"])
            print(f"  [PASS] Gambar pertama – dijadikan referensi.")
            continue

        # Bandingkan dengan gambar referensi terakhir
        is_redundant = redundancy_filter(reference_image, img)
        if not is_redundant:
            selected_images.append(img)
            selected_names.append(info["name"])
            reference_image = img   # update referensi ke gambar terbaru yang lolos

    print(f"\n[INFO] Gambar lolos filter: {len(selected_images)} / {len(image_info)}")
    print(f"[INFO] Gambar yang digunakan: {selected_names}")

    if len(selected_images) < 2:
        print("\n[ERROR] Minimal 2 gambar dibutuhkan untuk stitching.")
        print("        Coba turunkan threshold filter di bagian konfigurasi.")
        sys.exit(1)

    # ── Langkah 4: Batch Stitching Bertingkat ────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 4 — Batch Stitching Bertingkat")
    print(f"           {len(selected_images)} gambar → batch size {BATCH_SIZE}")
    print("="*60)

    t_stitch = time.time()
    output_path, elapsed_stitch = hierarchical_stitch(
        selected_images, selected_names, FOLDER_OUTPUT
    )

    if output_path is None:
        print("\n[ERROR] Semua batch gagal. Lihat log di atas untuk detail.")
        sys.exit(1)

    elapsed_stitch = time.time() - t_stitch

    # ── Langkah 5: Simpan hasil ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 5 — Hasil Final")
    print("="*60)
    final_path = os.path.join(FOLDER_OUTPUT, RESULT_NAME)

    if output_path != final_path:
        import shutil
        shutil.copy2(output_path, final_path)

    pano = cv2.imread(final_path)
    if pano is not None:
        print(f"[OK]   Ortomosaic final : {final_path}")
        print(f"       Dimensi           : {pano.shape[1]} x {pano.shape[0]} piksel")

        thumb_h = 800
        scale   = thumb_h / pano.shape[0]
        thumb   = cv2.resize(pano, (int(pano.shape[1] * scale), thumb_h))
        thumb_path = os.path.join(FOLDER_OUTPUT, "preview_thumb.jpg")
        cv2.imwrite(thumb_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"[OK]   Preview thumbnail: {thumb_path}")

    # ── Ringkasan ──────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print("\n" + "="*60)
    print("  RINGKASAN")
    print("="*60)
    print(f"  Total gambar input       : {len(image_info)}")
    print(f"  Lolos redundancy filter  : {len(selected_images)}")
    print(f"  Gambar dibuang (duplikat): {len(image_info) - len(selected_images)}")
    print(f"  Waktu stitching          : {elapsed_stitch:.1f} detik")
    print(f"  Total waktu pipeline     : {total_time:.1f} detik")
    print(f"  Output                   : {final_path}")
    print("="*60)


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Izinkan override folder dari command line
    # Contoh: python stitch_uav.py "D:\MY_FOLDER\100GOPRO" "D:\output"
    if len(sys.argv) >= 2:
        FOLDER_INPUT  = sys.argv[1]
    if len(sys.argv) >= 3:
        FOLDER_OUTPUT = sys.argv[2]

    print("\n╔══════════════════════════════════════════════╗")
    print("║        UAV ORTHOMOSAIC STITCHING PIPELINE    ║")
    print("║  GPS Sort → Redundancy Filter → cv2.Stitcher ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"\n  Input  : {FOLDER_INPUT}")
    print(f"  Output : {FOLDER_OUTPUT}")

    run_pipeline()
"""
============================================================
  Orthomosaic Stitching Pipeline — Manual Implementation v2
  Pipeline 2D Grid (All-Pairs Matching + BFS Graph)
============================================================
  Perubahan dari v1:
  - All-pairs matching: setiap gambar dicoba dengan SEMUA
    gambar lain (bukan hanya sequential i → i+1)
  - Graph BFS: mulai dari gambar referensi paling terhubung,
    ekspansi ke semua arah (atas/bawah/kiri/kanan)
  - Distance-weighted blending untuk overlap yang halus
============================================================
"""
 
import cv2
import numpy as np
import os
import glob
import logging
from pathlib import Path
from itertools import combinations
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stitcher_v2")
 
 
# ════════════════════════════════════════════════════════
#  KONFIGURASI
# ════════════════════════════════════════════════════════
class Config:
    IMAGE_DIR        = "./dataset"
    OUTPUT_PATH      = "./orthomosaic.jpg"
    EXTENSIONS       = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
 
    SIFT_NFEATURES   = 1000   # keypoints per gambar
    RATIO_THRESH     = 0.75   # Lowe's ratio test
    MIN_MATCH_COUNT  = 10     # minimal good matches
 
    # Skala saat deteksi fitur (lebih kecil = lebih cepat)
    WORK_SCALE       = 0.4
    # Skala saat warp ke canvas (hemat memori)
    COMPOSE_SCALE    = 0.5
 
    MAX_CANVAS_DIM   = 30_000
 
 
# ════════════════════════════════════════════════════════
#  STITCHER V2
# ════════════════════════════════════════════════════════
class OrthomosaicStitcher2D:
 
    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
 
    # ── 1. LOAD ─────────────────────────────────────────
    def load_images(self):
        paths = []
        for ext in self.cfg.EXTENSIONS:
            paths.extend(glob.glob(os.path.join(self.cfg.IMAGE_DIR, ext)))
            paths.extend(glob.glob(os.path.join(self.cfg.IMAGE_DIR, ext.upper())))
        paths = sorted(set(paths))
 
        if len(paths) < 2:
            raise FileNotFoundError(
                f"Minimal 2 gambar di '{self.cfg.IMAGE_DIR}'. "
                f"Ditemukan: {len(paths)}"
            )
 
        images = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                log.warning(f"Skip: {p}")
                continue
            h, w = img.shape[:2]
            img_s = cv2.resize(
                img,
                (int(w * self.cfg.COMPOSE_SCALE),
                 int(h * self.cfg.COMPOSE_SCALE))
            )
            images.append(img_s)
            log.info(f"Load  {Path(p).name:40s}  {img_s.shape[1]}×{img_s.shape[0]}")
 
        log.info(f"Total gambar valid: {len(images)}")
        return images
 
    # ── 2. DETECT ALL FEATURES ──────────────────────────
    def detect_all_features(self, images):
        log.info("\n[Feature Detection] …")
        detector = cv2.SIFT_create(nfeatures=self.cfg.SIFT_NFEATURES)
        result = []
        for i, img in enumerate(images):
            h, w = img.shape[:2]
            scale = self.cfg.WORK_SCALE / self.cfg.COMPOSE_SCALE
            small = cv2.resize(img, (int(w * scale), int(h * scale)))
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            kp, desc = detector.detectAndCompute(gray, mask)
            inv = 1.0 / scale
            for k in kp:
                k.pt   = (k.pt[0] * inv, k.pt[1] * inv)
                k.size *= inv
            result.append((kp, desc))
            log.info(f"  img[{i:02d}]: {len(kp)} keypoints")
        return result
 
    # ── 3. ALL-PAIRS MATCHING ───────────────────────────
    def match_all_pairs(self, all_features):
        n = len(all_features)
        total = n * (n - 1) // 2
        log.info(f"\n[All-Pairs Matching] Mencoba {total} kombinasi …")
        matcher     = cv2.BFMatcher(cv2.NORM_L2)
        match_graph = {}
 
        for i, j in combinations(range(n), 2):
            kp1, desc1 = all_features[i]
            kp2, desc2 = all_features[j]
            if desc1 is None or desc2 is None:
                continue
            try:
                raw = matcher.knnMatch(desc2, desc1, k=2)
            except Exception:
                continue
 
            good = [
                m for pair in raw
                if len(pair) == 2
                for m, nn in [pair]
                if m.distance < self.cfg.RATIO_THRESH * nn.distance
            ]
 
            status = "✔" if len(good) >= self.cfg.MIN_MATCH_COUNT else "✗"
            log.info(
                f"  img[{i:02d}] ↔ img[{j:02d}]: "
                f"{len(good):4d} matches  {status}"
            )
            if len(good) >= self.cfg.MIN_MATCH_COUNT:
                match_graph[(i, j)] = good
 
        log.info(f"  Pasangan valid: {len(match_graph)}/{total}")
        return match_graph
 
    # ── 4. ESTIMATE TRANSFORM ───────────────────────────
    def estimate_transform(self, kp1, kp2, matches):
        src = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
 
        A, _ = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0
        )
        if A is not None:
            return A, None
 
        H, _ = cv2.findHomography(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0
        )
        return None, H
 
    # ── 5. BUILD TRANSFORM GRAPH ────────────────────────
    def build_transform_graph(self, all_features, match_graph):
        log.info("\n[Transform Estimation] …")
        tgraph = {}

        # Urutkan dari match terbanyak → paling sedikit
        sorted_pairs = sorted(match_graph.items(), key=lambda x: len(x[1]), reverse=True)

        for (i, j), matches in sorted_pairs:
            kp1, _ = all_features[i]
            kp2, _ = all_features[j]
            A, H   = self.estimate_transform(kp1, kp2, matches)
            if A is not None or H is not None:
                mode = "Affine" if A is not None else "Homography"
                log.info(f"  img[{i:02d}] ↔ img[{j:02d}]: {mode}  ({len(matches)} matches)")
                tgraph[(i, j)] = (A, H, len(matches))

        return tgraph
 
    
    def compute_absolute_transforms(self, n_images, tgraph):
            log.info("\n[MST-BFS] Membangun posisi absolut via Maximum Spanning Tree …")

            # ── Bangun MST dengan Kruskal (ambil edge terkuat dulu) ──
            # Union-Find
            parent = list(range(n_images))
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                parent[find(x)] = find(y)

            # Urutkan edge dari match terbanyak (paling terpercaya)
            edges = sorted(tgraph.items(), key=lambda x: x[1][2], reverse=True)

            mst_edges = {}
            for (i, j), (A, H, n_match) in edges:
                if find(i) != find(j):
                    union(i, j)
                    mst_edges[(i, j)] = (A, H)
                    log.info(f"  MST edge: img[{i:02d}] ↔ img[{j:02d}]  ({n_match} matches)")

            log.info(f"  MST edges: {len(mst_edges)}")

            # ── BFS hanya lewat MST edges ──
            conn = {}
            for i, j in mst_edges:
                conn[i] = conn.get(i, 0) + 1
                conn[j] = conn.get(j, 0) + 1

            if not conn:
                raise RuntimeError("Tidak ada transformasi valid!")

            ref = max(conn, key=conn.get)
            log.info(f"  Referensi: img[{ref}] ({conn[ref]} koneksi)")

            absolute = {ref: np.eye(3, dtype=np.float64)}
            queue    = [ref]
            visited  = {ref}

            while queue:
                cur   = queue.pop(0)
                M_cur = absolute[cur]

                for (i, j), (A, H) in mst_edges.items():
                    M_rel = (
                        np.vstack([A, [0, 0, 1]]).astype(np.float64)
                        if A is not None else H.astype(np.float64)
                    )

                    try:
                        M_inv = np.linalg.inv(M_rel)
                    except np.linalg.LinAlgError:
                        continue

                    if np.any(np.abs(M_inv) > 1e6):
                        continue

                    if i == cur and j not in visited:
                        M_new = M_cur @ M_inv
                        if np.any(np.abs(M_new) > 1e6):
                            continue
                        absolute[j] = M_new
                        visited.add(j)
                        queue.append(j)
                    elif j == cur and i not in visited:
                        M_new = M_cur @ M_rel
                        if np.any(np.abs(M_new) > 1e6):
                            continue
                        absolute[i] = M_new
                        visited.add(i)
                        queue.append(i)

            isolated = set(range(n_images)) - visited
            log.info(f"  Terhubung: {len(absolute)}/{n_images}")
            if isolated:
                log.warning(f"  Terisolasi: {sorted(isolated)}")

            return absolute
 
    # ── 7. WARP ALL → CANVAS (distance-weighted blend) ──
    def warp_all_to_canvas(self, images, absolute_transforms):
        log.info("\n[Warping] Hitung ukuran canvas …")
 
        all_corners = []
        for idx, M in absolute_transforms.items():
            h, w    = images[idx].shape[:2]
            corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
            wc      = cv2.perspectiveTransform(corners, M).reshape(-1,2)
            all_corners.append(wc)
 
        all_corners = np.vstack(all_corners)
        x_min = int(np.floor(all_corners[:,0].min()))
        y_min = int(np.floor(all_corners[:,1].min()))
        x_max = int(np.ceil(all_corners[:,0].max()))
        y_max = int(np.ceil(all_corners[:,1].max()))
 
        cw = x_max - x_min
        ch = y_max - y_min
        log.info(f"  Canvas: {cw}×{ch} px")
 
        if cw > self.cfg.MAX_CANVAS_DIM or ch > self.cfg.MAX_CANVAS_DIM:
            raise RuntimeError(
                f"Canvas terlalu besar ({cw}×{ch}). "
                "Kurangi --compose-scale."
            )
 
        T      = np.float64([[1,0,-x_min],[0,1,-y_min],[0,0,1]])
        canvas = np.zeros((ch, cw, 3), dtype=np.float64)
        wmap   = np.zeros((ch, cw),    dtype=np.float64)
 
        for idx, M_abs in absolute_transforms.items():
            img = images[idx].astype(np.float64)
            M   = T @ M_abs
            h, w = img.shape[:2]
 
            warped = cv2.warpPerspective(
                img, M, (cw, ch),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
 
            # Distance-weight: piksel di tengah gambar lebih dipercaya
            dx = np.minimum(
                np.arange(w, dtype=np.float32),
                np.arange(w-1, -1, -1, dtype=np.float32)
            )
            dy = np.minimum(
                np.arange(h, dtype=np.float32),
                np.arange(h-1, -1, -1, dtype=np.float32)
            )
            weight_img = np.minimum(dx[np.newaxis,:], dy[:,np.newaxis])
            weight_img = (weight_img / (weight_img.max() + 1e-8)).astype(np.float32)
 
            warped_w = cv2.warpPerspective(
                weight_img, M, (cw, ch),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
 
            for c in range(3):
                canvas[:,:,c] += warped[:,:,c] * warped_w
            wmap += warped_w
 
            log.info(f"  Warp img[{idx:02d}] selesai")
 
        wmap = np.maximum(wmap, 1e-8)
        for c in range(3):
            canvas[:,:,c] /= wmap
 
        return np.clip(canvas, 0, 255).astype(np.uint8)
 
    # ── 8. CROP ─────────────────────────────────────────
    @staticmethod
    def crop_black(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords  = cv2.findNonZero(mask)
        if coords is None:
            return img
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]
 
    # ── MAIN ─────────────────────────────────────────────
    def run(self):
        log.info("=" * 60)
        log.info("  ORTHOMOSAIC v2  —  All-Pairs Graph Stitching")
        log.info("=" * 60)
 
        images       = self.load_images()
        all_features = self.detect_all_features(images)
        match_graph  = self.match_all_pairs(all_features)
        tgraph       = self.build_transform_graph(all_features, match_graph)
        abs_trans    = self.compute_absolute_transforms(len(images), tgraph)
        result       = self.warp_all_to_canvas(images, abs_trans)
 
        log.info("\n[Post-processing] Crop hitam …")
        result = self.crop_black(result)
 
        cv2.imwrite(self.cfg.OUTPUT_PATH, result)
        log.info(f"\n✔  Disimpan → '{self.cfg.OUTPUT_PATH}'")
        log.info(f"   Ukuran akhir: {result.shape[1]}×{result.shape[0]} px")
        log.info("=" * 60)
 
        try:
            max_w   = 1400
            scale   = max_w / result.shape[1]
            preview = cv2.resize(result, (max_w, int(result.shape[0] * scale)))
            cv2.imshow("Orthomosaic Result v2", preview)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            pass
 
        return result
 
 
# ════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════
def main():
    import argparse
    cfg = Config()
    p   = argparse.ArgumentParser(description="Orthomosaic v2 — All-Pairs Graph")
    p.add_argument("--input",         "-i", default=cfg.IMAGE_DIR)
    p.add_argument("--output",        "-o", default=cfg.OUTPUT_PATH)
    p.add_argument("--sift-features",       type=int,   default=cfg.SIFT_NFEATURES)
    p.add_argument("--ratio",               type=float, default=cfg.RATIO_THRESH)
    p.add_argument("--min-matches",         type=int,   default=cfg.MIN_MATCH_COUNT)
    p.add_argument("--compose-scale",       type=float, default=cfg.COMPOSE_SCALE)
    args = p.parse_args()
 
    cfg.IMAGE_DIR       = args.input
    cfg.OUTPUT_PATH     = args.output
    cfg.SIFT_NFEATURES  = args.sift_features
    cfg.RATIO_THRESH    = args.ratio
    cfg.MIN_MATCH_COUNT = args.min_matches
    cfg.COMPOSE_SCALE   = args.compose_scale
 
    OrthomosaicStitcher2D(cfg).run()
 
 
if __name__ == "__main__":
    main()
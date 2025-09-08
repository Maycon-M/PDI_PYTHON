import os
import random
import argparse
from typing import List, Tuple
from PIL import Image


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_images(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    files = []
    for root, _dirs, fnames in os.walk(folder):
        for f in fnames:
            if os.path.splitext(f)[1].lower() in exts:
                files.append(os.path.join(root, f))
    return files


def load_roi_with_mask(path: str, white_threshold: int = 245) -> Tuple[Image.Image, Image.Image]:
    """
    Returns (roi_rgba, mask_L). If the ROI has an alpha channel, uses it as mask.
    Otherwise, builds a mask by treating near-white pixels as background (transparent).
    """
    im = Image.open(path)
    if im.mode in ("RGBA", "LA"):
        rgba = im.convert("RGBA")
        mask = rgba.split()[-1]
        return rgba, mask
    # Build mask from near-white background
    rgb = im.convert("RGB")
    # Create mask: pixels strictly below threshold on any channel -> foreground
    # We'll use a simple per-pixel pass to avoid numpy dependency.
    w, h = rgb.size
    mask = Image.new("L", (w, h), 0)
    rgb_px = rgb.load()
    m_px = mask.load()
    for y in range(h):
        for x in range(w):
            r, g, b = rgb_px[x, y]
            if not (r >= white_threshold and g >= white_threshold and b >= white_threshold):
                m_px[x, y] = 255
    rgba = rgb.copy()
    rgba.putalpha(mask)
    return rgba, mask


def scale_roi_to_fit(
    roi: Image.Image, bg_size: Tuple[int, int], min_frac: float = 0.2, max_frac: float = 0.5
) -> Image.Image:
    """
    Scales ROI keeping aspect ratio. Picks a random width fraction of background
    between min_frac and max_frac, clamped so ROI fits entirely.
    """
    bw, bh = bg_size
    iw, ih = roi.size
    if iw == 0 or ih == 0:
        return roi

    # Clamp target width fraction to ensure it fits in height too
    # Compute the maximum fraction that still fits both width and height.
    max_w_frac_by_w = 1.0
    max_w_frac_by_h = (bh / ih) * (iw / bw)
    max_allowed = min(max_w_frac_by_w, max_w_frac_by_h, max_frac)
    if max_allowed < min_frac:
        target_frac = max_allowed  # best effort; may be smaller than min_frac
    else:
        target_frac = random.uniform(min_frac, max_allowed)

    target_w = max(1, int(round(bw * target_frac)))
    scale = target_w / iw
    target_h = max(1, int(round(ih * scale)))
    return roi.resize((target_w, target_h), Image.Resampling.LANCZOS)


def random_position_anywhere(bg_size: Tuple[int, int], obj_size: Tuple[int, int]) -> Tuple[int, int]:
    """Allows a random top-left that may place the ROI partially outside the canvas."""
    bw, bh = bg_size
    ow, oh = obj_size
    # allow top-left from -ow..bw-1 and -oh..bh-1
    x = random.randint(-ow + 1, bw - 1)
    y = random.randint(-oh + 1, bh - 1)
    return x, y


def coverage_of_position(bg_size: Tuple[int, int], obj_size: Tuple[int, int], xy: Tuple[int, int]) -> float:
    bw, bh = bg_size
    ow, oh = obj_size
    x, y = xy
    inter_w = max(0, min(x + ow, bw) - max(x, 0))
    inter_h = max(0, min(y + oh, bh) - max(y, 0))
    inter_area = inter_w * inter_h
    obj_area = max(1, ow * oh)
    return inter_area / obj_area


def choose_position_with_min_coverage(
    bg_size: Tuple[int, int], obj_size: Tuple[int, int], min_coverage: float = 2.0 / 3.0, attempts: int = 32
) -> Tuple[int, int]:
    """
    Try random off-canvas positions but require that at least `min_coverage` of ROI area
    is inside the background. Falls back to fully-inside placement if needed.
    """
    best_xy = None
    best_cov = -1.0
    for _ in range(max(1, attempts)):
        xy = random_position_anywhere(bg_size, obj_size)
        cov = coverage_of_position(bg_size, obj_size, xy)
        if cov > best_cov:
            best_cov, best_xy = cov, xy
        if cov >= min_coverage:
            return xy
    # fallback: place fully inside roughly at random
    bw, bh = bg_size
    ow, oh = obj_size
    x = 0 if ow >= bw else random.randint(0, max(0, bw - ow))
    y = 0 if oh >= bh else random.randint(0, max(0, bh - oh))
    return (x, y)


def paste_with_mask_clipped(
    bg: Image.Image, roi_rgba: Image.Image, mask: Image.Image, xy: Tuple[int, int]
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Pastes ROI possibly off-canvas, clipping to background bounds.
    Returns (bg, visible_box) where visible_box is the intersection box on bg (x0,y0,x1,y1).
    """
    if bg.mode != "RGB":
        bg = bg.convert("RGB")
    bw, bh = bg.size
    ow, oh = roi_rgba.size
    x, y = xy
    # compute intersection on bg
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(bw, x + ow)
    y1 = min(bh, y + oh)
    if x0 >= x1 or y0 >= y1:
        return bg, (0, 0, 0, 0)  # nothing visible
    # corresponding crop on ROI
    rx0 = x0 - x
    ry0 = y0 - y
    rx1 = rx0 + (x1 - x0)
    ry1 = ry0 + (y1 - y0)
    roi_c = roi_rgba.crop((rx0, ry0, rx1, ry1))
    mask_c = mask.crop((rx0, ry0, rx1, ry1))
    bg.paste(roi_c, (x0, y0), mask_c)
    return bg, (x0, y0, x1, y1)


def make_composites(
    backgrounds_dir: str = "backgrounds_640",
    rois_dir: str = "rois",
    out_images_dir: str = "dataset/images",
    out_labels_dir: str = "dataset/labels",
    num_images: int = 200,
    objects_per_image: int = 3,
    seed: int | None = None,
) -> None:
    ensure_dir(out_images_dir)
    ensure_dir(out_labels_dir)

    if seed is not None:
        random.seed(seed)

    backgrounds = list_images(backgrounds_dir)
    rois = list_images(rois_dir)
    if not backgrounds:
        raise SystemExit(f"Sem fundos em: {backgrounds_dir}")
    if not rois:
        raise SystemExit(f"Sem ROIs em: {rois_dir}")

    for idx in range(num_images):
        bg_path = random.choice(backgrounds)
        bg = Image.open(bg_path).convert("RGB")
        bw, bh = bg.size

        boxes: List[Tuple[float, float, float, float]] = []

        for _ in range(objects_per_image):
            roi_path = random.choice(rois)
            roi_rgba, mask = load_roi_with_mask(roi_path)
            # Scale to fit and add variety
            roi_rgba = scale_roi_to_fit(roi_rgba, (bw, bh), min_frac=0.18, max_frac=0.45)
            # Resize mask alongside
            mask = mask.resize(roi_rgba.size, Image.Resampling.NEAREST)
            ow, oh = roi_rgba.size
            # choose a position that keeps at least 2/3 visible
            x, y = choose_position_with_min_coverage((bw, bh), (ow, oh), min_coverage=2.0 / 3.0, attempts=32)
            bg, vis = paste_with_mask_clipped(bg, roi_rgba, mask, (x, y))
            vx0, vy0, vx1, vy1 = vis
            vw, vh = max(0, vx1 - vx0), max(0, vy1 - vy0)
            if vw > 0 and vh > 0:
                # YOLO bbox from the visible region only
                cx = (vx0 + vw / 2) / bw
                cy = (vy0 + vh / 2) / bh
                nw = vw / bw
                nh = vh / bh
                boxes.append((cx, cy, nw, nh))

        base_name = f"comp_{idx:04d}"
        img_out = os.path.join(out_images_dir, base_name + ".jpg")
        txt_out = os.path.join(out_labels_dir, base_name + ".txt")
        bg.save(img_out, format="JPEG", quality=92, optimize=True)
        with open(txt_out, "w", encoding="utf-8") as f:
            for (cx, cy, nw, nh) in boxes:
                f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        print(img_out)


def get_args():
    p = argparse.ArgumentParser(description="Gera composições com ROIs sobre fundos e labels YOLO.")
    p.add_argument("--backgrounds", "-b", default="backgrounds_640", help="Pasta de fundos")
    p.add_argument("--rois", "-r", default="rois", help="Pasta de objetos (ROIs)")
    p.add_argument("--out-images", default="dataset/images", help="Saída das imagens")
    p.add_argument("--out-labels", default="dataset/labels", help="Saída dos labels YOLO")
    p.add_argument("--count", "-c", type=int, default=200, help="Quantidade de imagens a gerar")
    p.add_argument("--objects", "-n", type=int, default=3, help="Objetos por imagem")
    p.add_argument("--seed", type=int, default=None, help="Seed aleatória (opcional)")
    return p.parse_args()


def main():
    args = get_args()
    make_composites(
        backgrounds_dir=args.backgrounds,
        rois_dir=args.rois,
        out_images_dir=args.out_images,
        out_labels_dir=args.out_labels,
        num_images=args.count,
        objects_per_image=args.objects,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

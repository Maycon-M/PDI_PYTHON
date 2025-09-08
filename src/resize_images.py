import os
import argparse
from typing import Tuple
from PIL import Image


def parse_size(size_str: str) -> Tuple[int, int]:
    if "x" in size_str.lower():
        w, h = size_str.lower().split("x", 1)
        return int(w), int(h)
    v = int(size_str)
    return v, v


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def letterbox(img: Image.Image, target_size: Tuple[int, int], bg_color=(0, 0, 0)) -> Image.Image:
    tw, th = target_size
    iw, ih = img.size
    if iw == 0 or ih == 0:
        return Image.new("RGB", (tw, th), bg_color)
    scale = min(tw / iw, th / ih)
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tw, th), bg_color)
    left = (tw - nw) // 2
    top = (th - nh) // 2
    canvas.paste(resized, (left, top))
    return canvas


def center_crop_resize(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    tw, th = target_size
    iw, ih = img.size
    # crop to match aspect ratio, then resize
    target_ratio = tw / th
    img_ratio = iw / ih if ih else 1
    if img_ratio > target_ratio:
        # wider than target: crop width
        new_w = int(ih * target_ratio)
        left = (iw - new_w) // 2
        box = (max(0, left), 0, min(iw, left + new_w), ih)
    else:
        # taller than target: crop height
        new_h = int(iw / target_ratio) if target_ratio else ih
        top = (ih - new_h) // 2
        box = (0, max(0, top), iw, min(ih, top + new_h))
    cropped = img.crop(box)
    return cropped.resize((tw, th), Image.Resampling.LANCZOS)


def stretch_resize(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    return img.resize(target_size, Image.Resampling.LANCZOS)


def process_image(path: str, out_path: str, target_size: Tuple[int, int], mode: str, bg_color=(0, 0, 0)) -> None:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if mode == "letterbox":
            out = letterbox(im, target_size, bg_color)
        elif mode == "crop":
            out = center_crop_resize(im, target_size)
        elif mode == "stretch":
            out = stretch_resize(im, target_size)
        else:
            raise ValueError(f"Modo desconhecido: {mode}")
        # pick format based on extension
        ext = os.path.splitext(out_path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            out.save(out_path, format="JPEG", quality=92, optimize=True)
        elif ext == ".png":
            out.save(out_path, format="PNG", optimize=True)
        else:
            # default to jpg
            base, _ = os.path.splitext(out_path)
            out.save(base + ".jpg", format="JPEG", quality=92, optimize=True)


def iter_images(in_dir: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for root, _dirs, files in os.walk(in_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                yield os.path.join(root, f)


def resize_images(
    in_dir: str = "backgrounds",
    out_dir: str = "backgrounds_640",
    size: Tuple[int, int] = (640, 640),
    mode: str = "letterbox",
    bg_color=(0, 0, 0),
    in_place: bool = False,
):
    if in_place:
        out_dir = in_dir
    else:
        ensure_dir(out_dir)

    tw, th = size
    count = 0
    for src in iter_images(in_dir):
        rel = os.path.relpath(src, in_dir)
        dst = os.path.join(out_dir, rel)
        ensure_dir(os.path.dirname(dst))
        try:
            process_image(src, dst, (tw, th), mode, bg_color)
            print(dst)
            count += 1
        except Exception as e:
            print(f"erro ao processar {src}: {e}")
    print(f"total processado: {count}")


def get_args():
    p = argparse.ArgumentParser(description="Redimensiona imagens para um tamanho alvo.")
    p.add_argument("--input", "-i", default="backgrounds", help="Pasta de entrada")
    p.add_argument("--output", "-o", default="backgrounds_640", help="Pasta de saída (ignorado com --in-place)")
    p.add_argument("--size", "-s", default="640x640", help="Tamanho alvo, ex: 640 ou 640x640")
    p.add_argument(
        "--mode",
        "-m",
        choices=["letterbox", "crop", "stretch"],
        default="crop",
        help="Estratégia: letterbox (mantém proporção com barras), crop (corta central), stretch (deforma)",
    )
    p.add_argument("--bg", default="0,0,0", help="Cor de fundo RGB para letterbox, ex: 0,0,0 ou 255,255,255")
    p.add_argument("--in-place", action="store_true", help="Grava por cima (cuidado)")
    return p.parse_args()


def main():
    args = get_args()
    size = parse_size(args.size)
    try:
        bg_tuple = tuple(int(x) for x in args.bg.split(","))
        if len(bg_tuple) != 3:
            raise ValueError
    except Exception:
        raise SystemExit("--bg deve ser algo como 0,0,0")

    resize_images(
        in_dir=args.input,
        out_dir=args.output,
        size=size,
        mode=args.mode,
        bg_color=bg_tuple,
        in_place=args.in_place,
    )


if __name__ == "__main__":
    main()

# extract_rois_rembg_maskblend.py
from __future__ import annotations
import os
import cv2
import numpy as np
from rembg import remove, new_session

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# ---------------- Config ----------------
INPUT_DIR      = "./motorcycle_raw"
OUTPUT_DIR     = "./rois"
SINGLE_OBJECT  = True      # True = 1 ROI (moto inteira); False = várias
MIN_AREA       = 4000
PAD            = 12
FEATHER_R      = 2         # suaviza borda do alfa (0 desliga)

# Modelos rembg (fallback em cascata)
_MODEL_CANDIDATES = ["isnet-general-use", "u2net", "u2netp"]

def _get_session():
    last_err = None
    for name in _MODEL_CANDIDATES:
        try:
            print(f"[rembg] tentando modelo: {name}")
            return new_session(name)
        except Exception as e:
            last_err = e
            print(f"[rembg] falhou '{name}': {e}")
    raise RuntimeError(f"Não foi possível criar sessão rembg. Último erro: {last_err}")

SESSION = _get_session()

# -------------- Helpers -----------------
def feather_alpha(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0: return mask
    inv = 255 - mask
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    dist = np.clip(dist, 0, radius).astype(np.float32) / max(1, radius)
    alpha = (mask.astype(np.float32) / 255.0) * 255.0
    alpha = np.clip(alpha - (dist * 255), 0, 255).astype(np.uint8)
    return alpha

def get_rembg_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Pede ao rembg APENAS a MÁSCARA (grayscale 0-255) e devolve np.uint8.
    Fazemos a composição de cores manualmente para preservar a imagem original.
    """
    # Entrada correta: RGB/PIL
    if _HAS_PIL:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_in = Image.fromarray(rgb)
        # pede só a máscara + pós-processamento de borda do rembg
        mask_out = remove(
            pil_in, session=SESSION,
            only_mask=True, post_process_mask=True,
            alpha_matting=False  # deixe False; ativa só se precisar refino extra
        )
        # Convert para numpy (garante 0-255)
        if "PIL" in str(type(mask_out)):
            mask = np.array(mask_out.convert("L"))
        else:
            mask = np.array(mask_out)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = remove(
            rgb, session=SESSION,
            only_mask=True, post_process_mask=True,
            alpha_matting=False
        )
        if isinstance(mask, (bytes, bytearray)):
            buf = np.frombuffer(mask, np.uint8)
            mask = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

    # Normaliza para uint8
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)

    return mask

# --------------- Core -------------------
def process_image(path: str, fname: str):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"[ERRO] Falha ao ler {path}")
        return
    base = os.path.splitext(fname)[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Máscara pelo rembg (sem mexer nas cores)
    mask = get_rembg_mask(bgr)

    # 2) Limpeza BEM leve da máscara (sem destruir detalhes claros)
    mask = cv2.medianBlur(mask, 3)
    # opcional: fecho pequenos gaps sem encolher objeto
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), 1)

    # 3) Alfa com feather para evitar halo
    alpha = feather_alpha(mask, FEATHER_R)

    # 4) Composição manual: original + alfa
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha

    h, w = alpha.shape

    if SINGLE_OBJECT:
        ys, xs = np.where(alpha > 5)
        if xs.size == 0:
            print(f"[WARN] Sem frente detectada: {fname}")
            return
        x0, x1 = max(0, xs.min() - PAD), min(w, xs.max() + PAD)
        y0, y1 = max(0, ys.min() - PAD), min(h, ys.max() + PAD)
        crop = bgra[y0:y1, x0:x1].copy()
        out = os.path.join(OUTPUT_DIR, f"{base}_ROI_0001.png")
        cv2.imwrite(out, crop)
        print(f"[OK] {fname}: ROI única salva.")
        return

    # múltiplas ROIs
    comps = (alpha > 5).astype(np.uint8)
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(comps, connectivity=8)
    saved = 0
    for i, st in enumerate(stats[1:], start=1):
        x, y, bw, bh, area = st
        if area < MIN_AREA: continue
        x0, y0 = max(0, x - PAD), max(0, y - PAD)
        x1, y1 = min(w, x + bw + PAD), min(h, y + bh + PAD)
        crop = bgra[y0:y1, x0:x1].copy()
        comp_mask = (lbl == i).astype(np.uint8) * 255
        crop[:, :, 3] = cv2.bitwise_and(alpha[y0:y1, x0:x1], comp_mask[y0:y1, x0:x1])
        out = os.path.join(OUTPUT_DIR, f"{base}_ROI_{saved+1:04d}.png")
        cv2.imwrite(out, crop)
        saved += 1
    print(f"[OK] {fname}: {saved} ROI(s) salvas.")

def run():
    if not os.path.isdir(INPUT_DIR):
        print(f"[ERRO] Diretório não encontrado: {INPUT_DIR}")
        return
    for fname in os.listdir(INPUT_DIR):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
            process_image(os.path.join(INPUT_DIR, fname), fname)

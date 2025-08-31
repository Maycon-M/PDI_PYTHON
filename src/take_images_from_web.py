# -*- coding: utf-8 -*-
"""
Lê src/models.json (array de "Marca Modelo") e, para cada item:
- Busca no Bing (icrawler) focando fundo branco
- Filtra (fundo branco + conteúdo)
- Salva como MARCA_MODELO_0001.jpg, MARCA_MODELO_0002.jpg, ...
"""

import os, sys, re, time, json, traceback, unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import cv2
import numpy as np
from PIL import Image, ImageOps
from icrawler.builtin import BingImageCrawler

from urllib.parse import urlparse
from icrawler import ImageDownloader

# ==== Paths ====
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_JSON = os.path.join(HERE, "models.json")

BASE_DIR = os.path.abspath(os.path.join(HERE, ".."))
RAW_DIR = os.path.join(BASE_DIR, "motorcycle_raw")
FILTERED_DIR = os.path.join(BASE_DIR, "motorcycle_filtered")
META_CSV = os.path.join(FILTERED_DIR, "metadata.csv")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(FILTERED_DIR, exist_ok=True)

# ==== Config ====
MAX_PER_MODEL = 20      # até 1000 por limitação do Bing; ajuste se quiser
THREADS = 8              # triagem paralela por modelo
NEGATIVE_TERMS = "-silhouette -vector -illustration -outline -drawing -render -clipart"
BING_FILTER_CANDIDATES = [
    {"type": "photo", "size": "large"},
    {"type": "photo"},
    {},
]

# --- DOWNLOADER COM PREFIXO (MARCA_MODELO) ---
class PrefixNameDownloader(ImageDownloader):
    def __init__(self, *args, prefix="IMG", **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = prefix  # Ex.: Yamaha_YZF_R9

    def get_filename(self, task, default_ext):
        # tenta extensão da URL; cai para default se não conhecida
        url_path = urlparse(task['file_url']).path
        ext = os.path.splitext(url_path)[1].lower()
        if ext.replace(".", "") not in {"jpg","jpeg","png","bmp","tiff","gif","ppm","pgm","webp"}:
            ext = f".{default_ext}" if not ext else ext  # garante algo
        # >>> ESSA é a parte importante: use fetched_num + file_idx_offset <<<
        idx = self.fetched_num + self.file_idx_offset
        return f"{self.prefix}_{idx:04d}{ext or '.jpg'}"

def _set_prefix_and_offset(crawler, prefix: str):
    """Define prefixo e ajusta o offset inicial com base no que já existe no RAW."""
    crawler.downloader.prefix = prefix
    # procura o maior índice já salvo com este prefixo
    pat = re.compile(rf"^{re.escape(prefix)}_(\d{{4}})\.(jpg|jpeg|png|webp|bmp|tiff|gif|ppm|pgm)$", re.I)
    max_idx = 0
    for fname in os.listdir(RAW_DIR):
        m = pat.match(fname)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    # agora diz ao downloader para começar do próximo número
    crawler.downloader.set_file_idx_offset(max_idx)  # API oficial do icrawler


# ==== Nomes/Numeração thread-safe por prefixo ====
_seq_global_lock = Lock()
_seq_inited = set()
_seq_counters = {}  # prefix -> last_int

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def sanitize_component(s: str) -> str:
    s = _strip_accents(s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)  # troca não-alfanum por _
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def next_seq_for_prefix(prefix: str) -> int:
    with _seq_global_lock:
        if prefix not in _seq_inited:
            # varre existentes para definir ponto de partida
            pat = re.compile(rf"^{re.escape(prefix)}_(\d{{4}})\.jpg$", re.IGNORECASE)
            mx = 0
            for fname in os.listdir(FILTERED_DIR):
                m = pat.match(fname)
                if m:
                    mx = max(mx, int(m.group(1)))
            _seq_counters[prefix] = mx
            _seq_inited.add(prefix)
        _seq_counters[prefix] += 1
        return _seq_counters[prefix]

# ==== Dedup global por pHash ====
_hash_lock = Lock()
_seen_hashes = set()
try:
    import imagehash as ih
except Exception:
    ih = None

def compute_phash(pil_img):
    if ih is None:
        return None
    return str(ih.phash(pil_img))

# ==== Imagem utils e filtros (mesmos do pipeline anterior) ====
def pil_open_rgb(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        return im

def pil_to_np_rgb(pil_img): return np.array(pil_img)

def rgb2hsv01(arr_rgb):
    arr = arr_rgb.astype(np.float32) / 255.0
    return cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

def estimate_bg_from_border(img_rgb, border_frac=0.03):
    h, w, _ = img_rgb.shape
    b = max(1, int(border_frac * min(h, w)))
    mask = np.zeros((h, w), dtype=bool)
    mask[:b, :] = True; mask[-b:, :] = True; mask[:, :b] = True; mask[:, -b:] = True
    border_pixels = img_rgb[mask]
    bg_rgb = np.median(border_pixels, axis=0).clip(0,255).astype(np.uint8)
    return bg_rgb

def background_is_white_like(bg_rgb, s_thresh=0.25, v_thresh=0.88):
    hsv = rgb2hsv01(bg_rgb.reshape(1,1,3))[0,0]
    return (hsv[1] <= s_thresh) and (hsv[2] >= v_thresh)

def build_bg_mask(img_rgb, bg_rgb, tol=35):
    diff = np.linalg.norm(img_rgb.astype(np.int16) - bg_rgb.astype(np.int16), axis=2)
    rgb_similar = diff <= tol
    hsv = rgb2hsv01(img_rgb)
    low_sat = hsv[...,1] <= 0.18
    high_val = hsv[...,2] >= 0.90
    return rgb_similar & low_sat & high_val

def large_contiguous_bg_from_borders(bg_mask, min_ratio=0.55):
    h, w = bg_mask.shape
    seeds = np.zeros_like(bg_mask, dtype=np.uint8)
    seeds[0, :] = bg_mask[0, :]
    seeds[-1, :] = bg_mask[-1, :]
    seeds[:, 0] |= bg_mask[:, 0]
    seeds[:, -1] |= bg_mask[:, -1]
    _, labels = cv2.connectedComponents(bg_mask.astype(np.uint8), connectivity=4)
    border_labels = np.unique(labels[seeds.astype(bool)])
    if len(border_labels) == 0: return False
    best = 0
    for lb in border_labels:
        if lb == 0: continue
        area = int(np.sum(labels == lb))
        if area > best: best = area
    return best / float(h*w) >= min_ratio

def is_white_bg_photo(path):
    try:
        pil = pil_open_rgb(path)
        img = pil_to_np_rgb(pil)
        bg_rgb = estimate_bg_from_border(img, border_frac=0.03)
        if not background_is_white_like(bg_rgb): return False
        bg_mask = build_bg_mask(img, bg_rgb, tol=35)
        if not large_contiguous_bg_from_borders(bg_mask, min_ratio=0.55): return False
        # rejeita letterbox
        h, w = bg_mask.shape
        border = max(1, int(0.02 * min(h, w)))
        if np.mean(bg_mask[:border,:]) < 0.5 or np.mean(bg_mask[-border:,:]) < 0.5:
            return False
        return True
    except Exception:
        return False

def is_monochrome_or_lineart(img_rgb, sat_mean_max=0.12, edge_ratio_min=0.18):
    hsv = rgb2hsv01(img_rgb)
    if float(np.mean(hsv[...,1])) <= sat_mean_max: return True
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return float(np.mean(edges > 0)) >= edge_ratio_min

def is_silhouette(img_rgb, dark_thresh=0.15, white_thresh=0.92, min_dark_ratio=0.35, min_white_ratio=0.40):
    hsv = rgb2hsv01(img_rgb)
    dark = hsv[...,2] <= dark_thresh
    white = (hsv[...,2] >= white_thresh) & (hsv[...,1] <= 0.18)
    return (np.mean(dark) >= min_dark_ratio) and (np.mean(white) >= min_white_ratio)

def looks_like_cg_render(img_rgb, blur_sigma=1.2, var_thr=2.0, sat_thr=0.6):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(cv2.GaussianBlur(gray, (0,0), blur_sigma), cv2.CV_32F)
    hi_var = float(np.var(lap))
    hsv = rgb2hsv01(img_rgb)
    sat_mean = float(np.mean(hsv[...,1]))
    return (hi_var < var_thr) and (sat_mean > sat_thr)

def dominant_big_circle_like(img_rgb, min_circle_ratio=0.36):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h,w)//2,
        param1=120, param2=60,
        minRadius=int(0.25*min(h,w)//2), maxRadius=int(0.9*min(h,w)//2)
    )
    if circles is None: return False
    r_max = np.max(circles[0,:,2])
    area_ratio = np.pi * (r_max**2) / float(h*w)
    return area_ratio >= min_circle_ratio

def passes_content_filter(pil_img):
    img = pil_to_np_rgb(pil_img)
    if is_monochrome_or_lineart(img):   return False
    if is_silhouette(img):              return False
    if looks_like_cg_render(img):       return False
    if dominant_big_circle_like(img):   return False
    return True

# ==== Bing downloader ====
def download_bing_for_query(query: str, max_num: int, brand: str, model: str):
    prefix = f"{sanitize_component(brand)}_{sanitize_component(model)}"
    crawler = BingImageCrawler(
        storage={"root_dir": RAW_DIR},
        downloader_cls=PrefixNameDownloader,
        downloader_threads=4,
    )
    _set_prefix_and_offset(crawler, prefix)

    for f in [{"type":"photo","size":"large"}, {"type":"photo"}, {}]:
        try:
            crawler.crawl(keyword=query, max_num=max_num, filters=f)
            return
        except ValueError:
            continue
    crawler.crawl(keyword=query, max_num=max_num)

# ==== Worker por arquivo (classifica + salva) ====
def process_one_file(src_path: str, brand: str, model: str):
    try:
        if not is_white_bg_photo(src_path):
            return None
        pil = pil_open_rgb(src_path)
        if not passes_content_filter(pil):
            return None

        ph = compute_phash(pil)
        if ph is not None:
            with _hash_lock:
                if ph in _seen_hashes:
                    return None
                _seen_hashes.add(ph)

        brand_s = sanitize_component(brand)
        model_s = sanitize_component(model)
        prefix = f"{brand_s}_{model_s}"
        seq = next_seq_for_prefix(prefix)
        out_name = f"{prefix}_{seq:04d}.jpg"
        out_path = os.path.join(FILTERED_DIR, out_name)
        pil.save(out_path, format="JPEG", quality=92, subsampling=1, optimize=True)

        return (out_name, os.path.basename(src_path), ph or "", pil.width, pil.height)
    except Exception:
        return None

# ==== Helpers ====
KNOWN_BRANDS = ["Royal Enfield", "Harley-Davidson", "Yamaha", "Honda", "BMW", "Ducati", "Triumph", "Suzuki", "Kawasaki", "KTM", "Aprilia"]
# ordenar por tamanho desc. para acertar prefixos com 2 palavras
KNOWN_BRANDS.sort(key=len, reverse=True)

def split_brand_model(item: str):
    s = item.strip()
    for b in KNOWN_BRANDS:
        if s.lower().startswith(b.lower() + " "):
            return b, s[len(b):].strip()
        if s.lower() == b.lower():
            return b, ""
    # fallback
    parts = s.split(" ", 1)
    return (parts[0], parts[1] if len(parts) > 1 else "")

def build_bing_query(brand: str, model: str) -> str:
    # frase exata para marca+modelo + foco em estúdio/fundo branco + termos negativos
    core = f"\"{brand} {model}\"".strip()
    extras = "on white background"
    return f"{core} {extras} {NEGATIVE_TERMS}".strip()

def load_models(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("models.json deve ser um array de strings.")
    return [str(x).strip() for x in data if str(x).strip()]

# ==== Main ====
def take_images_from_web():
    # carrega modelos
    models = load_models(MODELS_JSON)
    print(f"{len(models)} modelos carregados de {MODELS_JSON}")

    # CSV header
    if not os.path.exists(META_CSV):
        with open(META_CSV, "w", encoding="utf-8", newline="") as f:
            import csv
            csv.writer(f).writerow(["out_name","src_name","phash","width","height"])

    for item in models:
        brand, model = split_brand_model(item)
        if not model:
            print(f"Ignorando entrada sem modelo: {item}")
            continue

        query = build_bing_query(brand, model)
        print(f"\n🟦 Bing: {brand} | {model}")
        print(f"🔎 {query}")

        # baixa só o que vier nesta rodada
        before = set(os.listdir(RAW_DIR))
        download_bing_for_query(query, MAX_PER_MODEL, brand, model)
        time.sleep(1.0)
        after = set(os.listdir(RAW_DIR))
        new_files = [os.path.join(RAW_DIR, f) for f in (after - before)]

        # triagem paralela + salvamento já com MARCA_MODELO_seq
        rows = []
        with ThreadPoolExecutor(max_workers=THREADS) as ex:
            futs = [ex.submit(process_one_file, p, brand, model) for p in new_files]
            for fut in as_completed(futs):
                r = fut.result()
                if r:
                    rows.append(r)

        # anexa metadados
        if rows:
            with open(META_CSV, "a", encoding="utf-8", newline="") as f:
                import csv
                csv.writer(f).writerows(rows)

        print(f"{brand} {model}: {len(rows)} imagens aprovadas nesta rodada.")

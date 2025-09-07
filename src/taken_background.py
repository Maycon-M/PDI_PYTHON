# -*- coding: utf-8 -*-
import os, requests, time, random
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

load_dotenv()
ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

SAVE_DIR = "backgrounds"
os.makedirs(SAVE_DIR, exist_ok=True)

QUERIES = [
    "empty motorcycle track daylight",
    "motorcycle race track empty night lights",
    "curved mountain road empty",
    "long desert highway empty",
    "empty coastal road cliffside",
    "rural road with trees empty",
    "empty parking garage underground",
    "empty rooftop parking lot",
    "urban street at night empty",
    "wide avenue sunrise empty",
    "empty bridge road cityscape",
    "industrial alley empty",
    "forest road empty misty morning",
    "mountain pass empty road",
    "empty dirt road countryside",
    "abandoned rural road with grass",
    "snowy road empty landscape",
    "abandoned warehouse interior",
    "empty hangar interior",
    "old gas station empty night",
    "industrial garage empty",
    "empty mechanic workshop",
    "abandoned factory interior",
    "neon lit street empty night rain",
    "foggy road empty morning",
    "sunset highway empty scene",
    "night city empty street long exposure",
    "moonlit rural road empty",
]

def search_unsplash(query, per_page=10):
    url = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {ACCESS_KEY}"}
    params = {
        "query": query,
        "orientation": "landscape",
        "per_page": per_page,
    }
    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()["results"]

def download_image(url, out_path):
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    img.save(out_path, format="JPEG", quality=92, optimize=True)

def take_background():
    for q in QUERIES:
        results = search_unsplash(q, per_page=8)
        safe_q = "".join(c if c.isalnum() else "_" for c in q).strip("_")
        for i, r in enumerate(results):
            img_url = r["urls"]["full"]
            fname = f"{safe_q}_{i:03d}.jpg"
            out_path = os.path.join(SAVE_DIR, fname)
            try:
                download_image(img_url, out_path)
                print(f"{fname}")
                time.sleep(random.uniform(0.4, 1.2))
            except Exception as e:
                print(f"erro em {fname}: {e}")

if __name__ == "__main__":
    take_background()

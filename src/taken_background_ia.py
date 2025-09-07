import os, base64, time, random
from openai import OpenAI

QUERIES = [
    "Posto de gasolina durante o dia",
    "Posto de gasolina durante a noite",
    "Posto de gasolina durante o entardecer",
    "Posto de gasolina durante a chuva",
]

NEGATIVE = (
    "no motorcycle, no bicycle, no motorbike, no watermark"
)

STYLE_GUIDE = (
    "photorealistic, natural lighting, sharp details, pro photography, "
    "clean composition, neutral color grading"
)


# Parâmetros úteis:
SIZE = "1536x1024"   # paisagem (ótimo p/ motos)
QUALITY = "medium"   # custo/qualidade equilibrado
N_PER_QUERY = 3


def prompt_for(q: str) -> str:
    return (
        f"{q} — empty scene. {NEGATIVE}. {STYLE_GUIDE}. "
        "wide angle, horizon centered. This is a realistic empty background for product compositing."
    )
    
def take_background():
    client = OpenAI()  # lê OPENAI_API_KEY do ambiente

    os.makedirs("./backgrounds", exist_ok=True)
    for q in QUERIES:
        print(f"Tirando fotos de fundo para: '{q}'\n")
        for i in range(N_PER_QUERY):
            print(f"  {i+1}/{N_PER_QUERY} ... \n", end="", flush=True)
            resp = client.images.generate(
                model="gpt-image-1",
                prompt=prompt_for(q),
                size=SIZE,
                quality=QUALITY,   # "low" | "medium" | "high"
                n=1,
                # background="transparent"  # só para PNG com alpha em alguns casos
            )
            b64 = resp.data[0].b64_json
            img = base64.b64decode(b64)
            safe_q = "".join(c if c.isalnum() else "_" for c in q).strip("_")
            out = f"backgrounds/{safe_q}_{i:02d}.jpg"
            with open(out, "wb") as f:
                f.write(img)
            # pequena pausa ajuda a estabilidade
            time.sleep(60 + random.uniform(+10, +20))
        time.sleep(120)

    print("Concluído.")

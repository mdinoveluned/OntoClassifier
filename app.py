# app.py
import os
import re
import time
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

# -----------------------------
# Config por entorno (opcional)
# -----------------------------
CSV_PATH = os.environ.get("ONTO_CSV_PATH", "categorias.csv")
API_KEY  = os.environ.get("ONTO_API_KEY", "")           # si no está -> sin auth
PUBLIC_URL = os.environ.get("PUBLIC_URL", "")           # si lo pones, se añade a OpenAPI

# Pesos por defecto de las señales (puedes sobreescribir vía query en /classify)
DEFAULT_W_COS   = float(os.environ.get("W_COS", 0.65))
DEFAULT_W_FUZZ  = float(os.environ.get("W_FUZZ", 0.25))
DEFAULT_W_OV    = float(os.environ.get("W_OV", 0.10))
DEFAULT_DECISION_THRESH = float(os.environ.get("DECISION_THRESH", 0.62))
DEFAULT_TOP = int(os.environ.get("TOP", 5))
DEFAULT_K   = int(os.environ.get("TOPK", 30))

# -----------------------------
# Utilidades
# -----------------------------
def norm(s: str) -> str:
    if s is None: 
        return ""
    s = s.strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", " ", s)
    return s

def main_label(cat: str) -> str:
    # Toma la parte principal antes de " / " (ej. "bueno y perfecto / malo e imperfecto" -> "bueno y perfecto")
    return (cat or "").split(" / ")[0].strip()

def token_overlap(a: str, b: str) -> float:
    # compatible con re estándar (sin \p{L}); simple y efectiva
    A = set(re.findall(r"\w+", a.lower()))
    B = set(re.findall(r"\w+", b.lower()))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def require_api_key(x_api_key: Optional[str]):
    if not API_KEY:
        return  # sin clave definida -> abierto
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# -----------------------------
# Carga de datos y modelo
# -----------------------------
class OntoIndex:
    def __init__(self, csv_path: str):
        self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.model: Optional[SentenceTransformer] = None
        self.df: Optional[pd.DataFrame] = None
        self.cat_emb: Optional[np.ndarray] = None
        self.load(csv_path)

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def load(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise RuntimeError(f"No se encontró el CSV en: {csv_path}")
        df = pd.read_csv(csv_path)
        if not {"id", "categoria"}.issubset(df.columns):
            raise RuntimeError("El CSV debe tener columnas 'id' y 'categoria'.")

        df["categoria_main"] = df["categoria"].astype(str).map(main_label)
        df["categoria_norm"] = df["categoria_main"].map(norm)
        df = df.reset_index(drop=True)

        self.df = df

        # Modelo + embeddings
        self._load_model()
        cat_texts = df["categoria_main"].tolist()
        # Normalizamos embeddings para que coseno sea producto punto
        emb = self.model.encode(cat_texts, normalize_embeddings=True)
        self.cat_emb = np.asarray(emb, dtype=np.float32)

    def _cosine_topk(self, vec: np.ndarray, k: int):
        # vec: (d,) normalizado; mat: (N,d) normalizado -> cos = dot
        sims = self.cat_emb @ vec  # (N,)
        k = min(k, sims.shape[0])
        idx = np.argpartition(-sims, kth=k-1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return idx, sims[idx]

    def score_candidates(self, text: str, k: int, w_cos: float, w_fuzzy: float, w_ov: float):
        assert self.df is not None and self.cat_emb is not None and self.model is not None
        # embedding del texto
        vec = self.model.encode([text], normalize_embeddings=True)[0].astype(np.float32)

        idxs, sims = self._cosine_topk(vec, k=k)

        # calcular señales
        out = []
        for i, cosv in zip(idxs, sims):
            row = self.df.iloc[i]
            cat_main = row["categoria_main"]
            fuzzy = fuzz.WRatio(text.lower(), cat_main.lower()) / 100.0  # 0..1
            ov = token_overlap(text, cat_main)                           # 0..1
            final = w_cos*float(cosv) + w_fuzzy*float(fuzzy) + w_ov*float(ov)
            out.append({
                "id": int(row["id"]),
                "categoria": row["categoria_main"],
                "cos": float(cosv),
                "fuzzy": round(float(fuzzy), 3),
                "overlap": round(float(ov), 3),
                "final": round(final, 3)
            })

        out.sort(by=lambda x: x["final"])
        out.sort(key=lambda x: x["final"], reverse=True)
        return out

    def classify(self, text: str, top: int, decision_thresh: float,
                 w_cos: float, w_fuzzy: float, w_ov: float, k: int):
        cands = self.score_candidates(text, k=k, w_cos=w_cos, w_fuzzy=w_fuzzy, w_ov=w_ov)
        topc = cands[:top]
        best = topc[0] if topc else None
        if not best or best["final"] < decision_thresh:
            return {
                "input": text,
                "label_id": None,
                "label_text": "OUT_OF_SCOPE",
                "confidence": "low",
                "score_final": 0.0,
                "alternativas": topc
            }
        conf = "high" if best["final"] >= 0.75 else ("medium" if best["final"] >= 0.65 else "low")
        # devuelve también topc para depuración (puedes quitarlo si quieres respuesta mínima)
        return {
            "input": text,
            "label_id": best["id"],
            "label_text": best["categoria"],
            "confidence": conf,
            "score_final": best["final"],
            "alternativas": topc
        }

onto = OntoIndex(CSV_PATH)

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="OntoClassifier (Hybrid)", version="1.0.0")

# CORS amplio por sencillez (ajusta si lo necesitas)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Inyecta "servers" en OpenAPI para que el GPT Builder no se queje
def custom_openapi():
    if app.openapi_schema:
        # actualiza servers en caliente
        if PUBLIC_URL:
            app.openapi_schema["servers"] = [{"url": PUBLIC_URL}]
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )
    if PUBLIC_URL:
        openapi_schema["servers"] = [{"url": PUBLIC_URL}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "rows": int(onto.df.shape[0]) if onto.df is not None else 0,
        "model": onto.model_name,
        "csv": CSV_PATH
    }

@app.get("/classify")
def classify(
    q: str = Query(..., description="Voz o acepción a clasificar"),
    top: int = Query(DEFAULT_TOP, ge=1, le=20),
    decision_thresh: float = Query(DEFAULT_DECISION_THRESH, ge=0.0, le=1.0),
    w_cos: float = Query(DEFAULT_W_COS, ge=0.0, le=1.0),
    w_fuzzy: float = Query(DEFAULT_W_FUZZ, ge=0.0, le=1.0),
    w_ov: float = Query(DEFAULT_W_OV, ge=0.0, le=1.0),
    k: int = Query(DEFAULT_K, ge=5, le=100),
    x_api_key: Optional[str] = Header(None)
):
    require_api_key(x_api_key)
    text = q.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Parametro q vacío")
    # normaliza pesos si no suman 1 (opcional, por estabilidad)
    s = w_cos + w_fuzzy + w_ov
    if s == 0:
        w_cos, w_fuzzy, w_ov = DEFAULT_W_COS, DEFAULT_W_FUZZ, DEFAULT_W_OV
        s = w_cos + w_fuzzy + w_ov
    w_cos, w_fuzzy, w_ov = w_cos/s, w_fuzzy/s, w_ov/s

    return onto.classify(text, top=top, decision_thresh=decision_thresh,
                         w_cos=w_cos, w_fuzzy=w_fuzzy, w_ov=w_ov, k=k)

@app.post("/reload")
def reload_index(x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
    t0 = time.time()
    onto.load(CSV_PATH)
    return {"status": "reloaded", "rows": int(onto.df.shape[0]), "took_s": round(time.time()-t0, 3)}
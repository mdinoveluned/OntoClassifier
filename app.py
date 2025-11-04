import os
import re
import unicodedata
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from fastapi.openapi.utils import get_openapi

# ---------------------------
# Configuración
# ---------------------------

CATEGORIES_CSV = os.getenv("CATEGORIES_CSV", "categorias.csv")
MODEL_NAME = os.getenv("EMBEDDINGS_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://delma-unevangelic-alvera.ngrok-free.dev")

ALLOWED_ORIGINS = [
    PUBLIC_BASE_URL,
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000"
]

TAXON_HINTS = {
    "ave": {"aliases": ["ave", "aves", "pájaro", "pájaros", "aviar", "ornitología", "ornitológico"], "boost": 0.12},
    "árbol": {"aliases": ["árbol", "árboles", "arborícola", "arboricultura", "árboreo"], "boost": 0.12},
    "planta": {"aliases": ["planta", "plantas", "vegetal", "vegetales", "botánica", "botánico", "flora"], "boost": 0.12},
    "pez": {"aliases": ["pez", "peces", "ictiología", "ictiológico", "piscícola"], "boost": 0.10},
    "reptil": {"aliases": ["reptil", "reptiles", "herpetología", "herpetológico"], "boost": 0.10},
    "anfibio": {"aliases": ["anfibio", "anfibios"], "boost": 0.08},
    "insecto": {"aliases": ["insecto", "insectos", "entomología", "entomológico"], "boost": 0.10},
    "arbusto": {"aliases": ["arbusto", "arbustos", "matorral"], "boost": 0.08},
    "hongo": {"aliases": ["hongo", "hongos", "micología", "fúngico"], "boost": 0.08}
}

def normalize(text: str) -> str:
    if text is None:
        return ""
    text = text.strip().lower()
    text = ''.join(ch for ch in unicodedata.normalize('NFKD', text)
                   if not unicodedata.combining(ch))
    return text

def safe_get(row: pd.Series, key: str) -> str:
    return str(row[key]) if key in row and pd.notna(row[key]) else ""

def split_acepciones_respecting_quotes(s: str) -> List[str]:
    if s is None:
        return []
    parts = []
    current = []
    in_quotes = False
    i = 0
    while i < len(s):
        if s[i] == '"':
            in_quotes = not in_quotes
            current.append(s[i])
            i += 1
            continue
        if not in_quotes and s[i:i+2] == '||':
            part = ''.join(current).strip()
            parts.append(part)
            current = []
            i += 2
            continue
        current.append(s[i])
        i += 1
    part = ''.join(current).strip()
    if part:
        parts.append(part)
    cleaned = []
    for p in parts:
        p = p.strip()
        if len(p) >= 2 and p[0] == '"' and p[-1] == '"':
            cleaned.append(p[1:-1])
        else:
            cleaned.append(p)
    return cleaned

class CategoryIndex:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.model = SentenceTransformer(MODEL_NAME)
        self.df = None
        self.cat_texts = []
        self.emb = None
        self._load()

    def _load(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"No se encontró {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        for col in ["id", "categoria", "aliases"]:
            if col not in df.columns:
                raise ValueError("El CSV debe tener columnas: id, categoria, aliases")
        cat_texts = []
        for _, row in df.iterrows():
            name = safe_get(row, "categoria")
            aliases = safe_get(row, "aliases").replace("|", " ").strip()
            text = " | ".join([name, aliases]).strip(" |")
            cat_texts.append(text if text else name)
        emb = self.model.encode(cat_texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        self.df = df.reset_index(drop=True)
        self.cat_texts = cat_texts
        self.emb = emb

    def reload(self):
        self._load()

    def rank(self, query_text: str, taxon_hints: Dict[str, Any], top_n: int = 10) -> List[Dict[str, Any]]:
        q = query_text.strip()
        if not q:
            return []
        q_emb = self.model.encode([q], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
        cosine = np.dot(self.emb, q_emb)
        boosts = np.zeros_like(cosine)
        for i, row in self.df.iterrows():
            base_text = f"{safe_get(row,'categoria')} {safe_get(row,'aliases').replace('|',' ')}".strip()
            if base_text:
                fuzzy = fuzz.partial_ratio(normalize(q), normalize(base_text)) / 100.0
                boosts[i] += 0.05 * fuzzy
        q_norm = " " + normalize(q) + " "
        for i, row in self.df.iterrows():
            name = " " + normalize(safe_get(row, "categoria")) + " "
            aliases = " " + normalize(safe_get(row, "aliases").replace("|", " ")) + " "
            cat_blob = name + aliases
            for _, info in taxon_hints.items():
                alias_list = info.get("aliases", [])
                if any((" " + normalize(a) + " ") in q_norm for a in alias_list):
                    if any((" " + normalize(a) + " ") in cat_blob for a in alias_list):
                        boosts[i] += info.get("boost", 0.1)
                    else:
                        boosts[i] += info.get("boost", 0.1) * 0.6
        score_semantic = cosine
        score_keyword = boosts
        score_final = score_semantic + score_keyword
        idx_sorted = np.argsort(-score_final)[:top_n]
        results = []
        for rank, idx in enumerate(idx_sorted):
            row = self.df.iloc[idx]
            results.append({
                "rank": rank + 1,
                "id": safe_get(row, "id") or None,
                "categoria": safe_get(row, "categoria"),
                "aliases": safe_get(row, "aliases") or None,
                "score_semantic": float(score_semantic[idx]),
                "score_keyword": float(score_keyword[idx]),
                "score_final": float(score_final[idx]),
                "is_best": rank == 0
            })
        return results

# --- App con 'servers' explícito en OpenAPI ---
app = FastAPI(
    title="Clasificador Ontológico (lema + acepción)",
    version="1.1.1",
    description="API para clasificar definiciones en categorías leyendo CSV con columnas id,categoria,aliases.",
    servers=[{"url": PUBLIC_BASE_URL}],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fuerza `servers` también en /openapi.json, por si algún servidor lo omite
def custom_openapi():
    if app.openapi_schema:
        # asegúrate de que 'servers' esté presente
        app.openapi_schema["servers"] = [{"url": PUBLIC_BASE_URL}]
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["servers"] = [{"url": PUBLIC_BASE_URL}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

INDEX = CategoryIndex(CATEGORIES_CSV)

class ClassifyRequest(BaseModel):
    lema: str
    acepcion: str
    top_n: Optional[int] = 10
    split_acepciones: Optional[bool] = True

class ClassifyResponseItem(BaseModel):
    rank: int
    id: Optional[str] = None
    categoria: str
    aliases: Optional[str] = None
    score_semantic: float
    score_keyword: float
    score_final: float
    is_best: bool

class ClassifyResponse(BaseModel):
    lema: str
    acepcion: str
    query_used: str
    results: List[ClassifyResponseItem]

class MultiAcepcionResponse(BaseModel):
    lema: str
    items: List[ClassifyResponse]

@app.post("/classify", response_model=MultiAcepcionResponse, summary="Clasifica una o varias acepciones")
def classify(req: ClassifyRequest):
    if not req.lema or not req.acepcion:
        raise HTTPException(status_code=400, detail="Faltan 'lema' o 'acepcion'.")
    acepciones = [req.acepcion]
    if req.split_acepciones and "||" in req.acepcion:
        acepciones = split_acepciones_respecting_quotes(req.acepcion)
        acepciones = [a for a in acepciones if a.strip()]
    items = []
    for ac in acepciones:
        query = f"{req.lema.strip()}. {ac.strip()}"
        ranked = INDEX.rank(query, TAXON_HINTS, top_n=req.top_n or 10)
        items.append(ClassifyResponse(
            lema=req.lema,
            acepcion=ac,
            query_used=query,
            results=[ClassifyResponseItem(**r) for r in ranked]
        ))
    return MultiAcepcionResponse(lema=req.lema, items=items)

@app.post("/reload", summary="Recarga el CSV de categorías")
def reload_index():
    try:
        INDEX.reload()
        return {"ok": True, "num_categorias": len(INDEX.df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", summary="Estado básico")
def health():
    return {"ok": True, "model": MODEL_NAME, "num_categorias": len(INDEX.df)}

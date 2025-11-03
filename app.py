# app.py — OntoClassifier sin pandas (para Render)
import os, csv, re
import numpy as np
from fastapi import FastAPI, Query, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

# ============================================================
# CONFIGURACIÓN BÁSICA
# ============================================================
API_KEY = os.getenv("ONTO_API_KEY", "clave_publica_demo")
CSV_PATH = os.getenv("ONTO_CSV_PATH", "categorias.csv")

app = FastAPI(
    title="OntoClassifier",
    version="1.0.0",
    description="Clasificador ontológico híbrido (cosine + fuzzy + overlap)",
)

# Permitir CORS para pruebas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# UTILIDADES
# ============================================================

def main_label(txt: str) -> str:
    """Quita contenido alternativo tras / o paréntesis."""
    txt = re.split(r"[/(]", txt)[0].strip()
    return txt.lower()

def text_clean(t: str) -> str:
    return re.sub(r"[^a-záéíóúüñ\s]", "", t.lower()).strip()

# ============================================================
# CARGA DE CATEGORÍAS
# ============================================================

class OntoIndex:
    def __init__(self):
        self.model = None
        self.cat_emb = None
        self.categorias_main = []
        self.categorias_full = []
        self.ids = []

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def load(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise RuntimeError(f"No se encontró el CSV: {csv_path}")

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows or not {"id", "categoria"}.issubset(rows[0].keys()):
            raise RuntimeError("El CSV debe tener columnas 'id' y 'categoria'.")

        for r in rows:
            self.ids.append(int(r["id"]))
            cat = str(r["categoria"]).strip()
            self.categorias_full.append(cat)
            self.categorias_main.append(main_label(cat))

        self._load_model()
        emb = self.model.encode(self.categorias_main, normalize_embeddings=True)
        self.cat_emb = np.asarray(emb, dtype=np.float32)

# Instancia global
onto = OntoIndex()
onto.load(CSV_PATH)

# ============================================================
# CLASIFICADOR HÍBRIDO
# ============================================================

def classify_text(texto: str, top=5, k=30, decision_thresh=0.62,
                  w_cos=0.6, w_fuzzy=0.3, w_ov=0.1):
    txt = text_clean(texto)
    if not txt:
        return {"input": texto, "label_id": None, "label_text": "OUT_OF_SCOPE", "confidence": "low"}

    emb = onto.model.encode([txt], normalize_embeddings=True)
    sims = np.dot(onto.cat_emb, emb[0])

    idx_top = np.argsort(sims)[::-1][:k]
    candidatos = [(i, float(sims[i])) for i in idx_top]

    scores = []
    for i, sc_cos in candidatos:
        cat = onto.categorias_main[i]
        fz = fuzz.token_set_ratio(txt, cat) / 100
        ov = len(set(txt.split()) & set(cat.split())) / max(1, len(set(txt.split())))
        score = w_cos * sc_cos + w_fuzzy * fz + w_ov * ov
        scores.append((i, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    best_i, best_score = scores[0]
    conf = "high" if best_score >= decision_thresh + 0.05 else (
            "medium" if best_score >= decision_thresh - 0.05 else "low")

    result = {
        "input": texto,
        "label_id": onto.ids[best_i],
        "label_text": onto.categorias_full[best_i],
        "confidence": conf,
        "score_final": round(best_score, 3),
        "alternativas": [
            {"label": onto.categorias_full[i], "score": round(s, 3)}
            for i, s in scores[:top]
        ],
    }

    if best_score < decision_thresh:
        result["label_text"] = "OUT_OF_SCOPE"
    return result

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "ok": True,
        "message": "OntoClassifier online",
        "try": ["/health", "/docs", "/openapi.json", "/classify?q=ejemplo"]
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "rows": len(getattr(onto, "ids", [])),
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    }

@app.get("/classify")
def classify(
    q: str = Query(..., description="Texto o acepción a clasificar"),
    x_api_key: str = Header(None),
    top: int = 5,
    k: int = 30,
    decision_thresh: float = 0.62,
    w_cos: float = 0.6,
    w_fuzzy: float = 0.3,
    w_ov: float = 0.1,
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key inválida")

    return classify_text(q, top=top, k=k, decision_thresh=decision_thresh,
                         w_cos=w_cos, w_fuzzy=w_fuzzy, w_ov=w_ov)

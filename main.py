import os
import hashlib
import json
import re
from typing import Optional, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pymilvus import (
    connections, utility, Collection,
    FieldSchema, CollectionSchema, DataType
)

from openai import OpenAI

load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "phrases")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "64"))

# fake | openai
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "fake").strip().lower()

# RRF params
RRF_K = int(os.getenv("RRF_K", "60"))                 # constante típica
RRF_CANDIDATES_MULT = int(os.getenv("RRF_MULT", "5")) # top_k * MULT por cada ranking (lex/sem)

app = FastAPI(title="FastAPI + Milvus (Vector DB)")


# ---------- Requests ----------
class IngestRequest(BaseModel):
    frases: list[str] = Field(..., min_length=1)
    metadatos: Optional[dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


# ---------- Fake embeddings (para desarrollo / mock) ----------
def fake_embed(text: str) -> list[float]:
    """
    Embedding determinista de prueba.
    Permite probar todo el flujo sin IA real.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vec: list[float] = []
    b = h * ((EMBEDDING_DIM * 4 // len(h)) + 1)

    for i in range(EMBEDDING_DIM):
        chunk = b[i * 4:(i + 1) * 4]
        n = int.from_bytes(chunk, "little", signed=False)
        vec.append((n % 100000) / 100000.0)

    return vec

# ---------- OpenAI embeddings (preparado para producción) ----------
def openai_embed(text: str) -> list[float]:
    """
    Genera embeddings reales usando OpenAI.
    Requiere permisos model.request en la key / proyecto.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="Falta OPENAI_API_KEY en el .env")

    client = OpenAI(api_key=api_key)

    response = client.embeddings.create(
        model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        input=text
    )

    vec = response.data[0].embedding

    # Seguridad: si la colección está creada con otra dimensión, que falle con un mensaje claro
    if len(vec) != EMBEDDING_DIM:
        raise HTTPException(
            status_code=500,
            detail=f"Dimensión embedding OpenAI ({len(vec)}) != EMBEDDING_DIM ({EMBEDDING_DIM}). "
                   f"Cambia EMBEDDING_DIM o usa una colección distinta."
        )

    return vec

# cambiamos de función dependiendo del método que utilicemos
def embed(text: str) -> list[float]:
    """
    Punto único de generación de embeddings.
    Cambiamos EMBEDDING_PROVIDER en .env y el resto del código no se toca.
    """
    if EMBEDDING_PROVIDER == "openai":
        return openai_embed(text)
    return fake_embed(text)


# ---------- Milvus helpers ----------
def connect_milvus():
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)


def get_or_create_collection() -> Collection:
    connect_milvus()

    if utility.has_collection(COLLECTION_NAME):
        col = Collection(COLLECTION_NAME)
        col.load()
        return col

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="meta", dtype=DataType.VARCHAR, max_length=2048),
    ]
    schema = CollectionSchema(fields, description="Phrases + embeddings")
    col = Collection(COLLECTION_NAME, schema=schema)

    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200}
    }
    col.create_index(field_name="embedding", index_params=index_params)
    col.load()
    return col


@app.on_event("startup")
def startup():
    get_or_create_collection()


# ---------- RRF helpers ----------
def _tokenize(q: str) -> list[str]:
    # tokens simples: palabras/nums (evita comillas y cosas raras)
    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9]+", q.lower())
    # quita tokens muy cortos tipo "a", "y" (opcional)
    return [t for t in tokens if len(t) >= 2]


def _lexical_candidates(col: Collection, query: str, limit: int) -> list[dict]:
    """
    Recupera candidatos léxicos en Milvus (expr LIKE).
    Mejora respecto a tu versión: usa OR por tokens para no depender de la frase exacta.
    """
    q_escaped = query.replace('"', '\\"')
    tokens = _tokenize(q_escaped)

    if not tokens:
        expr = f'text like "%{q_escaped}%"'
    else:
        # OR: si aparece alguna palabra ya vale como candidato
        parts = []
        for t in tokens:
            safe_t = t.replace('"', '\\"')
            parts.append(f'text like "%{safe_t}%"')
        expr = " or ".join(parts)

    docs = col.query(
        expr=expr,
        output_fields=["id", "text", "meta"],
        limit=limit
    )

    # Ranking léxico simple (en Python): cuenta ocurrencias de tokens
    if tokens:
        def lex_score(d: dict) -> int:
            txt = (d.get("text") or "").lower()
            return sum(txt.count(t) for t in tokens)

        docs.sort(key=lambda d: lex_score(d), reverse=True)

    return docs


def _semantic_candidates(col: Collection, query: str, limit: int) -> list[dict]:
    qvec = embed(query)
    results = col.search(
        data=[qvec],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=limit,
        output_fields=["id", "text", "meta"]
    )

    hits: list[dict] = []
    for h in results[0]:
        hits.append({
            "id": h.entity.get("id"),
            "text": h.entity.get("text"),
            "meta": h.entity.get("meta"),
            "semantic_score": float(h.score),
        })
    return hits


def _rrf_fuse(
    lexical_docs: list[dict],
    semantic_docs: list[dict],
    top_k: int,
    rrf_k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion:
    score(d) = Σ 1 / (rrf_k + rank_i(d))
    """
    fused: dict[int, dict] = {}

    # ranks empiezan en 1
    for rank, d in enumerate(lexical_docs, start=1):
        doc_id = int(d["id"])
        entry = fused.setdefault(doc_id, {
            "id": doc_id,
            "text": d.get("text"),
            "meta": d.get("meta"),
            "lexical_rank": None,
            "semantic_rank": None,
            "lexical_score": 0.0,
            "semantic_score": None,
            "rrf_score": 0.0,
        })
        entry["lexical_rank"] = rank
        entry["lexical_score"] += 1.0 / (rrf_k + rank)
        entry["rrf_score"] += 1.0 / (rrf_k + rank)

    for rank, d in enumerate(semantic_docs, start=1):
        doc_id = int(d["id"])
        entry = fused.setdefault(doc_id, {
            "id": doc_id,
            "text": d.get("text"),
            "meta": d.get("meta"),
            "lexical_rank": None,
            "semantic_rank": None,
            "lexical_score": 0.0,
            "semantic_score": None,
            "rrf_score": 0.0,
        })
        entry["semantic_rank"] = rank
        entry["semantic_score"] = d.get("semantic_score")
        entry["rrf_score"] += 1.0 / (rrf_k + rank)

        # si el doc ya existía por lexical pero le faltaban campos, rellena
        if entry.get("text") is None:
            entry["text"] = d.get("text")
        if entry.get("meta") is None:
            entry["meta"] = d.get("meta")

    # orden final por rrf_score desc
    ordered = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
    return ordered[:top_k]


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {
        "ok": True,
        "milvus": f"{MILVUS_HOST}:{MILVUS_PORT}",
        "collection": COLLECTION_NAME,
        "dim": EMBEDDING_DIM,
        "embedding_provider": EMBEDDING_PROVIDER
    }


@app.post("/collection/create")
def create_collection():
    col = get_or_create_collection()
    return {"ok": True, "collection": col.name, "dim": EMBEDDING_DIM}


@app.post("/ingest")
def ingest(req: IngestRequest):
    col = get_or_create_collection()

    metas = req.metadatos or {}
    texts = req.frases

    # usamos embed() (fake u openai según .env)
    vectors = [embed(t) for t in texts]

    # meta como JSON string (más seguro que str())
    meta_str = json.dumps(metas, ensure_ascii=False)
    meta_strs = [meta_str for _ in texts]

    res = col.insert([texts, vectors, meta_strs])
    col.flush()

    ids = [int(x) for x in list(res.primary_keys)]
    return {"ok": True, "inserted_count": len(texts), "ids": ids}


@app.post("/search/lexical")
def search_lexical(req: SearchRequest):
    col = get_or_create_collection()
    q = req.query.replace('"', '\\"')
    expr = f'text like "%{q}%"'
    docs = col.query(expr=expr, output_fields=["id", "text", "meta"], limit=req.top_k)
    return {"ok": True, "type": "lexical", "results": docs}


@app.post("/search/semantic")
def search_semantic(req: SearchRequest):
    col = get_or_create_collection()

    # aquí también embed()
    qvec = embed(req.query)

    results = col.search(
        data=[qvec],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=req.top_k,
        output_fields=["id", "text", "meta"]
    )

    hits = []
    for h in results[0]:
        hits.append({
            "id": h.entity.get("id"),
            "text": h.entity.get("text"),
            "meta": h.entity.get("meta"),
            "score": float(h.score),
        })

    return {"ok": True, "type": "semantic", "results": hits}


@app.post("/search/hybrid")
def search_hybrid(req: SearchRequest):
    col = get_or_create_collection()

    # cuántos candidatos cogemos por cada ranking
    per_list = max(req.top_k * RRF_CANDIDATES_MULT, 20)

    lexical_docs = _lexical_candidates(col, req.query, limit=per_list)
    semantic_docs = _semantic_candidates(col, req.query, limit=per_list)

    fused = _rrf_fuse(
        lexical_docs=lexical_docs,
        semantic_docs=semantic_docs,
        top_k=req.top_k,
        rrf_k=RRF_K
    )

    # Por compatibilidad, devolvemos "score" como el score final (RRF)
    results = []
    for d in fused:
        results.append({
            "id": d["id"],
            "text": d.get("text"),
            "meta": d.get("meta"),
            "score": float(d["rrf_score"]),          # score final
            "rrf_score": float(d["rrf_score"]),
            "semantic_score": d.get("semantic_score"),
            "lexical_score": float(d.get("lexical_score", 0.0)),
            "lexical_rank": d.get("lexical_rank"),
            "semantic_rank": d.get("semantic_rank"),
        })

    return {
        "ok": True,
        "type": "hybrid_rrf",
        "rrf_k": RRF_K,
        "per_list_candidates": per_list,
        "lexical_count": len(lexical_docs),
        "semantic_count": len(semantic_docs),
        "results": results
    }

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int):
    col = get_or_create_collection()
    col.delete(expr=f"id == {doc_id}")
    col.flush()
    return {"ok": True, "deleted_id": doc_id}

class UpdateRequest(BaseModel):
    text: Optional[str] = None
    metadatos: Optional[dict[str, Any]] = None


@app.put("/documents/{doc_id}")
def update_document(doc_id: int, req: UpdateRequest):
    col = get_or_create_collection()

    # 1) comprobar que existe
    results = col.query(expr=f"id == {doc_id}", output_fields=["id", "text", "meta"])
    if not results:
        raise HTTPException(status_code=404, detail="Documento no encontrado")

    current = results[0]
    new_text = req.text if req.text is not None else current["text"]

    # 2) nuevo embedding con el provider actual (fake/openai)
    new_vec = embed(new_text)

    # 3) metadatos
    new_meta = (
        json.dumps(req.metadatos, ensure_ascii=False)
        if req.metadatos is not None
        else current.get("meta", "{}")
    )

    # 4) update en Milvus = delete + insert (id cambia por auto_id=True)
    col.delete(expr=f"id == {doc_id}")
    col.flush()

    res = col.insert([[new_text], [new_vec], [new_meta]])
    col.flush()

    new_id = int(list(res.primary_keys)[0])
    return {"ok": True, "old_id": doc_id, "new_id": new_id, "text": new_text}

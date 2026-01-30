import os
import hashlib
import json
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
    q = req.query.replace('"', '\\"')

    candidates = col.query(
        expr=f'text like "%{q}%"',
        output_fields=["id"],
        limit=max(req.top_k * 5, 20)
    )
    ids = [c["id"] for c in candidates]
    if not ids:
        return {"ok": True, "type": "hybrid", "results": [], "candidate_count": 0}

    # embed() también aquí (consistencia)
    qvec = embed(req.query)
    expr_ids = f"id in {ids}"

    results = col.search(
        data=[qvec],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=req.top_k,
        expr=expr_ids,
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

    return {"ok": True, "type": "hybrid", "candidate_count": len(ids), "results": hits}

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
